import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import tyro
from src.vectorised_training_util import (
    Adam,
    AtariTransition,
    FrameReplayBuffer,
    FrameStackBuffer,
    LazyRMSNorm,
    NormaliseAtari,
    NvcompReplayBuffer,
    ValidActionMask,
    stack_models,
    to_stacked_channels_last,
)
from torch.func import grad_and_value, vmap

from src.atari.vectorised_environments import make_atari_vector_env

torch.set_float32_matmul_precision("high")

FRAMESTACK = 4


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    env_id: str = "ALE/Pong-v5"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """per-agent env steps; each of the N independent pairs runs this many (N pairs == N seeds)"""
    num_agent_env_pairs: int = 8
    """number of independent agent-env pairs (== num_seeds)"""
    learning_rate: float = 1e-4
    buffer_size: int = 1_000_000
    """per-seed replay capacity"""
    replay_compress: str = "nvcomp"
    """replay obs backend: "nvcomp" (zstd blobs on top of 4-bit) or "bitpack" (dense 4-bit)"""
    gamma: float = 0.99
    target_network_frequency: int = 2500
    """updates between target-network syncs"""
    batch_size: int = 32
    """per-seed minibatch size"""
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_transitions: int = 1_000_000
    """per-agent transitions over which epsilon decays"""
    learning_starts: int = 12_500
    train_frequency: int = 4
    max_grad_norm: float = 1.0

    compile: bool = True
    channels_last: bool = True
    """lay stacked conv weights + obs out NHWC-per-slice so cudnn skips the per-step transpose"""
    metrics_path: str = ""
    detect_syncs: bool = False
    """if set, torch.cuda.set_sync_debug_mode("warn") - warns on every implicit host<->device sync"""


class QNetwork(nn.Module):
    """Q-value network: NormaliseAtari -> Nature CNN with RMSNorm, no bias."""

    def __init__(self, num_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            NormaliseAtari(),
            nn.LazyConv2d(32, 8, stride=4, bias=False),
            LazyRMSNorm(),
            nn.ReLU(),
            nn.LazyConv2d(64, 4, stride=2, bias=False),
            LazyRMSNorm(),
            nn.ReLU(),
            nn.LazyConv2d(64, 3, stride=1, bias=False),
            LazyRMSNorm(),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(512, bias=False),
            LazyRMSNorm(),
            nn.ReLU(),
            nn.LazyLinear(num_actions),
        )
        self._setup_network(num_actions)

    @torch.no_grad()
    def _setup_network(self, num_actions: int) -> None:
        out = self.network(torch.zeros((1, 4, 84, 84)))
        assert out.shape[-1] == num_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNVecAgents:
    """N independent Q-networks + targets stacked into one param TensorDict, driven under vmap."""

    def __init__(self, args: Args, num_actions: int, framestack: int, device: torch.device, num_seeds: int):
        self.args = args
        self.device = device
        self.num_seeds = num_seeds
        self.num_actions = num_actions
        self.framestack = framestack
        self.action_counts = torch.tensor([num_actions] * num_seeds, dtype=torch.long, device=device)
        self.action_mask = ValidActionMask(self.action_counts, num_actions).to(device)

        # channels_last so cudnn doesn't spend time swapping channels: NCHW<->NHWC transpose.
        self._to_cl = to_stacked_channels_last if args.channels_last else (lambda t: t)
        models = [QNetwork(num_actions).to(device) for _ in range(num_seeds)]
        self.params, self.base_model = stack_models(models)
        self.target_params = self.params.clone()
        if args.channels_last:
            for k in list(self.params.keys(include_nested=True, leaves_only=True)):
                if self.params.get(k).dim() == 5:  # only conv weights are rank-5 stacks
                    self.params.set(k, self._to_cl(self.params.get(k)))
                    self.target_params.set(k, self._to_cl(self.target_params.get(k)))

        # Optimiser state
        self.opt_state = vmap(partial(Adam.init, device=device, lr=args.learning_rate, max_norm=args.max_grad_norm))(
            self.params
        )

        self.replay = self._make_replay()

        self._build_forward_and_backward()

        self.update_count = 0
        self.agent_decisions = 0
        if args.compile:
            self._warmup_compiled()

    def _make_replay(self):
        """Build the replay buffer (override to swap in a different buffer)."""
        buffer_cls = {"bitpack": FrameReplayBuffer, "nvcomp": NvcompReplayBuffer}[self.args.replay_compress]
        return buffer_cls(
            capacity=self.args.buffer_size,
            num_seeds=self.num_seeds,
            batch_size=self.args.batch_size,
            framestack=self.framestack,
            device=self.device,
        )

    def _warmup_compiled(self) -> None:
        """Run the compiled policy + train step on dummy data so compilation and CUDA-graph
        capture happen here, not inside the training timer. Initial weights/optimiser are
        snapshotted and restored, so warmup does not perturb the run."""

        params0, target0, opt0 = self.params.clone(), self.target_params.clone(), self.opt_state.clone()

        # Warmup inference
        frame = torch.zeros(self.num_seeds, 1, 84, 84, dtype=torch.uint8, device=self.device)
        no_done = torch.zeros(self.num_seeds, dtype=torch.bool, device=self.device)
        for _ in range(20):
            torch.compiler.cudagraph_mark_step_begin()
            self.select_action(frame, no_done, 0.0)

        # Warmup update
        bs = self.args.batch_size
        dummy = AtariTransition(
            observations=torch.zeros(self.num_seeds, bs, 4, 84, 84, dtype=torch.uint8, device=self.device),
            next_observations=torch.zeros(self.num_seeds, bs, 4, 84, 84, dtype=torch.uint8, device=self.device),
            actions=torch.zeros(self.num_seeds, bs, dtype=torch.int64, device=self.device),
            rewards=torch.zeros(self.num_seeds, bs, dtype=torch.float32, device=self.device),
            terminated=torch.zeros(self.num_seeds, bs, dtype=torch.bool, device=self.device),
            truncated=torch.zeros(self.num_seeds, bs, dtype=torch.bool, device=self.device),
            batch_size=[self.num_seeds, bs],
        )
        for _ in range(20):
            torch.compiler.cudagraph_mark_step_begin()
            self.update(dummy)

        # Restore initial state (in-place so CUDA-graph buffers stay valid).
        self.params.update_(params0)
        self.target_params = target0
        self.opt_state.mu.update_(opt0.mu)
        self.opt_state.nu.update_(opt0.nu)
        self.opt_state.step = opt0.step.clone()
        self.frame_stack.buf.zero_()  # drop the dummy frames warmup rolled in
        self.update_count = 0
        self.agent_decisions = 0

    def _build_forward_and_backward(self) -> None:
        gamma = self.args.gamma
        to_cl = self._to_cl

        self.frame_stack = FrameStackBuffer(self.num_seeds, self.framestack, 84, 84).to(self.device)

        base_model = self.base_model
        def forward_single(params, obs):
            with params.to_module(base_model):
                return base_model(obs)

        frame_stack = self.frame_stack
        mask_logits = self.action_mask

        def policy_fn(new_frame, done, params):
            obs = frame_stack.update(new_frame, done)  # in-place stack roll, captured in the graph
            obs_b = obs.float().unsqueeze(1)  # (N, 1, C, H, W)
            obs_b = to_cl(obs_b)  # match the channels_last weights; skip the swap
            q = vmap(forward_single)(params, obs_b).squeeze(1)
            q = mask_logits(q)
            return q.argmax(dim=-1)

        def _train_single(params, target_params, opt_state, obs, next_obs, actions, rewards, dones, action_mask):
            def loss_fn(p):
                q = forward_single(p, obs)
                q_sel = q.gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q = forward_single(target_params, next_obs)
                next_q = next_q.masked_fill(~action_mask, -torch.inf)
                td_target = rewards + gamma * next_q.max(dim=1)[0] * (1 - dones.float())
                return F.mse_loss(q_sel, td_target.detach())

            grads, loss = grad_and_value(loss_fn)(params)
            grads = opt_state.clip_grads(grads)
            adam_step, new_opt_state = opt_state.update(grads)
            new_params = new_opt_state.apply_grads(params, adam_step)
            return loss, new_params, new_opt_state

        if self.args.compile:
            mode = "reduce-overhead"
            self._policy_fn = torch.compile(policy_fn, mode=mode, fullgraph=True)
            self._train_step_fn = torch.compile(vmap(_train_single), mode=mode)
        else:
            self._policy_fn = policy_fn
            self._train_step_fn = vmap(_train_single)

    def get_exploration_rate(self, frame_no: int) -> float:
        return max(
            self.args.end_e,
            self.args.start_e + (self.args.end_e - self.args.start_e) * (frame_no / self.args.exploration_transitions),
        )

    @torch.no_grad()
    def select_action(self, new_frame: torch.Tensor, done: torch.Tensor, epsilon: float) -> torch.Tensor:
        greedy = self._policy_fn(new_frame, done, self.params)
        rand = torch.floor(torch.rand(greedy.shape, device=greedy.device) * self.action_counts).long()
        use_policy = torch.rand(greedy.shape, device=greedy.device).gt(epsilon)
        self.agent_decisions += 1
        return torch.where(use_policy, greedy, rand)

    def store_transition(self, obs, action, next_obs, reward, terminated, truncated) -> None:
        self.replay.add(obs, action, reward, terminated, truncated)

    def update(self, batch) -> torch.Tensor:
        loss, new_params, new_opt_state = self._train_step_fn(
            self.params,
            self.target_params,
            self.opt_state,
            self._to_cl(batch.observations.float()),
            self._to_cl(batch.next_observations.float()),
            batch.actions.long(),
            batch.rewards,
            batch.terminated,
            self.action_mask.mask,
        )
        self.params.update_(new_params)
        self.opt_state.mu.update_(new_opt_state.mu)
        self.opt_state.nu.update_(new_opt_state.nu)
        self.opt_state.step = new_opt_state.step.clone()
        self.update_count += 1
        if self.update_count % self.args.target_network_frequency == 0:
            self.target_params = self.params.clone()
        return loss.detach()


def _log_progress(pbar, args, metrics_file, global_step, burnin_step, start_time, avg_returns, epsilon) -> None:
    speed = (global_step - burnin_step) / (time.time() - start_time)
    avg_ret = float(np.mean(avg_returns)) if avg_returns else float("nan")
    pbar.set_description(f"speed: {speed:5.0f} sps, eps: {epsilon:.2f}, returns: {avg_ret:5.2f}")
    if metrics_file is not None:
        metrics_file.write(
            json.dumps(
                {
                    "global_step": int(global_step),
                    "episode_return": None if np.isnan(avg_ret) else avg_ret,
                    "speed_sps": float(speed),
                }
            )
            + "\n"
        )
        metrics_file.flush()


def main(args, metrics_file):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = make_atari_vector_env(env_id=args.env_id, num_envs=args.num_agent_env_pairs, frameskip=5, stack_num=1)
    num_actions = int(envs.single_action_space.n)
    agents = DQNVecAgents(args, num_actions=num_actions, framestack=4, device=device, num_seeds=args.num_agent_env_pairs)
    frame_np, _ = envs.reset(seed=args.seed)
    frame = torch.as_tensor(frame_np, device=device, dtype=torch.uint8)
    done = torch.zeros(args.num_agent_env_pairs, dtype=torch.bool, device=device)

    # per-agent step budget drives the schedules.
    avg_returns = deque(maxlen=100)
    per_agent_steps = args.total_timesteps
    start_time = time.time()
    burnin_step = None

    # "warn" prints a stack at each implicit host<->device sync (expected one: the action .cpu()).
    if args.detect_syncs:
        torch.cuda.set_sync_debug_mode("warn")

    pbar = tqdm.tqdm(range(per_agent_steps))
    for step in pbar:
        if step == args.learning_starts:
            burnin_step = step
            start_time = time.time()

        epsilon = agents.get_exploration_rate(step)
        torch.compiler.cudagraph_mark_step_begin()
        action = agents.select_action(frame, done, epsilon).clone()

        next_frame_np, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        agents.store_transition(frame_np, action.cpu().numpy(), next_frame_np, reward, terminated, truncated)
        frame_np = next_frame_np
        frame = torch.as_tensor(next_frame_np, device=device, dtype=torch.uint8)
        done = torch.as_tensor(terminated | truncated, device=device)

        if "episode" in info:
            avg_returns.extend(info["episode"]["r"][info["_episode"]])

        if step > args.learning_starts and step % args.train_frequency == 0:
            torch.compiler.cudagraph_mark_step_begin()
            agents.update(agents.replay.sample())

        if burnin_step is not None and step % 2500 == 0:
            _log_progress(pbar, args, metrics_file, step, burnin_step, start_time, avg_returns, epsilon)

    envs.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.metrics_path, "Must specify --metrics-path to save results."
    metrics_file = open(args.metrics_path, "w")
    main(args, metrics_file)
