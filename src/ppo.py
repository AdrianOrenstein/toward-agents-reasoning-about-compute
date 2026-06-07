# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
#
# Vectorised-training PPO: N independent agent-env pairs trained in one process. Unlike the
# baseline (one shared network over N copies of a game), here each of the N envs has its OWN
# network, optimiser and rollout. The N networks are stacked into one TensorDict and driven
# with torch.func.vmap, so inference and the PPO update run as single batched kernels over the
# stack. This is the design from src/agents/ppo/agent_design.py, reduced to one file.
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import contextlib
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import tqdm
import tyro
from src.vectorised_training_util import (
    Adam,
    FrameStackBuffer,
    LazyRMSNorm,
    NormaliseAtari,
    ValidActionMask,
    stack_models,
)
from torch.func import grad_and_value, vmap

# Repo ALE vector interface + the shared vectorised-training building blocks.
from src.atari.vectorised_environments import make_atari_vector_env

torch.set_float32_matmul_precision("high")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    env_id: str = "ALE/Pong-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total env steps (summed across the N pairs)"""
    num_agent_env_pairs: int = 8
    """number of independent agent-env pairs (== num_seeds)"""
    learning_rate: float = 2.5e-4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    measure_burnin: int = 3
    compile: bool = False
    cudagraphs: bool = False
    metrics_path: str = ""
    detect_syncs: bool = False
    """if set, torch.cuda.set_sync_debug_mode("warn") - warns on every implicit host<->device sync"""

    # filled at runtime
    minibatch_size: int = 0
    num_iterations: int = 0


class PPOActorCritic(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.trunk = nn.Sequential(
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
        )
        self.actor = nn.LazyLinear(num_actions)
        self.critic = nn.LazyLinear(1)
        self._setup_network()

    @torch.no_grad()
    def _setup_network(self) -> None:
        dummy = torch.zeros((1, 4, 84, 84))
        logits, value = self(dummy)
        assert logits.ndim == 2
        assert value.shape == (1,)

    def forward(self, x: torch.Tensor):
        hidden = self.trunk(x)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class PPOVecAgent:
    """N independent actor-critics stacked into one param TensorDict, driven under vmap."""

    def __init__(self, args: Args, num_actions: int, framestack: int, device: torch.device, num_seeds: int):
        self.args = args
        self.device = device
        self.num_seeds = num_seeds
        self.num_actions = num_actions
        self.action_counts = torch.tensor([num_actions] * num_seeds, dtype=torch.long, device=device)
        self.action_mask = ValidActionMask(self.action_counts, num_actions).to(device)
        self.frame_stack = FrameStackBuffer(num_seeds, framestack, 84, 84).to(device)

        models = [PPOActorCritic(num_actions).to(device) for _ in range(num_seeds)]
        self.params, self.base_model = stack_models(models)
        self.opt_state = vmap(partial(Adam.init, device=device, lr=args.learning_rate, max_norm=args.max_grad_norm))(
            self.params
        )
        # one param slice run functionally through the zeroed base model; wrap with vmap for all N.
        base_model = self.base_model

        def forward_single(params, obs):
            with params.to_module(base_model):
                return base_model(obs)

        self._forward_single = forward_single
        self._build_compiled_functions()

        if args.compile:
            self._warmup_compiled()

    def _warmup_compiled(self) -> None:
        """Run the compiled rollout + update on dummy data so compilation and CUDA-graph capture
        happen here, not inside the timed loop. Weights/optimiser are snapshotted and restored
        in-place (so CUDA-graph buffers stay valid), so warmup does not perturb the run. Mirrors
        DQNVecAgents._warmup_compiled so the sps measurement excludes graph-capture overhead."""
        args = self.args
        S, fs, ns, mb = self.num_seeds, self.frame_stack.framestack, args.num_steps, args.minibatch_size
        params0, opt0 = self.params.clone(), self.opt_state.clone()

        obs = torch.zeros(S, fs, 84, 84, dtype=torch.uint8, device=self.device)
        rew = torch.zeros(S, ns, dtype=torch.float32, device=self.device)
        done = torch.zeros(S, ns, dtype=torch.float32, device=self.device)
        val = torch.zeros(S, ns, dtype=torch.float32, device=self.device)
        nxt_val = torch.zeros(S, dtype=torch.float32, device=self.device)
        nxt_done = torch.zeros(S, dtype=torch.float32, device=self.device)
        mb_obs = torch.zeros(S, mb, fs, 84, 84, dtype=torch.uint8, device=self.device)
        mb_z = torch.zeros(S, mb, dtype=torch.float32, device=self.device)
        mb_a = torch.zeros(S, mb, dtype=torch.int64, device=self.device)

        for _ in range(10):
            torch.compiler.cudagraph_mark_step_begin()
            self.select_action(obs)
            torch.compiler.cudagraph_mark_step_begin()
            self.get_value(obs)
            torch.compiler.cudagraph_mark_step_begin()
            self.compute_gae(rew, done, val, nxt_val, nxt_done)
            torch.compiler.cudagraph_mark_step_begin()
            self.update_minibatch(mb_obs, mb_a, mb_z, mb_z, mb_z, mb_z)

        self.params.update_(params0)
        self.opt_state.mu.update_(opt0.mu)
        self.opt_state.nu.update_(opt0.nu)
        self.opt_state.step = opt0.step.clone()

    def _make_gae_single(self):
        args = self.args

        def _gae_single(rewards, dones, values, next_value, next_done):
            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
            for t in range(rewards.shape[0] - 1, -1, -1):
                if t == rewards.shape[0] - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[t]
                    next_values = values[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values
            return advantages, returns

        return _gae_single

    def _make_update_single(self):
        args = self.args
        forward_single = self._forward_single

        def _update_single(params, opt_state, obs, actions, old_logprobs, advantages, returns, old_values, action_mask):
            if args.norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            def loss_fn(p):
                logits, values = forward_single(p, obs)
                logits = logits.masked_fill(~action_mask, -torch.inf)
                log_probs_all = torch.log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                new_logprob = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
                entropy = -(probs * torch.nan_to_num(log_probs_all, neginf=0.0)).sum(dim=-1).mean()
                logratio = new_logprob - old_logprobs
                ratio = logratio.exp()
                pg_loss = torch.maximum(
                    -advantages * ratio,
                    -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                if args.clip_vloss:
                    value_delta = values - old_values
                    v_clipped = old_values + torch.clamp(value_delta, -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.maximum((values - returns).pow(2), (v_clipped - returns).pow(2)).mean()
                else:
                    v_loss = 0.5 * (values - returns).pow(2).mean()
                loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                return loss, (approx_kl, pg_loss, v_loss, entropy, clipfrac)

            grads, (loss, aux) = grad_and_value(loss_fn, has_aux=True)(params)
            grads = opt_state.clip_grads(grads)
            adam_step, new_opt_state = opt_state.update(grads)
            adam_step = new_opt_state.apply_weight_decay(adam_step, params)
            new_params = new_opt_state.apply_grads(params, adam_step)
            approx_kl, pg_loss, v_loss, entropy, clipfrac = aux
            return new_params, new_opt_state, loss, approx_kl, pg_loss, v_loss, entropy, clipfrac

        return _update_single

    def _build_compiled_functions(self) -> None:
        forward_single = self._forward_single

        def _sample_action_single(flat_params, obs, action_mask):
            logits, value = forward_single(flat_params, obs.unsqueeze(0))
            logits = logits.squeeze(0)
            logits = logits.masked_fill(~action_mask, -torch.inf)
            value = value.squeeze(0)
            gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            action = (logits + gumbels).argmax()
            log_probs = torch.log_softmax(logits, dim=-1)
            return action, log_probs[action], value

        def _value_single(flat_params, obs):
            _, value = forward_single(flat_params, obs.unsqueeze(0))
            return value.squeeze(0)

        self._sample_action_fn = vmap(_sample_action_single, randomness="different")
        self._value_fn = vmap(_value_single)
        self._gae_fn = vmap(self._make_gae_single())
        self._update_fn = vmap(self._make_update_single())

        if self.args.compile:
            mode = "reduce-overhead"
            self._sample_action_fn = torch.compile(self._sample_action_fn, mode=mode)
            self._value_fn = torch.compile(self._value_fn, mode=mode)
            self._gae_fn = torch.compile(self._gae_fn, mode=mode, fullgraph=True)
            self._update_fn = torch.compile(self._update_fn, mode=mode)

    @torch.no_grad()
    def select_action(self, obs):
        return self._sample_action_fn(self.params, obs, self.action_mask.mask)

    @torch.no_grad()
    def get_value(self, obs):
        return self._value_fn(self.params, obs)

    def compute_gae(self, rewards, dones, values, next_value, next_done):
        return self._gae_fn(rewards, dones, values, next_value, next_done)

    def set_learning_rate(self, lr: float) -> None:
        self.opt_state.lr = torch.full_like(self.opt_state.lr, lr).clone()

    def update_minibatch(self, obs, actions, old_logprobs, advantages, returns, old_values):
        new_params, new_opt_state, loss, approx_kl, pg_loss, v_loss, entropy, clipfrac = self._update_fn(
            self.params,
            self.opt_state,
            obs,
            actions,
            old_logprobs,
            advantages,
            returns,
            old_values,
            self.action_mask.mask,
        )
        with torch.no_grad():
            self.params.update_(new_params)
            self.opt_state.mu.update_(new_opt_state.mu)
            self.opt_state.nu.update_(new_opt_state.nu)
            self.opt_state.step = new_opt_state.step.clone()
        return {"loss": loss.detach(), "approx_kl": approx_kl.detach(), "entropy": entropy.detach()}


def _collect_rollout(agent, envs, obs, buf, args, device, avg_returns):
    """Run num_steps of the policy into the persistent buffers `buf`. Returns (obs, done) for the
    bootstrap, where obs is the on-GPU stacked observation after the last step."""
    done = None
    for step in range(args.num_steps):
        torch.compiler.cudagraph_mark_step_begin()
        action, logprob, value = agent.select_action(obs)
        buf["obs"][:, step].copy_(obs)
        buf["act"][:, step].copy_(action)
        buf["logp"][:, step].copy_(logprob)
        buf["val"][:, step].copy_(value)

        next_frame_np, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        done = torch.as_tensor(np.logical_or(terminated, truncated), device=device, dtype=torch.float32)
        buf["rew"][:, step].copy_(torch.as_tensor(reward, dtype=torch.float32, device=device))
        buf["done"][:, step].copy_(done)
        next_frame = torch.as_tensor(next_frame_np, device=device, dtype=torch.uint8)
        obs = agent.frame_stack.update(next_frame, done.bool())

        if "episode" in info:
            avg_returns.extend(info["episode"]["r"][info["_episode"]])
    return obs, done


def _log_progress(pbar, args, metrics_file, global_step, burnin_step, start_time, avg_returns) -> None:
    speed = (global_step - burnin_step) / (time.time() - start_time)
    avg_ret = float(np.mean(avg_returns)) if avg_returns else float("nan")
    pbar.set_description(f"speed: {speed:4.0f} sps, returns: {avg_ret:5.2f}")
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
    args.minibatch_size = args.num_steps // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.num_steps
    framestack = 4

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    num_seeds = args.num_agent_env_pairs

    envs = make_atari_vector_env(
        env_id=args.env_id,
        num_envs=num_seeds,
        frameskip=5,
        # the env sends one frame/step and the 4-stack is rebuilt on the GPU, cutting host->device traffic
        stack_num=1,
    )
    num_actions = int(envs.single_action_space.n)

    agent = PPOVecAgent(args, num_actions=num_actions, framestack=framestack, device=device, num_seeds=num_seeds)

    frame_np, _ = envs.reset(seed=args.seed)
    frame = torch.as_tensor(frame_np, device=device, dtype=torch.uint8)
    obs = agent.frame_stack.update(frame, torch.zeros(num_seeds, dtype=torch.bool, device=device))

    buf = {
        "obs": torch.empty((num_seeds, args.num_steps, framestack, 84, 84), dtype=torch.uint8, device=device),
        "act": torch.empty((num_seeds, args.num_steps), dtype=torch.int64, device=device),
        "logp": torch.empty((num_seeds, args.num_steps), dtype=torch.float32, device=device),
        "val": torch.empty((num_seeds, args.num_steps), dtype=torch.float32, device=device),
        "rew": torch.empty((num_seeds, args.num_steps), dtype=torch.float32, device=device),
        "done": torch.empty((num_seeds, args.num_steps), dtype=torch.float32, device=device),
    }

    avg_returns = deque(maxlen=100)
    global_step = 0
    start_time = time.time()
    global_step_burnin = None
    # Implicit host<->device syncs serialise CPU and GPU and silently kill RL throughput. "warn"
    # prints a stack at each one; the expected sync here is the per-step action .cpu().numpy().
    # See torch.cuda.set_sync_debug_mode. (Compile cost is excluded via measure_burnin.)
    if args.detect_syncs:
        torch.cuda.set_sync_debug_mode("warn")
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            agent.set_learning_rate(frac * args.learning_rate)

        obs, done = _collect_rollout(agent, envs, obs, buf, args, device, avg_returns)
        global_step += args.num_steps

        torch.compiler.cudagraph_mark_step_begin()
        next_value = agent.get_value(obs)
        advantages, returns = agent.compute_gae(buf["rew"], buf["done"], buf["val"], next_value, done)

        for _epoch in range(args.update_epochs):
            perm = torch.randperm(args.num_steps, device=device)
            for mb in perm.split(args.minibatch_size):
                torch.compiler.cudagraph_mark_step_begin()
                agent.update_minibatch(
                    obs=buf["obs"][:, mb],
                    actions=buf["act"][:, mb],
                    old_logprobs=buf["logp"][:, mb],
                    advantages=advantages[:, mb],
                    returns=returns[:, mb],
                    old_values=buf["val"][:, mb],
                )

        if global_step_burnin is not None and iteration % 10 == 0:
            _log_progress(pbar, args, metrics_file, global_step, global_step_burnin, start_time, avg_returns)

    envs.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    with open(args.metrics_path, "a") if args.metrics_path else contextlib.nullcontext() as metrics_file:
        main(args, metrics_file)
