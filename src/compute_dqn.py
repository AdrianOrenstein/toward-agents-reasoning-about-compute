import random
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from src.dqn import Args, DQNVecAgents, _log_progress
from src.vectorised_training_util import AtariTransition, FrameStackBuffer, pack_4bit, unpack_4bit
from torch.func import grad_and_value, vmap

from src.atari.vectorised_environments import make_atari_vector_env

# Reuse the replay's nvcomp codec (zstd over 4-bit) unchanged; only the SMDP blob bookkeeping is new.
try:
    import nvidia.nvcomp as _nvcomp
    from src.vectorised_training_util import _codec

    HAVE_NVCOMP = True
except Exception:
    HAVE_NVCOMP = False

torch.set_float32_matmul_precision("high")


@dataclass
class ComputeArgs(Args):
    exp_name: str = "compute_dqn"
    option_lengths: list[int] = field(default_factory=lambda: [1, 2, 4])
    """env steps each option commits to; the agent picks one per decision (e.g. [1, 2, 4])"""
    option_deliberation_cost: float = 0.01
    """0.01 is a default penalty that is dominated by any improvement in return (a cost on deliberating)"""


class OptionReplayBuffer:
    """Per-seed circular buffer of full SMDP transitions (obs, action, accumulated reward,
    next_obs, done). Each seed finishes options on its own schedule, so pointers advance per seed
    via a write mask. Obs frames are 4-bit packed; the obs backend is behind _init_obs_storage /
    _store_obs / _read_obs so a subclass can swap dense storage for nvcomp blobs."""

    def __init__(self, capacity, num_seeds, batch_size, framestack, device, frame_shape=(84, 84)):
        self._cap = capacity
        self._S = num_seeds
        self._bs = batch_size
        self._fs = framestack
        self._device = torch.device(device)
        self._h, w = frame_shape
        self._w_stored = w // 2
        self._numel = framestack * self._h * self._w_stored
        self._action = torch.zeros(num_seeds, capacity, dtype=torch.int64, device=self._device)
        self._reward = torch.zeros(num_seeds, capacity, dtype=torch.float32, device=self._device)
        self._done = torch.zeros(num_seeds, capacity, dtype=torch.bool, device=self._device)
        self._ptr = torch.zeros(num_seeds, dtype=torch.int64, device=self._device)
        self._size = torch.zeros(num_seeds, dtype=torch.int64, device=self._device)
        self._seed_idx = torch.arange(num_seeds, device=self._device)
        self._init_obs_storage()

    # -- dense obs backend (override to swap storage) --
    def _init_obs_storage(self) -> None:
        shape = (self._S, self._cap, self._fs, self._h, self._w_stored)
        self._obs = torch.zeros(shape, dtype=torch.uint8, device=self._device)
        self._next = torch.zeros(shape, dtype=torch.uint8, device=self._device)

    def _store_obs(self, mask, obs, next_obs) -> None:
        s, p = self._seed_idx, self._ptr
        m4 = mask[:, None, None, None]
        self._obs[s, p] = torch.where(m4, pack_4bit(obs), self._obs[s, p])
        self._next[s, p] = torch.where(m4, pack_4bit(next_obs), self._next[s, p])

    def _read_obs(self, idx):
        s = self._seed_idx[:, None]
        return unpack_4bit(self._obs[s, idx]), unpack_4bit(self._next[s, idx])

    def add(self, mask, obs, next_obs, action, reward, done) -> None:
        s, p = self._seed_idx, self._ptr
        self._store_obs(mask, obs, next_obs)
        self._action[s, p] = torch.where(mask, action, self._action[s, p])
        self._reward[s, p] = torch.where(mask, reward, self._reward[s, p])
        self._done[s, p] = torch.where(mask, done, self._done[s, p])
        self._ptr = torch.where(mask, (p + 1) % self._cap, p)
        self._size = torch.where(mask, torch.clamp(self._size + 1, max=self._cap), self._size)

    def sample(self) -> AtariTransition:
        size = self._size.clamp(min=1)
        rand = torch.rand(self._S, self._bs, device=self._device)
        idx = (rand * size[:, None].to(torch.float32)).long().clamp(max=self._cap - 1)
        s = self._seed_idx[:, None]
        obs, next_obs = self._read_obs(idx)
        done = self._done[s, idx]
        return AtariTransition(
            observations=obs,
            actions=self._action[s, idx],
            next_observations=next_obs,
            rewards=self._reward[s, idx],
            terminated=done,
            truncated=torch.zeros_like(done),
            batch_size=[self._S, self._bs],
        )


class OptionNvcompReplayBuffer(OptionReplayBuffer):
    """obs / next_obs stored as per-(seed, slot) zstd blobs via the shared nvcomp codec, decoded
    on sample. The codec (vectorised_training_util) is reused unchanged; only the SMDP/async blob
    bookkeeping lives here. Trades VRAM for host-side blob lists plus a GPU decode."""

    def _init_obs_storage(self) -> None:
        if not HAVE_NVCOMP:
            raise RuntimeError("OptionNvcompReplayBuffer requires `nvidia.nvcomp` (not installed).")
        self._obs_blobs = [[None] * self._cap for _ in range(self._S)]
        self._next_blobs = [[None] * self._cap for _ in range(self._S)]
        n = self._S * self._bs
        self._dec_obs = torch.empty(n, self._numel, dtype=torch.uint8, device=self._device)
        self._dec_next = torch.empty(n, self._numel, dtype=torch.uint8, device=self._device)
        self._dec_obs_arrays = [_nvcomp.as_array(self._dec_obs[i]) for i in range(n)]
        self._dec_next_arrays = [_nvcomp.as_array(self._dec_next[i]) for i in range(n)]

    def _store_obs(self, mask, obs, next_obs) -> None:
        active = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if not active:
            return
        po = pack_4bit(obs).reshape(self._S, -1).contiguous()
        pn = pack_4bit(next_obs).reshape(self._S, -1).contiguous()
        co = _codec().encode([po[s] for s in active])
        cn = _codec().encode([pn[s] for s in active])
        ptr = self._ptr.tolist()
        for j, s in enumerate(active):
            self._obs_blobs[s][ptr[s]] = co[j]
            self._next_blobs[s][ptr[s]] = cn[j]

    def _read_obs(self, idx):
        slots = idx.tolist()
        arrs_o = [self._obs_blobs[s][slots[s][b]] for s in range(self._S) for b in range(self._bs)]
        arrs_n = [self._next_blobs[s][slots[s][b]] for s in range(self._S) for b in range(self._bs)]
        _codec().decode(arrs_o, data_type="|u1", out=self._dec_obs_arrays)
        _codec().decode(arrs_n, data_type="|u1", out=self._dec_next_arrays)
        shape = (self._S, self._bs, self._fs, self._h, self._w_stored)
        return unpack_4bit(self._dec_obs.reshape(shape)), unpack_4bit(self._dec_next.reshape(shape))


class ComputeDQNVecAgents(DQNVecAgents):
    """DQN agents over the expanded (base_action, option_length) action space."""

    def __init__(self, args, num_actions, framestack, device, num_seeds):
        self.option_lengths = args.option_lengths
        self.K = len(args.option_lengths)
        self.option_lengths_tensor = torch.tensor(args.option_lengths, device=device, dtype=torch.long)
        super().__init__(args, num_actions=num_actions * self.K, framestack=framestack, device=device, num_seeds=num_seeds)

    def _make_replay(self):
        # Full-obs SMDP buffer (the streaming buffer can't hold k-step jumps). nvcomp blobs are the
        # only practical backend at large --buffer-size; bitpack is dense and for small capacities.
        buffer_cls = {"bitpack": OptionReplayBuffer, "nvcomp": OptionNvcompReplayBuffer}[self.args.replay_compress]
        return buffer_cls(self.args.buffer_size, self.num_seeds, self.args.batch_size, self.framestack, self.device)

    def decode(self, expanded):
        """expanded action -> (base_action, option_length), both [S]."""
        return expanded // self.K, self.option_lengths_tensor[expanded % self.K]

    def _build_forward_and_backward(self) -> None:
        gamma = self.args.gamma
        to_cl = self._to_cl
        olt = self.option_lengths_tensor
        K = self.K

        self.frame_stack = FrameStackBuffer(self.num_seeds, self.framestack, 84, 84).to(self.device)
        base_model = self.base_model

        def forward_single(params, obs):
            with params.to_module(base_model):
                return base_model(obs)

        frame_stack = self.frame_stack
        mask_logits = self.action_mask

        def policy_fn(new_frame, done, params):
            obs = frame_stack.update(new_frame, done)
            obs_b = to_cl(obs.float().unsqueeze(1))
            q = vmap(forward_single)(params, obs_b).squeeze(1)
            q = mask_logits(q)
            return q.argmax(dim=-1)

        def _train_single(params, target_params, opt_state, obs, next_obs, actions, rewards, dones, action_mask):
            def loss_fn(p):
                q = forward_single(p, obs)
                q_sel = q.gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q = forward_single(target_params, next_obs)
                next_q = next_q.masked_fill(~action_mask, -torch.inf)
                repeat = olt[actions % K]
                td_target = rewards + (gamma**repeat) * next_q.max(dim=1)[0] * (1 - dones.float())
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


def main(args, metrics_file):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = make_atari_vector_env(env_id=args.env_id, num_envs=args.num_agent_env_pairs, frameskip=5, stack_num=1)
    num_actions = int(envs.single_action_space.n)
    S = args.num_agent_env_pairs
    agents = ComputeDQNVecAgents(args, num_actions=num_actions, framestack=4, device=device, num_seeds=S)

    frame_np, _ = envs.reset(seed=args.seed)
    frame = torch.as_tensor(frame_np, device=device, dtype=torch.uint8)
    done = torch.zeros(S, dtype=torch.bool, device=device)

    # per-seed option state
    repeat_left = torch.zeros(S, dtype=torch.long, device=device)  # 0 => decide this tick
    committed_base = torch.zeros(S, dtype=torch.long, device=device)
    committed_exp = torch.zeros(S, dtype=torch.long, device=device)
    accum_reward = torch.zeros(S, dtype=torch.float32, device=device)
    disc = torch.ones(S, dtype=torch.float32, device=device)
    start_obs = torch.zeros(S, agents.framestack, 84, 84, dtype=torch.uint8, device=device)
    pend_done = torch.zeros(S, dtype=torch.bool, device=device)
    have_option = torch.zeros(S, dtype=torch.bool, device=device)
    delib = args.option_deliberation_cost

    avg_returns = deque(maxlen=100)
    start_time = time.time()
    burnin_step = None
    if args.detect_syncs:
        torch.cuda.set_sync_debug_mode("warn")

    pbar = tqdm.tqdm(range(args.total_timesteps))
    for step in pbar:
        if step == args.learning_starts:
            burnin_step = step
            start_time = time.time()

        epsilon = agents.get_exploration_rate(step)
        torch.compiler.cudagraph_mark_step_begin()
        expanded = agents.select_action(frame, done, epsilon).clone()
        cur_stack = agents.frame_stack.buf
        decision = repeat_left == 0

        store_mask = decision & have_option
        if bool(store_mask.any()):
            agents.replay.add(store_mask, start_obs, cur_stack, committed_exp, accum_reward, pend_done)

        base, length = agents.decode(expanded)
        committed_exp = torch.where(decision, expanded, committed_exp)
        committed_base = torch.where(decision, base, committed_base)
        repeat_left = torch.where(decision, length, repeat_left)
        accum_reward = torch.where(decision, torch.full_like(accum_reward, delib), accum_reward)
        disc = torch.where(decision, torch.ones_like(disc), disc)
        start_obs = torch.where(decision[:, None, None, None], cur_stack.clone(), start_obs)
        have_option = have_option | decision

        next_frame_np, reward, terminated, truncated, info = envs.step(committed_base.cpu().numpy())
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=device)
        accum_reward = accum_reward + disc * reward_t
        disc = disc * args.gamma
        repeat_left = repeat_left - 1

        frame = torch.as_tensor(next_frame_np, device=device, dtype=torch.uint8)
        term_t = torch.as_tensor(terminated, device=device)
        done = torch.as_tensor(terminated | truncated, device=device)
        ended = (repeat_left == 0) | done
        repeat_left = torch.where(done, torch.zeros_like(repeat_left), repeat_left)
        pend_done = torch.where(ended, term_t, pend_done)

        if "episode" in info:
            avg_returns.extend(info["episode"]["r"][info["_episode"]])

        if step > args.learning_starts and step % args.train_frequency == 0:
            torch.compiler.cudagraph_mark_step_begin()
            agents.update(agents.replay.sample())

        if burnin_step is not None and step % 2500 == 0:
            _log_progress(pbar, args, metrics_file, step, burnin_step, start_time, avg_returns, epsilon)

    envs.close()


if __name__ == "__main__":
    args = tyro.cli(ComputeArgs)
    assert args.metrics_path, "Must specify --metrics-path to save results."
    metrics_file = open(args.metrics_path, "w")
    main(args, metrics_file)
