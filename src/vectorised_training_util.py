"""Shared building blocks for the vectorised-training LeanRL scripts.

These are the pieces that let one process train N independent agent-env pairs at once: the
parameters of N networks stacked into a single TensorDict and driven under `vmap`, a functional
Adam that is itself `vmap`-able, and a memory-frugal Atari replay buffer (4-bit frame packing +
single-frame storage with on-sample stack reconstruction). They are copied here, not imported from
a private monorepo, so the LeanRL scripts stay self-contained.

Provenance (compute_methods repo):
  pack_4bit/unpack_4bit, FrameReplayBuffer: src/agents/dqn/compression/ (orig. FastReplayBuffer)
  Adam (functional, vmap-able)            : src/trainer/optimiser.py
  stack_models / mask helpers             : src/agents/dqn/agent_design.py
  FrameStackBuffer                        : src/scratchpad/speeding_up_dqn/frame_stack_buffer.py
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict, from_modules, tensorclass
from torch.func import vmap


# --------------------------------------------------------------------------------------
# 4-bit frame packing: two uint8 pixels -> one byte. Halves replay VRAM and PCIe traffic.
# --------------------------------------------------------------------------------------
def pack_4bit(grayscale_frames: torch.Tensor) -> torch.Tensor:
    """Takes [..., W] uint8 tensor and packs to [..., W//2]"""
    p1 = grayscale_frames[..., 0::2]
    p2 = grayscale_frames[..., 1::2]
    return ((p1 >> 4) << 4) | (p2 >> 4)


def unpack_4bit(packed_frames: torch.Tensor) -> torch.Tensor:
    """Takes [..., W//2] uint8 tensor and unpacks to [..., W]"""
    p1 = packed_frames & 0xF0
    p2 = (packed_frames & 0x0F) << 4
    unpacked_shape = list(packed_frames.shape)
    unpacked_shape[-1] *= 2
    unpacked = torch.empty(unpacked_shape, dtype=torch.uint8, device=packed_frames.device)
    unpacked[..., 0::2] = p1
    unpacked[..., 1::2] = p2
    return unpacked


def test_pack_4bit_roundtrip() -> None:
    x = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    packed = pack_4bit(x)
    assert packed.shape == (2, 4, 84, 42)
    # 4-bit packing keeps the top nibble; round-trip is exact to 16 levels.
    assert torch.equal(unpack_4bit(packed), (x >> 4) << 4)


# --------------------------------------------------------------------------------------
# Functional Adam: optimiser state + hyperparameters in one tensorclass, so a stack of N
# optimisers is one tensorclass with batch_size=[N] and `update` runs under vmap.
# --------------------------------------------------------------------------------------
@tensorclass
class Adam:
    """Functional Adam optimizer - state and hyperparams in one vmappable tensorclass."""

    mu: TensorDict
    nu: TensorDict
    step: torch.Tensor
    lr: torch.Tensor
    beta1: torch.Tensor
    beta2: torch.Tensor
    eps: torch.Tensor
    weight_decay: torch.Tensor  # sentinel -1.0 when disabled
    max_norm: torch.Tensor  # sentinel -1.0 when disabled

    @classmethod
    @torch.no_grad()
    def init(
        cls,
        params: TensorDict,
        device: torch.device,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: Optional[float] = None,
        max_norm: Optional[float] = None,
    ) -> Adam:
        return cls(
            mu=params.apply(torch.zeros_like),
            nu=params.apply(torch.zeros_like),
            step=torch.zeros((), dtype=torch.int64, device=device),
            lr=torch.tensor(lr, dtype=torch.float32, device=device),
            beta1=torch.tensor(betas[0], dtype=torch.float32, device=device),
            beta2=torch.tensor(betas[1], dtype=torch.float32, device=device),
            eps=torch.tensor(eps, dtype=torch.float32, device=device),
            weight_decay=torch.tensor(
                weight_decay if weight_decay is not None else -1.0, dtype=torch.float32, device=device
            ),
            max_norm=torch.tensor(max_norm if max_norm is not None else -1.0, dtype=torch.float32, device=device),
            batch_size=[],
        )

    @torch.no_grad()
    def clip_grads(self, grads: TensorDict) -> TensorDict:
        """Clip grads by global L2 norm to max_norm; no-op when the sentinel is -1. vmap-safe."""
        max_norm = torch.where(self.max_norm > 0, self.max_norm, self.max_norm.new_tensor(float("inf")))
        leaves = [g for _, g in grads.items(include_nested=True, leaves_only=True)]
        total_norm = torch.norm(torch.stack([g.reshape(-1).float().norm(2) for g in leaves]), 2)
        clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        return grads.apply(lambda g: g * clip_coef.to(g.dtype))

    @torch.no_grad()
    def apply_weight_decay(self, step: TensorDict, params: TensorDict) -> TensorDict:
        """Subtract L2 penalty from step. No-op when weight_decay sentinel is -1."""
        wd = torch.where(self.weight_decay > 0, self.weight_decay, torch.zeros_like(self.weight_decay))
        return step.apply(lambda s, p: s - self.lr * wd * p, params)

    @torch.no_grad()
    def apply_grads(self, params: TensorDict, step: TensorDict) -> TensorDict:
        return params.apply(lambda p, s: p + s, step)

    @torch.no_grad()
    def update(self, grads: TensorDict) -> tuple[TensorDict, Adam]:
        """Compute Adam step. Returns (step, new_state)."""
        t = self.step + 1
        bc1 = 1.0 - self.beta1 ** t.float()
        bc2 = 1.0 - self.beta2 ** t.float()
        new_mu = self.mu.apply(lambda m, g: m.lerp(g, (1.0 - self.beta1).to(device=m.device, dtype=m.dtype)), grads)
        new_nu = self.nu.apply(
            lambda v, g: v.lerp(g.pow(2), (1.0 - self.beta2).to(device=v.device, dtype=v.dtype)), grads
        )
        step = new_mu.apply(
            lambda m, v: (
                -(self.lr.to(device=m.device, dtype=m.dtype) * (m / bc1.to(device=m.device, dtype=m.dtype)))
                / ((v / bc2.to(device=v.device, dtype=v.dtype)).sqrt() + self.eps.to(device=v.device, dtype=v.dtype))
            ),
            new_nu,
        )
        return step, Adam(
            mu=new_mu,
            nu=new_nu,
            step=t,
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            weight_decay=self.weight_decay,
            max_norm=self.max_norm,
            batch_size=[],
        )


def test_adam_matches_pytorch() -> None:
    torch.manual_seed(0)
    params = TensorDict({"w": torch.randn(4, 4), "b": torch.randn(4)}, batch_size=[])
    grads_seq = [
        TensorDict({k: v + 0.01 * torch.randn_like(v) for k, v in params.items()}, batch_size=[]).clone()
        for _ in range(5)
    ]
    w = params.clone()
    state = Adam.init(w, device=torch.device("cpu"), lr=1e-3)
    for g in grads_seq:
        s, state = state.update(g)
        w = state.apply_grads(w, s)

    ref = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    opt = torch.optim.Adam(list(ref.values()), lr=1e-3)
    for g in grads_seq:
        opt.zero_grad()
        for k, p in ref.items():
            p.grad = g[k].clone()
        opt.step()
    for k in w.keys():
        assert torch.allclose(w[k], ref[k].detach(), atol=1e-6), f"Adam mismatch on {k}"


# --------------------------------------------------------------------------------------
# Stack N independent modules into one TensorDict (batch_size=[N]) plus a zeroed base model.
# Forward N networks at once with `vmap(forward_single)`, where forward_single installs one
# slice of params into the base model via the to_module context manager.
# --------------------------------------------------------------------------------------
def stack_models(models: list[nn.Module]) -> tuple[TensorDict, nn.Module]:
    """Returns (params, base_model). params has batch_size=[len(models)]; base_model is a
    zeroed structural copy used as the functional carrier under to_module."""
    base_model = copy.deepcopy(models[0])
    for p in base_model.parameters():
        p.data.zero_()
    for b in base_model.buffers():
        b.data.zero_()
    params = from_modules(*models, lock=False).detach().contiguous()
    return params, base_model


def to_stacked_channels_last(t: torch.Tensor) -> torch.Tensor:
    """Lay out a 5D stack [N, C0, C1, H, W] so each 4D slice is channels_last (NHWC).

    cudnn runs the convs in NHWC; if the stacked conv weights and obs are physically NHWC-per-slice,
    cudnn picks NHWC kernels at CUDA-graph capture and the per-step NCHW<->NHWC transpose disappears
    (~1.3-1.5x on the vmapped update - see leanrl_compression_profiling/DQN_PROFILE_FINDINGS.md).
    `.contiguous(memory_format=channels_last)` is NYI inside vmap and undefined on the rank-5 stack,
    so the strides are built by hand: slice [i] then satisfies is_contiguous(channels_last)=True.
    Must be applied to weights BEFORE CUDA-graph capture and to obs every step (outside vmap)."""
    assert t.dim() == 5, f"expected 5D stack, got {t.dim()}D"
    _, c0, c1, h, w = t.shape
    slice_strides = (c1 * h * w, 1, w * c1, c1)
    strides = (c0 * c1 * h * w,) + slice_strides
    out = torch.empty_strided(t.shape, strides, device=t.device, dtype=t.dtype)
    out.copy_(t)
    return out


def test_to_stacked_channels_last() -> None:
    t = torch.randn(3, 32, 4, 8, 8)
    cl = to_stacked_channels_last(t)
    for i in range(3):
        assert cl[i].is_contiguous(memory_format=torch.channels_last), f"slice {i} not channels_last"
    assert torch.equal(t, cl)


class NormaliseAtari(nn.Module):
    def forward(self, x):
        """Normalise Atari observations from [0, 255] to [-1.0, 1.0]"""
        x = x.float() / 255.0
        return (x - 0.5) / 0.5


class LazyRMSNorm(nn.Module):
    """RMSNorm that infers normalized_shape from the first forward call and caches it."""

    _normalized_shape: tuple | None = None

    def forward(self, input):
        if self._normalized_shape is None:
            self._normalized_shape = tuple(input.shape[1:])
        return torch.nn.functional.rms_norm(input, self._normalized_shape)


class ValidActionMask(nn.Module):
    """Holds the per agent-env action mask."""

    def __init__(self, action_counts: torch.Tensor, num_actions: int):
        super().__init__()
        mask = torch.arange(num_actions, device=action_counts.device) < action_counts.unsqueeze(1)
        self.register_buffer("mask", mask)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~self.mask, -torch.inf)


def test_valid_action_mask_module() -> None:
    counts = torch.tensor([6, 4, 5])
    masker = ValidActionMask(counts, 6)
    assert "mask" in dict(masker.named_buffers())
    out = masker(torch.randn(3, 6))
    valid = torch.arange(6) < counts.unsqueeze(1)
    assert torch.equal(out.isinf() & (out < 0), ~valid)


class FrameStackBuffer(nn.Module):
    """
    Inference-time GPU frame stack. Lets the env return stack_num=1 (4x less CPU->GPU data);
    the rolling [S, FS, H, W] stack is maintained in-place so CUDA graph captures stay valid.

    """

    def __init__(self, num_seeds: int, framestack: int, h: int, w: int):
        super().__init__()
        self.framestack = framestack
        self.register_buffer("buf", torch.zeros(num_seeds, framestack, h, w, dtype=torch.uint8))

    def reset(self, first_frame: torch.Tensor) -> None:
        """first_frame: [S, 1, H, W] uint8 on the same device as buf."""
        self.buf.copy_(first_frame[:, 0:1].expand_as(self.buf))

    def update(self, new_frame: torch.Tensor, terminations: torch.Tensor) -> torch.Tensor:
        """new_frame: [S, 1, H, W] uint8; terminations: [S] bool. Returns [S, FS, H, W].

        One fused shift (roll) instead of a per-slot copy loop: the 3-launch loop was the whole
        net cost at low env counts (see leanrl_compression_profiling/DQN_PROFILE_FINDINGS.md).
        Compiling this standalone is slower (cudagraph-tree overhead exceeds the sub-microsecond of
        work); the trainer instead folds this roll into its compiled policy graph."""
        self.buf.masked_fill_(terminations[:, None, None, None], 0)
        self.buf.copy_(torch.roll(self.buf, -1, dims=1))
        self.buf[:, -1].copy_(new_frame[:, 0])
        return self.buf


def test_stack_models_vmap_forward() -> None:
    torch.manual_seed(0)
    make = lambda: nn.Sequential(nn.Flatten(), nn.Linear(4, 3))
    models = [make(), make()]
    params, base = stack_models(models)
    assert params.batch_size == torch.Size([2])

    def fwd(p, obs):
        with p.to_module(base):
            return base(obs)

    obs = torch.randn(2, 1, 4)
    out = vmap(fwd)(params, obs)
    assert out.shape == (2, 1, 3)
    # vmapped forward must equal each model run independently.
    for i in range(2):
        assert torch.allclose(out[i], models[i](obs[i]), atol=1e-5)


# --------------------------------------------------------------------------------------
# Atari replay buffer (per-seed circular). Stores the frame of each step and reconstructs 
# the stack on sample, skipping episode boundaries. Rewards/dones are dense.
# reconstructs the stacked observation on sample, skipping episode boundaries.
# --------------------------------------------------------------------------------------
@tensorclass
class AtariTransition:
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class FrameReplayBuffer:
    """Circular replay buffer for N seeds. Dense obs storage + optional 4-bit packing."""

    def __init__(
        self,
        capacity: int,
        num_seeds: int,
        batch_size: int,
        framestack: int = 4,
        frame_shape: tuple = (84, 84),
        device: str | torch.device = "cuda",
        bitpack: bool = True,
    ):
        self._capacity = capacity
        self._num_seeds = num_seeds
        self._batch_size = batch_size
        self._framestack = framestack
        self._frame_shape = frame_shape
        self._bitpack = bitpack
        self._device = torch.device(device)
        self._w_stored = frame_shape[1] // 2 if bitpack else frame_shape[1]

        self._meta = TensorDict(
            {
                "action": torch.zeros(num_seeds, capacity, dtype=torch.int64, device=self._device),
                "reward": torch.zeros(num_seeds, capacity, dtype=torch.float32, device=self._device),
                "terminated": torch.zeros(num_seeds, capacity, dtype=torch.bool, device=self._device),
                "truncated": torch.zeros(num_seeds, capacity, dtype=torch.bool, device=self._device),
                "done": torch.zeros(num_seeds, capacity, dtype=torch.bool, device=self._device),
            },
            batch_size=[num_seeds, capacity],
            device=self._device,
        ).contiguous()
        self._init_obs_storage()

        self._ptr = 0
        self._size = 0

    def _init_obs_storage(self) -> None:
        h, _ = self._frame_shape
        self._obs = torch.zeros(
            self._num_seeds, self._capacity, h, self._w_stored, dtype=torch.uint8, device=self._device
        )

    def _maybe_pack(self, frames: torch.Tensor) -> torch.Tensor:
        return pack_4bit(frames) if self._bitpack else frames

    def _maybe_unpack(self, frames: torch.Tensor) -> torch.Tensor:
        return unpack_4bit(frames) if self._bitpack else frames

    def _write_obs(self, p: int, frames: torch.Tensor) -> None:
        """frames: [S, h, w] uint8 on device. Store the per-seed frame at slot p."""
        self._obs[:, p] = self._maybe_pack(frames)

    def _read_obs_frames(self, frame_slots: torch.Tensor) -> torch.Tensor:
        """frame_slots: [S, batch, frl] capacity indices. Return [S, batch, frl, h, w] uint8."""
        seed_idx = torch.arange(self._num_seeds, device=self._device)[:, None, None]
        return self._maybe_unpack(self._obs[seed_idx, frame_slots])

    def add(self, obs, action, reward, terminated, truncated) -> None:
        """obs: [S, framestack, H, W] uint8. Stores only the last frame per seed."""
        frames = torch.as_tensor(obs[:, -1], device=self._device)
        p = self._ptr
        self._write_obs(p, frames)
        self._meta["action"][:, p] = torch.as_tensor(action, device=self._device)
        self._meta["reward"][:, p] = torch.as_tensor(reward, dtype=torch.float32, device=self._device)
        self._meta["terminated"][:, p] = torch.as_tensor(terminated, device=self._device)
        self._meta["truncated"][:, p] = torch.as_tensor(truncated, device=self._device)
        self._meta["done"][:, p] = torch.as_tensor(np.asarray(terminated) | np.asarray(truncated), device=self._device)
        self._ptr = (p + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self) -> AtariTransition:
        window_size = 2 * (self._framestack + 1) + 1
        assert self._size > window_size, f"Buffer not ready: {self._size} <= {window_size}"
        max_start = self._size - window_size
        starts = torch.randint(0, max_start, (self._batch_size,), device=self._device)
        return self._sample_from_starts(starts)

    def _sample_from_starts(self, starts: torch.Tensor) -> AtariTransition:
        num_seeds, batch_size, capacity = self._num_seeds, self._batch_size, self._capacity
        h, w = self._frame_shape
        frame_run_length = self._framestack + 1
        window_size = 2 * (self._framestack + 1) + 1
        threshold = window_size - frame_run_length
        base = self._ptr if self._size == self._capacity else 0
        win_idx = (base + starts.unsqueeze(-1) + torch.arange(window_size, device=self._device)) % capacity

        # Read the (small, dense) done window first so the boundary offset is known before we touch
        # obs - then only the FRAMESTACK+1 frames actually used are decoded, not the full window.
        done_win = self._meta["done"][:, win_idx]
        act_win = self._meta["action"][:, win_idx]
        rew_win = self._meta["reward"][:, win_idx]
        term_win = self._meta["terminated"][:, win_idx]
        trunc_win = self._meta["truncated"][:, win_idx]

        # Pick a frame run that does not straddle a done flag (so a stacked obs never mixes
        # frames from two episodes); shift the run forward past the most recent boundary.
        done_trimmed = done_win[..., :-1]
        has_done = done_trimmed.any(-1)
        last_done_rev = done_trimmed.long().flip(-1).argmax(-1)
        last_done = (window_size - 2) - last_done_rev
        offset = torch.where(has_done & (last_done < threshold), last_done + 1, torch.zeros_like(last_done))

        # Capacity indices of the needed frame run, per (seed, batch): win_idx[b, offset[s,b] + k].
        frame_pos = offset.unsqueeze(-1) + torch.arange(frame_run_length, device=self._device)
        win_exp = win_idx.unsqueeze(0).expand(num_seeds, batch_size, window_size)
        frame_slots = win_exp.gather(2, frame_pos)  # [S, batch, frl]
        frame_run = self._read_obs_frames(frame_slots)  # [S, batch, frl, h, w]

        obs_windows = frame_run.unfold(2, self._framestack, 1).permute(0, 1, 2, 5, 3, 4).contiguous()
        observations = obs_windows[:, :, 0]
        next_observations = obs_windows[:, :, 1]

        t_idx = (offset + self._framestack - 1).unsqueeze(-1)
        actions = act_win.gather(2, t_idx).squeeze(-1)
        rewards = rew_win.gather(2, t_idx).squeeze(-1)
        terminated = term_win.gather(2, t_idx).squeeze(-1)
        truncated = trunc_win.gather(2, t_idx).squeeze(-1)

        return AtariTransition(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            batch_size=[num_seeds, batch_size],
        )


# --------------------------------------------------------------------------------------
# Optional nvcomp zstd codec.
# --------------------------------------------------------------------------------------
try:
    import nvidia.nvcomp as _nvcomp

    HAVE_NVCOMP = True
except ImportError:
    HAVE_NVCOMP = False

if HAVE_NVCOMP:
    _CODEC: _nvcomp.Codec | None = None

    def _codec() -> _nvcomp.Codec:
        global _CODEC
        if _CODEC is None:
            _CODEC = _nvcomp.Codec(algorithm="Zstd", bitstream_kind=_nvcomp.BitstreamKind.RAW)
        return _CODEC


class NvcompReplayBuffer(FrameReplayBuffer):
    """obs frames stored as per-(seed, slot) zstd blobs (nvcomp), decoded on sample. Stacks on
    top of 4-bit packing when bitpack=True (zstd compresses the packed nibbles). Trades VRAM for
    host-side blob bookkeeping + a GPU decode; the win grows with replay size and sparsity."""

    def _init_obs_storage(self) -> None:
        if not HAVE_NVCOMP:
            raise RuntimeError("NvcompReplayBuffer requires `nvidia.nvcomp` (not installed).")
        h, _ = self._frame_shape
        self._numel = h * self._w_stored
        # Store the nvcomp encode-output Arrays directly: each owns its compressed GPU buffer, so
        # there is no torch round-trip (from_dlpack/clone) on write and no re-wrap on read.
        self._blob_arrays: list[list[object]] = [[None] * self._capacity for _ in range(self._num_seeds)]
        n_total = self._num_seeds * self._batch_size * (self._framestack + 1)
        self._decode_out = torch.empty(n_total, self._numel, dtype=torch.uint8, device=self._device)
        self._decode_out_arrays = [_nvcomp.as_array(self._decode_out[i]) for i in range(n_total)]

    def _write_obs(self, p: int, frames: torch.Tensor) -> None:
        packed = self._maybe_pack(frames).reshape(self._num_seeds, -1).contiguous()
        comps = _codec().encode([packed[s] for s in range(self._num_seeds)])
        for s in range(self._num_seeds):
            self._blob_arrays[s][p] = comps[s]

    def _read_obs_frames(self, frame_slots: torch.Tensor) -> torch.Tensor:
        # One codec.decode for all seeds: inputs are the arrays pre-wrapped at encode, output is the
        # preallocated buffer wrapped once in _init. frame_slots is always [S, batch, framestack+1].
        num_seeds, batch, frl = frame_slots.shape
        h = self._frame_shape[0]
        slots = frame_slots.reshape(num_seeds, -1).tolist()
        arrs = [self._blob_arrays[s][i] for s in range(num_seeds) for i in slots[s]]
        # Pass the Array list straight to decode; wrapping it via as_arrays first is pure overhead.
        _codec().decode(arrs, data_type="|u1", out=self._decode_out_arrays)
        obs = self._decode_out.reshape(num_seeds, batch, frl, h, self._w_stored)
        return self._maybe_unpack(obs)


def test_replay_buffer_sample_shapes() -> None:
    if not torch.cuda.is_available():
        return
    num_seeds, batch_size, fs, hw = 3, 32, 4, 84
    buf = FrameReplayBuffer(capacity=1000, num_seeds=num_seeds, batch_size=batch_size, framestack=fs, device="cuda")
    rng = np.random.RandomState(0)
    for _ in range(1000):
        obs = rng.randint(0, 256, (num_seeds, fs, hw, hw), dtype=np.uint8)
        buf.add(
            obs,
            rng.randint(0, 6, (num_seeds,), dtype=np.int64),
            rng.randn(num_seeds).astype(np.float32),
            rng.random(num_seeds) < 0.05,
            np.zeros(num_seeds, dtype=np.bool_),
        )
    batch = buf.sample()
    assert batch.observations.shape == (num_seeds, batch_size, fs, hw, hw)
    assert batch.next_observations.shape == (num_seeds, batch_size, fs, hw, hw)
    assert batch.actions.shape == (num_seeds, batch_size)
    assert not batch.observations.isnan().any()


def test_nvcomp_buffer_matches_dense() -> None:
    """NvcompReplayBuffer must reconstruct the exact same frames as the dense buffer: fill both
    with identical data, then compare _read_obs_frames for fixed slots (zstd is lossless)."""
    if not torch.cuda.is_available() or not HAVE_NVCOMP:
        return
    num_seeds, batch_size, capacity, fs, hw = 2, 8, 200, 4, 84
    dense = FrameReplayBuffer(capacity, num_seeds, batch_size, framestack=fs, device="cuda")
    nvc = NvcompReplayBuffer(capacity, num_seeds, batch_size, framestack=fs, device="cuda")
    rng = np.random.RandomState(0)
    for _ in range(capacity):
        obs = rng.randint(0, 256, (num_seeds, fs, hw, hw), dtype=np.uint8)
        a = rng.randint(0, 6, (num_seeds,), dtype=np.int64)
        r = rng.randn(num_seeds).astype(np.float32)
        term = rng.random(num_seeds) < 0.05
        trunc = np.zeros(num_seeds, dtype=np.bool_)
        dense.add(obs, a, r, term, trunc)
        nvc.add(obs, a, r, term, trunc)
    frl = fs + 1
    frame_slots = (
        torch.arange(num_seeds * batch_size * frl, device="cuda").reshape(num_seeds, batch_size, frl) % capacity
    )
    d = dense._read_obs_frames(frame_slots)
    n = nvc._read_obs_frames(frame_slots)
    assert d.shape == n.shape == (num_seeds, batch_size, frl, hw, hw)
    assert torch.equal(d, n), "nvcomp decode does not match dense storage"


if __name__ == "__main__":
    test_pack_4bit_roundtrip()
    test_adam_matches_pytorch()
    test_stack_models_vmap_forward()
    test_to_stacked_channels_last()
    test_valid_action_mask_module()
    test_replay_buffer_sample_shapes()
    test_nvcomp_buffer_matches_dense()
