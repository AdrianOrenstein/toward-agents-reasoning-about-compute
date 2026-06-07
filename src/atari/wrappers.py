from __future__ import annotations

import collections
import time
from typing import Union

import numpy as np
from ale_py.vector_env import AtariVectorEnv
from gymnasium.logger import warn
from gymnasium.vector.vector_env import AutoresetMode


class AtariVectorEnvWrapper:
    """Base wrapper for AtariVectorEnv that preserves the send/recv API."""

    def __init__(self, env: Union[AtariVectorEnv, AtariVectorEnvWrapper]):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        return self.env.step(actions)

    def send(self, actions):
        self.env.send(actions)

    def recv(self):
        return self.env.recv()

    def wrap(self, *wrappers):
        """Apply additional wrappers in sequence and return the outermost wrapper."""
        env = self
        for w in wrappers:
            cls, *args = w if isinstance(w, tuple) else (w,)
            env = cls(env, *args)
        return env

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


class _PerEnvMeanBuffer:
    """Per-environment rolling mean buffer."""

    def __init__(self, num_envs: int, capacity: int):
        self._capacity = capacity
        self._queues = [collections.deque(maxlen=capacity) for _ in range(num_envs)]

    def add(self, env_idx: int, val) -> None:
        self._queues[env_idx].append(val)

    def mean_all(self) -> np.ndarray:
        return np.array([float(np.mean(q)) if q else np.nan for q in self._queues])


class RecordEpisodeStatistics(AtariVectorEnvWrapper):
    """Records cumulative rewards and episode lengths; adds them to info under 'episode'."""

    def __init__(self, env, buffer_length: int = 100, stats_key: str = "episode"):
        super().__init__(env)
        self._stats_key = stats_key
        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` in metadata; assuming AutoresetMode.NEXT_STEP."
            )
            self._autoreset_mode = AutoresetMode.NEXT_STEP
        else:
            assert isinstance(self.env.metadata["autoreset_mode"], AutoresetMode)
            self._autoreset_mode = self.env.metadata["autoreset_mode"]

        self.episode_count = 0
        self.episode_start_times = np.zeros((self.num_envs,))
        self.episode_returns = np.zeros((self.num_envs,))
        self.episode_lengths = np.zeros((self.num_envs,), dtype=int)
        self.prev_dones = np.zeros((self.num_envs,), dtype=bool)

        self.time_queue = _PerEnvMeanBuffer(self.num_envs, buffer_length)
        self.return_queue = _PerEnvMeanBuffer(self.num_envs, buffer_length)
        self.length_queue = _PerEnvMeanBuffer(self.num_envs, buffer_length)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        if options is not None and "reset_mask" in options:
            reset_mask = options.pop("reset_mask")
            assert isinstance(reset_mask, np.ndarray)
            assert reset_mask.shape == (self.num_envs,)
            assert reset_mask.dtype == np.bool_
            assert np.any(reset_mask)
            self.episode_start_times[reset_mask] = time.perf_counter()
            self.episode_returns[reset_mask] = 0
            self.episode_lengths[reset_mask] = 0
            self.prev_dones[reset_mask] = False
        else:
            self.episode_start_times = np.full(self.num_envs, time.perf_counter())
            self.episode_returns = np.zeros(self.num_envs)
            self.episode_lengths = np.zeros(self.num_envs, dtype=int)
            self.prev_dones = np.zeros(self.num_envs, dtype=bool)

        return obs, info

    def _update_episode_statistics(self, rewards, terminations, truncations, infos):
        assert isinstance(infos, dict), (
            f"`RecordEpisodeStatistics` requires info type to be dict, got {type(infos)}."
        )

        self.episode_returns[self.prev_dones] = 0
        self.episode_returns[~self.prev_dones] += rewards[~self.prev_dones]

        self.episode_lengths[self.prev_dones] = 0
        steps = infos.get("steps_taken", np.ones(self.num_envs, dtype=int))
        self.episode_lengths[~self.prev_dones] += steps[~self.prev_dones]

        self.episode_start_times[self.prev_dones] = time.perf_counter()

        self.prev_dones = dones = np.logical_or(terminations, truncations)
        num_dones = int(np.sum(dones))

        if not num_dones:
            return

        if self._stats_key in infos or f"_{self._stats_key}" in infos:
            raise ValueError(
                f"Key '{self._stats_key}' already exists in info: {list(infos.keys())}"
            )

        episode_time_length = np.round(time.perf_counter() - self.episode_start_times, 6)

        for i in np.where(dones)[0]:
            self.time_queue.add(i, float(episode_time_length[i]))
            self.return_queue.add(i, float(self.episode_returns[i]))
            self.length_queue.add(i, int(self.episode_lengths[i]))

        infos[self._stats_key] = {
            "r": np.where(dones, self.episode_returns, 0.0),
            "l": np.where(dones, self.episode_lengths, 0),
            "t": np.where(dones, episode_time_length, 0.0),
            "avg_r": self.return_queue.mean_all(),
            "avg_l": self.length_queue.mean_all(),
            "avg_t": self.time_queue.mean_all(),
        }
        infos[f"_{self._stats_key}"] = dones
        self.episode_count += num_dones

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self._update_episode_statistics(rewards, terminations, truncations, infos)
        return obs, rewards, terminations, truncations, infos

    def recv(self):
        obs, rewards, terminations, truncations, infos = super().recv()
        self._update_episode_statistics(rewards, terminations, truncations, infos)
        return obs, rewards, terminations, truncations, infos


class TransformReward(AtariVectorEnvWrapper):
    """Transform rewards using a given function (e.g. np.sign for reward clipping)."""

    def __init__(self, env, transform_fn=np.sign):
        super().__init__(env)
        self.transform_fn = transform_fn

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return obs, self.transform_fn(rewards), terminations, truncations, infos

    def recv(self):
        obs, rewards, terminations, truncations, infos = super().recv()
        return obs, self.transform_fn(rewards), terminations, truncations, infos


class TorchOpsWrapper(AtariVectorEnvWrapper):
    """Registers ALE torch custom ops; ale_recv routes through the Python wrapper chain."""

    def __init__(self, env):
        super().__init__(env)
        (
            self.handle_id,
            self.ale_send,
            self.ale_step,
            _,
            self._unregister,
        ) = env.torch()

    def ale_recv(self, handle_id: int):
        import torch

        obs, reward, term, trunc, infos = self.env.recv()
        return (
            torch.as_tensor(obs),
            torch.as_tensor(reward),
            torch.as_tensor(term),
            torch.as_tensor(trunc),
            infos,
        )

    def close(self):
        self._unregister()
        super().close()
