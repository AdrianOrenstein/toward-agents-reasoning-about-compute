import time
from typing import cast

import gymnasium as gym
import numpy as np
from ale_py.vector_env import AtariVectorEnv

from src.atari.rom import env_id_to_rom_name
from src.atari.wrappers import (
    AtariVectorEnvWrapper,
    RecordEpisodeStatistics,
    TransformReward,
)


def _to_rom_name(s: str) -> str:
    try:
        return env_id_to_rom_name(s)
    except Exception:
        return s


def make_atari_vector_env(**kwargs) -> AtariVectorEnvWrapper:
    """Create an AtariVectorEnv with the given parameters.

    Args:
        **kwargs: Keyword arguments for AtariVectorEnv. Overrides defaults.
            game: str | list[str] - ROM name(s) or ALE env ID(s). When a list,
                  num_envs is inferred from the list length.

    Returns:
        AtariVectorEnvWrapper: The created Atari vector environment wrapped with statistics and reward transform.
    """

    raw_game = kwargs.pop("env_id", None) or kwargs.pop("game", None) or "ALE/Pong-v5"
    if isinstance(raw_game, list):
        game: str | list[str] = [_to_rom_name(g) for g in raw_game]
    else:
        game = _to_rom_name(raw_game)

    # Set defaults
    defaults = dict(
        game=game,
        # Number of parallel environments (None lets AtariVectorEnv infer from game list)
        num_envs=None if isinstance(game, list) else 3,
        #
        # -- Preprocessing parameters --
        # Number of frames to skip (action repeat)
        frameskip=5,
        # Use grayscale observations
        grayscale=True,
        # Number of frames to stack
        stack_num=4,
        # Height to resize frames to
        img_height=84,
        # Width to resize frames to
        img_width=84,
        # If to maxpool sequential frames
        maxpool=True,
        # If to clip environment step rewards between -1 and 1. We do this in the TransformReward wrapper.
        reward_clipping=False,
        #
        # -- Environment behavior --
        noop_max=30,  # Maximum number of no-ops at reset
        use_fire_reset=True,  # Press FIRE on reset for games that require it
        # End episodes on life loss
        episodic_life=False,
        # Return termination signal on life loss but don't reset the environment until all lives are a lot.
        # If used, this MUST be indicated as has a significant impact on training performance.
        life_loss_info=False,
        # Max frames per episode (27000 steps * 5 frame skip = 135_000 frames)
        max_num_frames_per_episode=135_000,
        # Sticky actions probability (0.25 for sticky revisiting ALE paper)
        repeat_action_probability=0.25,
        # Use full action space (not minimal)
        full_action_space=False,
        # If to use continuous actions
        continuous=False,
        # The threshold at which to use continuous actions
        continuous_action_threshold=0.5,
        #
        # -- Performance options --
        # Number of environments to process at once (default=0 is the `num_envs`)
        batch_size=0,
        # Number of worker threads (0=auto)
        num_threads=0,
        # CPU core offset for thread affinity (-1=no affinity)
        thread_affinity_offset=-1,
        # How reset sub-environments when they terminated (https://farama.org/Vector-Autoreset-Mode)
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    # Override defaults with provided kwargs
    defaults.update(kwargs)

    env = AtariVectorEnv(**defaults)

    # for compatibility with torchRL wrapper
    env.autoreset_mode = defaults["autoreset_mode"]

    # make the type checker happy
    env.single_observation_space = cast(gym.spaces.Box, env.single_observation_space)
    env.single_action_space = cast(gym.spaces.Discrete, env.single_action_space)

    env = RecordEpisodeStatistics(env)
    env = TransformReward(env, np.sign)

    return env


def test_make_atari_vector_env() -> None:
    """Test the make_atari_vector_env function."""
    env = make_atari_vector_env(
        env_id="ALE/Breakout-v5",
        num_envs=2,
        stack_num=4,
        frameskip=5,
    )
    assert env.single_observation_space.shape == (4, 84, 84)
    assert int(env.single_action_space.n) == 4  # specific to breakout

    # Test reset
    action_space = env.action_space
    observation, info = env.reset(seed=1)
    assert observation.shape == (2, 4, 84, 84)
    assert "env_id" in info and info["env_id"].shape == (2,)
    assert "lives" in info and info["lives"].shape == (2,)
    assert "frame_number" in info and info["frame_number"].shape == (2,)
    assert "episode_frame_number" in info and info["episode_frame_number"].shape == (2,)

    # Test step
    _, reward, terminated, truncated, _ = env.step(action_space.sample())
    assert reward.shape == (2,)
    assert terminated.shape == (2,)
    assert truncated.shape == (2,)

    # Test step until episode ends
    while not terminated.any() and not truncated.any():
        _, _, terminated, truncated, info = env.step(action_space.sample())

    # Test episode statistics
    assert "episode" in info and isinstance(info["episode"], dict)

    # episode return is total game score, this is not the same as the return the agent could compute from its rewards.
    # Typically we show agents the sign of the score (the change in score) rather than the score.
    # Furthermore, agents are typically trained on undiscounted returns, and we usually add a discount on the agent side
    # to present an incorrect, but possibly easier problem to the agent. It's a bit wack.
    assert "r" in info["episode"] and info["episode"]["r"].shape == (2,)
    # episode length in steps: number of agent and environment interactions.
    assert "l" in info["episode"] and info["episode"]["l"].shape == (2,)
    # wallclock
    assert "t" in info["episode"] and info["episode"]["t"].shape == (2,)

    # average return, length, and time: mean of episode returns, lengths, and times.
    assert "avg_r" in info["episode"] and info["episode"]["avg_r"].shape == (2,)
    assert "avg_l" in info["episode"] and info["episode"]["avg_l"].shape == (2,)
    assert "avg_t" in info["episode"] and info["episode"]["avg_t"].shape == (2,)


def test_make_atari_vector_env_multi_rom() -> None:
    """Test make_atari_vector_env with a list of games."""
    env = make_atari_vector_env(game=["pong"] * 2 + ["breakout"] * 2)
    assert env.num_envs == 4
    obs, _ = env.reset(seed=1)
    assert obs.shape == (4, 4, 84, 84)
    _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert reward.shape == (4,)
    env.close()


def benchmark_atari_vector_env(
    env_id="ALE/Pong-v5",
    num_envs=1,
    stack_num=4,
    frameskip=5,
    bench_steps: int = 50_000,
    return_step_timings: bool = False,
) -> tuple[dict[str, float | np.ndarray], dict[str, float | np.ndarray]]:
    """Benchmark the atari vector environment to see the overhead due to async.
    """
    # https://ale.farama.org/vector-environment/#advanced-configuration

    envs = make_atari_vector_env(
        env_id=env_id,
        num_envs=num_envs,
        stack_num=stack_num,
        frameskip=frameskip,
    )

    _ = envs.observation_space
    action_space = envs.action_space
    _ = envs.reset(seed=1)
    observation, reward, terminated, truncated, info = envs.step(action_space.sample())
    print(observation.shape)

    sync_lats_us = np.empty(bench_steps, dtype=np.float64) if return_step_timings else None
    start_time = time.perf_counter()
    prev_t = start_time
    for i in range(bench_steps):
        observation, reward, terminated, truncated, info = envs.step(action_space.sample())
        if sync_lats_us is not None:
            now_t = time.perf_counter()
            sync_lats_us[i] = (now_t - prev_t) * 1e6
            prev_t = now_t

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    avg_time = elapsed_time / bench_steps
    sync_results = {
        "elapsed_time": elapsed_time,
        "avg_time_per_step": avg_time,
        "accumulated_fps": num_envs * frameskip / avg_time,
        "single_env_fps": frameskip / avg_time,
    }
    if sync_lats_us is not None:
        sync_results["latency_us"] = sync_lats_us
        sync_results["throughput_per_sec"] = num_envs * frameskip / (sync_lats_us / 1e6)

    # using Async
    prev_obs, _ = envs.reset(seed=1)
    envs.send(envs.action_space.sample())

    async_lats_us = np.empty(bench_steps, dtype=np.float64) if return_step_timings else None
    start_time = time.perf_counter()
    prev_t = start_time
    for i in range(bench_steps):
        next_obs, rewards, terminations, truncations, infos = envs.recv()
        actions = envs.action_space.sample()
        envs.send(actions)
        if async_lats_us is not None:
            now_t = time.perf_counter()
            async_lats_us[i] = (now_t - prev_t) * 1e6
            prev_t = now_t
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    avg_time = elapsed_time / bench_steps
    async_results = {
        "elapsed_time": elapsed_time,
        "avg_time_per_step": avg_time,
        "accumulated_fps": num_envs * frameskip / avg_time,
        "single_env_fps": frameskip / avg_time,
    }
    if async_lats_us is not None:
        async_results["latency_us"] = async_lats_us
        async_results["throughput_per_sec"] = num_envs * frameskip / (async_lats_us / 1e6)

    return sync_results, async_results

def test_benchmark_atari_vector_env() -> None:
    """Test the benchmark_atari_vector_env function."""
    benchmark_atari_vector_env()


if __name__ == "__main__":
    test_make_atari_vector_env()
    test_make_atari_vector_env_multi_rom()
    sync_results, async_results = benchmark_atari_vector_env()

    print("Synchronised:")
    print(f"Elapsed time for 50000 steps: {sync_results['elapsed_time']:.4f} seconds")
    print(f"Average time per step: {sync_results['avg_time_per_step']:.4f} seconds")
    print(f"Accumulated FPS: {sync_results['accumulated_fps']:.2f}")
    print(f"Single environment FPS: {sync_results['single_env_fps']:.2f}\n")

    print("Asynchronous:")
    print(f"Elapsed time for 50000 steps: {async_results['elapsed_time']:.4f} seconds")
    print(f"Average time per step: {async_results['avg_time_per_step']:.4f} seconds")
    print(f"Accumulated FPS: {async_results['accumulated_fps']:.2f}")
    print(f"Single environment FPS: {async_results['single_env_fps']:.2f}")
