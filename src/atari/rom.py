import ale_py
import gymnasium as gym


def env_id_to_rom_name(env_id: str) -> str:
    """Convert a gymnasium environment ID to its ROM name.

    Args:
        env_id: Gymnasium environment ID like "ALE/Pong-v5"

    Returns:
        ROM name like "pong"

    Examples:
        >>> env_id_to_rom_name("ALE/Pong-v5")
        'pong'
        >>> env_id_to_rom_name("ALE/Breakout-v5")
        'breakout'
    """
    # Register ALE environments if not already registered
    gym.register_envs(ale_py)

    # Get the environment spec and extract the game name from kwargs
    env_spec = gym.spec(env_id)
    return env_spec.kwargs["game"]


def test_env_id_to_rom_name() -> None:
    """As ale_py.vector_env.AtariVectorEnv requires the ROM name, we have a convenience function to
    convert the environment ID to the ROM name."""
    assert env_id_to_rom_name("ALE/Pong-v5") == "pong"
    assert env_id_to_rom_name("ALE/Breakout-v5") == "breakout"


if __name__ == "__main__":
    test_env_id_to_rom_name()
