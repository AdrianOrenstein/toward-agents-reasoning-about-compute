import torch
from typing import List, Annotated
import gymnasium as gym
from lightning.fabric import Fabric
from dataclasses import dataclass, field
import tyro

from src.dqn.agent import DQNAgent, AgentArgs
from src.dqn.network import QNetwork
from src.dqn.compressed_rbuffer import ReplayBufferSamples


@dataclass
class TDQNAgentArgs(AgentArgs):
    """Arguments specific to the tDQN agent algorithm."""

    option_lengths: Annotated[
        List[int],
        tyro.conf.arg(
            metavar="INT INT ...",
            help="List of temporal option lengths. For standard DQN use [1], for temporal options use e.g. [1,2,4] to allow 1, 2, or 4 step decisions.",
        ),
    ] = field(default_factory=lambda: [1])
    """List of temporal option lengths, e.g. [1] for standard DQN which takes 1 step per decision or [1, 2, 4] for temporal options which takes 1, 2, or 4 steps per decision. Each option takes the same action each time."""

    def __post_init__(self):
        # Remove duplicates and sort
        self.option_lengths = sorted(list(set(self.option_lengths)))
        if len(self.option_lengths) == 0:
            raise ValueError("Must provide at least one option length")
        if any(option_length <= 0 for option_length in self.option_lengths):
            raise ValueError("All option lengths must be positive integers")


class OptionDQNAgent(DQNAgent):
    ARGS = TDQNAgentArgs

    def __init__(
        self,
        args: TDQNAgentArgs,
        env: gym.vector.SyncVectorEnv,
        fabric: Fabric,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
    ):
        # Store temporal parameters
        self.option_lengths = args.option_lengths

        # Calculate expanded action space based on action space type
        if isinstance(action_space, gym.spaces.Discrete):
            n_actions = action_space.n
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            n_actions = action_space.nvec[0]  # Assuming all dimensions have same size
        else:
            raise ValueError("Only Discrete and MultiDiscrete action spaces are supported")

        expanded_action_space = gym.spaces.Discrete(n_actions * len(args.option_lengths))

        super().__init__(args, env, fabric, observation_space, expanded_action_space)

        with fabric.init_module():
            self.q_network = QNetwork(env, action_space_size=expanded_action_space.n).to(
                self.fabric.device, non_blocking=True
            )
            self.optimizer = torch.optim.Adam(
                self.q_network.parameters(),
                lr=args.learning_rate,
                capturable=args.compile,
                foreach=True,
            )
            self.target_network = QNetwork(env, action_space_size=expanded_action_space.n)
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decode_action(self, action):
        """Decode expanded action into base action and option length."""
        base_action = action // len(self.option_lengths)
        repeat_idx = action % len(self.option_lengths)
        option_length = self.option_lengths[repeat_idx]
        return base_action, option_length

    @staticmethod
    @torch.no_grad()
    def _compute_td_targets(
        target_network: QNetwork,
        data: ReplayBufferSamples,
        gamma: float,
        option_lengths: list,
    ) -> torch.Tensor:
        """Compute TD targets for temporal Q-learning update."""
        next_q_values = target_network(data.next_observations)
        next_q_max, greedy_actions = next_q_values.max(dim=1)

        # Get option lengths for next actions
        option_lengths_tensor = torch.tensor(option_lengths, device=next_q_max.device)
        repeat_amounts = option_lengths_tensor.gather(0, data.actions % len(option_lengths))

        # Apply temporal discounting based on option lengths and handle termination vs truncation
        # For termination, we don't bootstrap. For truncation, we do.
        return data.rewards.flatten() + (gamma**repeat_amounts) * next_q_max * (1 - data.dones.flatten())
