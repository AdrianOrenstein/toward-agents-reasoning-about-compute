import torch
import torch.nn as nn
from typing import Tuple
import gymnasium as gym
from lightning.fabric import Fabric
from dataclasses import dataclass
from src.dqn.compressed_rbuffer import CompressedReplayBuffer
from src.dqn.network import QNetwork
from src.dqn.utils import ReplayBufferSamples
import numpy as np


@dataclass
class AgentArgs:
    """Arguments specific to the DQN agent algorithm."""

    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 2_500
    """the updates until we move the target network"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_ends: int = 1_000_000
    """the frames it takes from start-e to go end-e"""
    learning_starts: int = 12_500
    """how many steps should be in the replay buffer before learning starts"""
    train_frequency: int = 4
    """how many steps until we do an update"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    compile: bool = True
    """whether the policy and update should be compiled"""
    cudagraphs: bool = True
    """whether to use cudagraphs on top of compile."""


class DQNAgent:
    ARGS = AgentArgs

    def __init__(
        self,
        args: AgentArgs,
        env: gym.vector.SyncVectorEnv,
        fabric: Fabric,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
    ):
        """Initialize DQN Agent.

        Args:
            args: Arguments specific to the DQN agent
            env: Training environment
            fabric: Lightning Fabric instance
            observation_space: Observation space of the environment
            action_space: Action space of the environment
        """
        self.args = args
        self.env = env
        self.fabric = fabric
        self.observation_space = observation_space
        self.action_space = action_space

        self.update_count = 0
        self.target_update_count = 0

        # Setup networks and ensure they're on the correct device
        with fabric.init_module():
            self.q_network = QNetwork(env).to(self.fabric.device, non_blocking=True)
            self.optimizer = torch.optim.Adam(
                self.q_network.parameters(),
                lr=args.learning_rate,
                capturable=args.compile,
                foreach=True,
            )
            self.target_network = QNetwork(env)
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Setup replay buffer
        self.replay_buffer = CompressedReplayBuffer(
            buffer_size=args.buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=self.fabric.device,
            compression_level=1,
        )

        # Compile models only if using CUDA
        if args.compile and torch.cuda.is_available():
            compile_options = {
                "triton.cudagraphs": args.cudagraphs,
                "shape_padding": True,
                "max_autotune": True,
            }
            # Remove update compilation and compile networks directly
            self.policy = torch.compile(self.policy, options=compile_options, fullgraph=True)
            self.q_network = torch.compile(self.q_network, options=compile_options, fullgraph=True)
            self.target_network = torch.compile(self.target_network, options=compile_options, fullgraph=True)
            self._update = torch.compile(self._update, options=compile_options, fullgraph=False)

        # Prepare models with Fabric
        self.q_network, self.optimizer = fabric.setup(self.q_network, self.optimizer, _reapply_compile=True)
        self.target_network = fabric.setup_module(self.target_network, _reapply_compile=True)

        self.agent_decisions_made_so_far = 0

    def _create_target_network(self, q_network):
        target_net = type(q_network)(self.env)
        target_net.load_state_dict(q_network.state_dict())
        return target_net

    def store_experience(self, obs, next_obs, action, reward, done, info):
        """Store experience in replay buffer"""
        self.replay_buffer.add(obs, next_obs, action, reward, done, info)

    @staticmethod
    @torch.no_grad()
    def policy(obs: torch.Tensor, q_network: QNetwork):
        """Select actions based on the Q-network."""
        q_values = q_network(obs)
        return torch.argmax(q_values, dim=1)

    def get_exploration_rate(self, frame_no: int) -> float:
        """Get the current exploration rate based on the frame number."""
        return max(
            self.args.end_e,
            self.args.start_e + (self.args.end_e - self.args.start_e) * (frame_no / self.args.exploration_ends),
        )

    def get_exploration_action(self) -> np.ndarray:
        """Get an exploration action."""
        return self.action_space.sample()

    @staticmethod
    @torch.no_grad()
    def _compute_td_targets(
        target_network: QNetwork,
        data: ReplayBufferSamples,
        gamma: float,
    ) -> torch.Tensor:
        """Compute TD targets for the Q-learning update.

        Returns:
            td_target: The computed TD target values
        """
        next_q_values = target_network(data.next_observations)
        next_q_max, _ = next_q_values.max(dim=1)
        return data.rewards.flatten() + gamma * next_q_max * (1 - data.dones.flatten())

    @staticmethod
    def _update(
        fabric: Fabric,
        q_network: QNetwork,
        target_network: QNetwork,
        optimizer: torch.optim.Optimizer,
        data: ReplayBufferSamples,
        gamma: float,
    ):
        td_targets = DQNAgent._compute_td_targets(target_network, data, gamma)

        est_q_values: torch.Tensor = q_network(data.observations).gather(1, data.actions).squeeze()
        loss = nn.functional.mse_loss(td_targets, est_q_values)

        optimizer.zero_grad(set_to_none=True)
        fabric.backward(loss)
        fabric.clip_gradients(q_network, optimizer, max_norm=10.0)
        optimizer.step()

        return loss, est_q_values

    def update(self, data: ReplayBufferSamples) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the Q-network using a batch of experiences from the replay buffer."""
        loss, est_q_values = self._update(
            fabric=self.fabric,
            q_network=self.q_network,
            target_network=self.target_network,
            optimizer=self.optimizer,
            data=data,
            gamma=self.args.gamma,
        )
        self.update_count += 1

        if self.update_count % self.args.target_network_frequency == 0:
            self.target_update_count += 1
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.detach(), est_q_values.detach()

    def save(self, path):
        """Save model weights"""
        self.fabric.save(path, {"q_network": self.q_network})

    def load(self, path):
        """Load model weights"""
        state_dict = self.fabric.load(path)
        self.q_network.load_state_dict(state_dict["q_network"])
