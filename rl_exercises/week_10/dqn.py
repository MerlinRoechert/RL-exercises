##
## This file is copied from week 4
##

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_exercises.week_10.buffers import ReplayBuffer
from rl_exercises.week_10.networks import QNetwork


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """Seed NumPy, PyTorch and the Gymnasium environment."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent:
    """DQN agent copied locally for the Week 10 HPO experiment."""

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        hidden_dim: int = 64,
        num_layers: int = 2,
        seed: int = 0,
    ) -> None:
        self.env = env
        set_seed(env, seed)

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.q = QNetwork(obs_dim, n_actions, hidden_dim, num_layers)
        self.target_q = QNetwork(obs_dim, n_actions, hidden_dim, num_layers)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.total_steps = 0

    def epsilon(self) -> float:
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(
            -self.total_steps / self.epsilon_decay
        )

    def predict_action(
        self, state: np.ndarray, info: Dict[str, Any] | None = None, evaluate: bool = False
    ) -> int:
        if evaluate or np.random.rand() >= self.epsilon():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q(state_t)
            return int(torch.argmax(q_values, dim=1).item())

        return int(self.env.action_space.sample())

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32)

        prediction = self.q(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_q(next_states_t).max(1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(self, num_frames: int, verbose: bool = False) -> List[float]:
        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_rewards: List[float] = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            episode_reward += reward

            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                self.update_agent(batch)

            if done or truncated:
                state, _ = self.env.reset()
                episode_rewards.append(episode_reward)
                episode_reward = 0.0

                if verbose and len(episode_rewards) % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(
                        f"Frame {frame}, AvgReward(10): {avg_reward:.2f}, "
                        f"epsilon={self.epsilon():.3f}"
                    )

        return episode_rewards
