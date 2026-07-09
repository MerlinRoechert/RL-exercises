##
## This file is copied from week 4
##

from typing import Any, Dict, List, Tuple

import numpy as np


class ReplayBuffer:
    """Simple FIFO replay buffer for DQN transitions."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.actions: List[int | float] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.infos: List[Dict] = []

    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.infos.pop(0)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.infos.append(info)

    def sample(
        self, batch_size: int = 32
    ) -> List[Tuple[Any, Any, float, Any, bool, Dict]]:
        idxs = np.random.choice(len(self.states), batch_size, replace=False)
        return [
            (
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.dones[i],
                self.infos[i],
            )
            for i in idxs
        ]

    def __len__(self) -> int:
        return len(self.states)
