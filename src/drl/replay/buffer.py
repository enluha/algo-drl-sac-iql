"""Simple replay buffer helpers (wrapper around d3rlpy if needed)."""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np


class SimpleReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=capacity)

    def append(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        obs, actions, rewards, next_obs, dones = zip(*(self.buffer[i] for i in idx))
        return (
            np.stack(obs),
            np.stack(actions),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_obs),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
