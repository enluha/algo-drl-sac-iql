"""Replay buffer utilities placeholder for potential custom extensions."""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np


class SimpleReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, actions, rewards, next_obs, dones = zip(*(self.buffer[idx] for idx in indices))
        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self.buffer)
