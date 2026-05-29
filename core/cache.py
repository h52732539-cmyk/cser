"""Shared frame cache (LRU). All tasks read from the same cache."""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np


class SharedFrameCache:
    """Simple OrderedDict-based LRU."""

    def __init__(self, max_size: int = 500) -> None:
        self.max_size = max_size
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def put(self, frame_idx: int, image: np.ndarray) -> None:
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            return
        self._cache[frame_idx] = image
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def get(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            self.hits += 1
            return self._cache[frame_idx]
        self.misses += 1
        return None

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._cache)
