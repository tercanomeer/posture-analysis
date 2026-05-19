from __future__ import annotations

import time
from typing import Optional


class FPSCounter:
    """Exponential-moving-average FPS estimator.

    ``alpha`` is the weight of the newest sample; larger values react faster,
    smaller values produce a steadier readout.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._last: Optional[float] = None
        self._fps: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0.0:
                instant = 1.0 / dt
                self._fps = (
                    instant
                    if self._fps == 0.0
                    else self._alpha * instant + (1.0 - self._alpha) * self._fps
                )
        self._last = now
        return self._fps

    @property
    def value(self) -> float:
        return self._fps

    def reset(self) -> None:
        self._last = None
        self._fps = 0.0
