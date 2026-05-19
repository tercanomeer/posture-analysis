"""Lightweight landmark smoothing via exponential moving average.

MediaPipe's per-frame pose estimate is noisy at the sub-pixel level even on
a static subject; that jitter propagates straight into derived angles like
``neck_angle`` and trips the classifier's mild/severe thresholds. This module
applies a per-coordinate EMA on the structured ``Pose`` output to reduce that
noise while keeping latency low.

Design choices:

- **Exponential moving average** (not a moving-average buffer) so smoothing
  state is a single small array — no per-frame history allocation.
- **Separate alpha for visibility** scores. Coordinates can be smoothed
  hard (low alpha) without delaying visibility gate decisions, which need
  to react quickly when the subject leaves the frame.
- **Reset on missing detection.** When ``smooth(None)`` is called, internal
  state is cleared so the next real detection starts fresh — otherwise the
  smoothed pose would drag toward stale positions after a long occlusion.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .landmarks import Landmark, Pose


class LandmarkSmoother:
    """Per-coordinate exponential moving average on ``Pose`` objects.

    ``alpha`` is the weight applied to the *newest* sample. Larger values
    react faster but smooth less; smaller values smooth more but introduce
    perceptible lag.

    Typical ranges:
        * ``alpha = 1.0`` — no smoothing (pass-through).
        * ``alpha ≈ 0.7`` — mild smoothing, near-zero perceived lag.
        * ``alpha ≈ 0.5`` — balanced (default).
        * ``alpha ≈ 0.2`` — strong smoothing, noticeable lag.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        visibility_alpha: float = 0.7,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 < visibility_alpha <= 1.0:
            raise ValueError("visibility_alpha must be in (0, 1]")
        self._alpha = float(alpha)
        self._vis_alpha = float(visibility_alpha)
        self._state: Optional[np.ndarray] = None  # (N, 4): x, y, z, visibility

    def reset(self) -> None:
        self._state = None

    def smooth(self, pose: Optional[Pose]) -> Optional[Pose]:
        if pose is None:
            self._state = None
            return None

        n = len(pose.landmarks)
        current = np.empty((n, 4), dtype=np.float64)
        for i, lm in enumerate(pose.landmarks):
            current[i, 0] = lm.x
            current[i, 1] = lm.y
            current[i, 2] = lm.z
            current[i, 3] = lm.visibility

        if self._state is None or self._state.shape != current.shape:
            self._state = current.copy()
        else:
            a = self._alpha
            va = self._vis_alpha
            s = self._state
            s[:, :3] = a * current[:, :3] + (1.0 - a) * s[:, :3]
            s[:, 3] = va * current[:, 3] + (1.0 - va) * s[:, 3]

        s = self._state
        smoothed = tuple(
            Landmark(
                x=float(s[i, 0]),
                y=float(s[i, 1]),
                z=float(s[i, 2]),
                visibility=float(s[i, 3]),
            )
            for i in range(n)
        )
        return Pose(landmarks=smoothed, image_size=pose.image_size)
