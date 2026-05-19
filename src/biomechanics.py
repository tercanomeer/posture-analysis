"""Biomechanics primitives and posture metrics.

Two layers:

1. Pure-NumPy geometry primitives (``angle_between``, ``angles_batch``,
   ``signed_angle_from_horizontal``, ``signed_angle_from_vertical``) — usable
   on any 2D/3D points, completely independent of the rest of the project.

2. Pose-aware helpers (``neck_angle``, ``shoulder_slope``,
   ``torso_inclination`` and ``PostureAnalyzer``) that consume a ``Pose`` from
   ``landmarks.py`` and produce posture metrics suitable for live display or
   downstream analytics.

Invalid inputs (degenerate vectors, occluded landmarks below the visibility
threshold) yield ``math.nan`` rather than raising — this keeps the realtime
pipeline crash-free when the subject is partially out of frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .landmarks import Landmark, Pose, PoseLandmark

ArrayLike = Sequence[float]
Point2D = Tuple[float, float]

_EPS = 1e-9


# ────────────────────────────────────────────────────────────────────────────
# Geometry primitives
# ────────────────────────────────────────────────────────────────────────────

def angle_between(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> float:
    """Return the angle ∠ABC in degrees at vertex ``b``.

    Inputs may be any array-like of length 2 or 3 (e.g. tuples, lists, or
    NumPy arrays). Returns ``math.nan`` if either segment has zero length —
    that's the only invalid case, so callers don't need to special-case it.
    """
    pa = np.asarray(a, dtype=np.float64)
    pb = np.asarray(b, dtype=np.float64)
    pc = np.asarray(c, dtype=np.float64)
    ba = pa - pb
    bc = pc - pb
    na = float(np.linalg.norm(ba))
    nc = float(np.linalg.norm(bc))
    if na < _EPS or nc < _EPS:
        return math.nan
    # Clip guards against floating-point overshoot like 1.0000001 → NaN from acos.
    cos = float(np.dot(ba, bc)) / (na * nc)
    if cos > 1.0:
        cos = 1.0
    elif cos < -1.0:
        cos = -1.0
    return math.degrees(math.acos(cos))


def angles_batch(triplets: np.ndarray) -> np.ndarray:
    """Vectorized angle computation for many (A, B, C) triples at once.

    ``triplets`` has shape ``(N, 3, D)`` with ``D in {2, 3}``. Returns an
    array of shape ``(N,)`` in degrees; degenerate rows are filled with NaN.
    """
    arr = np.asarray(triplets, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3, D); got {arr.shape}")
    a, b, c = arr[:, 0], arr[:, 1], arr[:, 2]
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba, axis=1)
    nc = np.linalg.norm(bc, axis=1)
    valid = (na > _EPS) & (nc > _EPS)
    denom = np.where(valid, na * nc, 1.0)
    cos = np.clip(np.einsum("ij,ij->i", ba, bc) / denom, -1.0, 1.0)
    out = np.degrees(np.arccos(cos))
    out[~valid] = np.nan
    return out


def signed_angle_from_horizontal(p_from: Point2D, p_to: Point2D) -> float:
    """Signed angle of the segment ``p_from → p_to`` measured from horizontal.

    Positive means the destination sits *above* the source in screen space
    (i.e. smaller y). Useful for shoulder-slope analysis where a non-zero
    value indicates left/right asymmetry.
    """
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    if abs(dx) < _EPS and abs(dy) < _EPS:
        return math.nan
    # Image y axis points down; flip dy so "above" is positive.
    return math.degrees(math.atan2(-dy, dx))


def signed_angle_from_vertical(p_from: Point2D, p_to: Point2D) -> float:
    """Signed angle of the segment ``p_from → p_to`` measured from vertical-up.

    Zero means perfectly upright (destination directly above source). Positive
    means the destination leans to the right of the source in screen space.
    Useful for torso-inclination analysis.
    """
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    if abs(dx) < _EPS and abs(dy) < _EPS:
        return math.nan
    # Image y axis points down; the "up" reference is (0, -1).
    return math.degrees(math.atan2(dx, -dy))


# ────────────────────────────────────────────────────────────────────────────
# Pose-aware helpers
# ────────────────────────────────────────────────────────────────────────────

def _midpoint(a: Landmark, b: Landmark) -> Tuple[float, float, float]:
    return (
        (a.x + b.x) * 0.5,
        (a.y + b.y) * 0.5,
        (a.z + b.z) * 0.5,
    )


def _all_visible(landmarks: Sequence[Landmark], threshold: float) -> bool:
    return all(lm.visibility >= threshold for lm in landmarks)


def neck_angle(pose: Optional[Pose], visibility_threshold: float = 0.5) -> float:
    """Forward-flexion angle of the head/neck.

    Measured as ∠(hip-mid, shoulder-mid, ear-mid) in 2D image coordinates.
    180° = head stacked above the torso (ideal posture); smaller values
    indicate forward-head flexion. Returns NaN if pose is missing or any of
    the contributing landmarks fall below the visibility threshold.
    """
    if pose is None:
        return math.nan
    ls = pose.get(PoseLandmark.LEFT_SHOULDER)
    rs = pose.get(PoseLandmark.RIGHT_SHOULDER)
    lh = pose.get(PoseLandmark.LEFT_HIP)
    rh = pose.get(PoseLandmark.RIGHT_HIP)
    le = pose.get(PoseLandmark.LEFT_EAR)
    re = pose.get(PoseLandmark.RIGHT_EAR)
    if not _all_visible((ls, rs, lh, rh, le, re), visibility_threshold):
        return math.nan
    sm = _midpoint(ls, rs)
    hm = _midpoint(lh, rh)
    em = _midpoint(le, re)
    return angle_between(hm[:2], sm[:2], em[:2])


def shoulder_slope(pose: Optional[Pose], visibility_threshold: float = 0.5) -> float:
    """Tilt of the shoulder line from horizontal, in degrees.

    Zero means level. Positive = right shoulder higher than left in the
    image; negative = left shoulder higher. Returns NaN when either shoulder
    landmark is occluded.
    """
    if pose is None:
        return math.nan
    ls = pose.get(PoseLandmark.LEFT_SHOULDER)
    rs = pose.get(PoseLandmark.RIGHT_SHOULDER)
    if not _all_visible((ls, rs), visibility_threshold):
        return math.nan
    return signed_angle_from_horizontal((ls.x, ls.y), (rs.x, rs.y))


def torso_inclination(pose: Optional[Pose], visibility_threshold: float = 0.5) -> float:
    """Lean of the torso from vertical, in degrees.

    Measured as the signed angle of the hip-midpoint → shoulder-midpoint
    vector relative to vertical-up. Zero = upright; positive = leaning right.
    Returns NaN when any of the four contributing landmarks are occluded.
    """
    if pose is None:
        return math.nan
    ls = pose.get(PoseLandmark.LEFT_SHOULDER)
    rs = pose.get(PoseLandmark.RIGHT_SHOULDER)
    lh = pose.get(PoseLandmark.LEFT_HIP)
    rh = pose.get(PoseLandmark.RIGHT_HIP)
    if not _all_visible((ls, rs, lh, rh), visibility_threshold):
        return math.nan
    sm = _midpoint(ls, rs)
    hm = _midpoint(lh, rh)
    return signed_angle_from_vertical((hm[0], hm[1]), (sm[0], sm[1]))


# ────────────────────────────────────────────────────────────────────────────
# Aggregate analyzer
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PostureMetrics:
    """Snapshot of posture-relevant angles for a single frame.

    Any field may be ``math.nan`` if the underlying landmarks were occluded
    or absent — consumers should use ``math.isnan`` before displaying.
    """

    neck_angle: float
    shoulder_slope: float
    torso_inclination: float

    @classmethod
    def empty(cls) -> "PostureMetrics":
        return cls(math.nan, math.nan, math.nan)


class PostureAnalyzer:
    """Computes posture metrics from a ``Pose``.

    Stateless and reusable. Configure once with a visibility threshold and
    call ``analyze(pose)`` per frame in real-time loops.
    """

    def __init__(self, visibility_threshold: float = 0.5) -> None:
        if not 0.0 <= visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold must be in [0, 1]")
        self._threshold = visibility_threshold

    def analyze(self, pose: Optional[Pose]) -> PostureMetrics:
        if pose is None:
            return PostureMetrics.empty()
        t = self._threshold
        return PostureMetrics(
            neck_angle=neck_angle(pose, t),
            shoulder_slope=shoulder_slope(pose, t),
            torso_inclination=torso_inclination(pose, t),
        )
