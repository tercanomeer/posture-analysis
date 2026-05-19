"""Real-time visual posture feedback overlay.

Consumes a ``PostureAssessment`` and composes three lightweight feedback
elements onto a BGR frame:

- A thick colored **border** around the entire frame.
- A solid colored **status banner** at the top.
- A vertical stack of **warning chips** at the bottom-right, one per
  mild/severe finding.

All draws are plain ``cv2.rectangle`` / ``cv2.putText`` calls — no alpha
blending, no per-pixel work, no allocations beyond what OpenCV does
internally. Safe to call every frame in a realtime loop.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .classifier import PostureAssessment, Severity

Color = Tuple[int, int, int]


_FEEDBACK_COLORS: dict[Severity, Color] = {
    Severity.OK: (0, 200, 0),
    Severity.MILD: (0, 200, 220),
    Severity.SEVERE: (0, 32, 220),
    Severity.UNKNOWN: (160, 160, 160),
}

_STATUS_TEXT: dict[Severity, str] = {
    Severity.OK: "POSTURE: GOOD",
    Severity.MILD: "POSTURE: ADJUST",
    Severity.SEVERE: "POSTURE: WARNING",
    Severity.UNKNOWN: "POSTURE: NO DETECTION",
}

_TEXT_DARK: Color = (20, 20, 20)


class FeedbackRenderer:
    """Renders posture-feedback overlays driven by a ``PostureAssessment``."""

    def __init__(
        self,
        border_thickness: int = 10,
        banner_height: int = 44,
        chip_height: int = 32,
        chip_gap: int = 6,
        margin: int = 12,
    ) -> None:
        self._border = border_thickness
        self._banner_h = banner_height
        self._chip_h = chip_height
        self._chip_gap = chip_gap
        self._margin = margin

    def draw(self, frame: np.ndarray, assessment: PostureAssessment) -> np.ndarray:
        severity = assessment.overall
        color = _FEEDBACK_COLORS[severity]
        self._draw_banner(frame, color, _STATUS_TEXT[severity])
        self._draw_border(frame, color)
        self._draw_warnings(frame, assessment)
        return frame

    # ── Component draws ────────────────────────────────────────────────────

    def _draw_border(self, frame: np.ndarray, color: Color) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, self._border)

    def _draw_banner(self, frame: np.ndarray, color: Color, text: str) -> None:
        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, self._banner_h), color, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        origin = ((w - tw) // 2, (self._banner_h + th) // 2)
        cv2.putText(frame, text, origin, font, scale, _TEXT_DARK, thickness, cv2.LINE_AA)

    def _draw_warnings(self, frame: np.ndarray, assessment: PostureAssessment) -> None:
        problems = [
            (label, sev) for label, sev in assessment
            if sev in (Severity.MILD, Severity.SEVERE)
        ]
        if not problems:
            return
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2
        pad_x, pad_y = 12, 8
        for i, (label, sev) in enumerate(reversed(problems)):
            text = f"WARNING: {label} ({sev.value})"
            (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
            x2 = w - self._margin
            x1 = x2 - tw - 2 * pad_x
            y2 = h - self._margin - i * (self._chip_h + self._chip_gap)
            y1 = y2 - self._chip_h
            cv2.rectangle(frame, (x1, y1), (x2, y2), _FEEDBACK_COLORS[sev], -1)
            cv2.putText(
                frame,
                text,
                (x1 + pad_x, y2 - pad_y),
                font,
                scale,
                _TEXT_DARK,
                thickness,
                cv2.LINE_AA,
            )
