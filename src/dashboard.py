"""Composed real-time posture-analysis dashboard.

A single ``Dashboard`` instance owns the visible UI: a top header strip with
title and FPS, the camera area (where the skeleton overlay is already drawn
in-place), and a right-hand sidebar with three sectioned panels —
``STATUS``, ``ANGLES``, ``WARNINGS``.

Design goals (in priority order):

1. **Real-time.** One pre-allocated canvas, reused across frames. No per-frame
   allocations beyond what OpenCV does internally. Compositing is a single
   memcpy plus a handful of ``cv2.rectangle`` / ``cv2.putText`` calls.
2. **Academic-clean.** Dark slate background, single accent color, generous
   spacing, no decorative chrome. Reads well on a projector at the back of
   a lecture hall.
3. **OpenCV-only.** No Qt/Tk dependency — the same wheel set runs the demo
   from a terminal on any platform.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .biomechanics import PostureMetrics
from .classifier import PostureAssessment, Severity

Color = Tuple[int, int, int]


class Palette:
    """Dashboard color tokens (BGR)."""

    bg = (28, 28, 32)
    panel = (44, 44, 50)
    panel_row = (38, 38, 44)
    divider = (60, 60, 68)
    text = (235, 235, 240)
    text_muted = (160, 160, 170)
    accent = (60, 180, 220)        # academic gold (BGR)
    on_severity = (20, 20, 28)     # dark text used on bright severity fills

    ok = (90, 200, 110)
    mild = (80, 200, 230)
    severe = (90, 90, 235)
    unknown = (140, 140, 150)


_SEVERITY_COLORS: dict[Severity, Color] = {
    Severity.OK: Palette.ok,
    Severity.MILD: Palette.mild,
    Severity.SEVERE: Palette.severe,
    Severity.UNKNOWN: Palette.unknown,
}

_SEVERITY_LABELS: dict[Severity, str] = {
    Severity.OK: "GOOD",
    Severity.MILD: "ADJUST",
    Severity.SEVERE: "WARNING",
    Severity.UNKNOWN: "NO DETECTION",
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Dashboard:
    """Compositor that wraps a camera frame in a sidebar/header UI."""

    def __init__(
        self,
        sidebar_width: int = 340,
        header_height: int = 56,
        title: str = "Posture Analysis  |  Realtime Demo",
    ) -> None:
        self._sidebar_w = sidebar_width
        self._header_h = header_height
        self._title = title
        self._canvas: Optional[np.ndarray] = None
        self._cam_size: Optional[Tuple[int, int]] = None  # (w, h)

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        self._title = value

    def render(
        self,
        frame: np.ndarray,
        metrics: PostureMetrics,
        assessment: PostureAssessment,
        fps: float,
    ) -> np.ndarray:
        cam_h, cam_w = frame.shape[:2]
        canvas = self._ensure_canvas(cam_w, cam_h)
        canvas[:] = Palette.bg

        self._draw_header(canvas, fps)
        # Camera area: composite the raw frame (with skeleton already drawn).
        canvas[self._header_h:self._header_h + cam_h, :cam_w] = frame
        cv2.rectangle(
            canvas,
            (0, self._header_h),
            (cam_w - 1, self._header_h + cam_h - 1),
            Palette.divider,
            1,
        )

        self._draw_sidebar(canvas, cam_w, metrics, assessment)
        return canvas

    # ── canvas management ─────────────────────────────────────────────────

    def _ensure_canvas(self, cam_w: int, cam_h: int) -> np.ndarray:
        if self._canvas is None or self._cam_size != (cam_w, cam_h):
            self._canvas = np.empty(
                (cam_h + self._header_h, cam_w + self._sidebar_w, 3),
                dtype=np.uint8,
            )
            self._cam_size = (cam_w, cam_h)
        return self._canvas

    # ── components ────────────────────────────────────────────────────────

    def _draw_header(self, canvas: np.ndarray, fps: float) -> None:
        h, w = self._header_h, canvas.shape[1]
        cv2.rectangle(canvas, (0, 0), (w, h), Palette.panel, -1)
        cv2.line(canvas, (0, h - 1), (w, h - 1), Palette.accent, 1)

        baseline_y = h - 20
        cv2.putText(
            canvas, self._title, (16, baseline_y),
            _FONT, 0.7, Palette.text, 1, cv2.LINE_AA,
        )
        fps_text = f"FPS  {fps:5.1f}"
        (tw, _), _ = cv2.getTextSize(fps_text, _FONT, 0.7, 1)
        cv2.putText(
            canvas, fps_text, (w - tw - 16, baseline_y),
            _FONT, 0.7, Palette.text, 1, cv2.LINE_AA,
        )

    def _draw_sidebar(
        self,
        canvas: np.ndarray,
        cam_w: int,
        metrics: PostureMetrics,
        assessment: PostureAssessment,
    ) -> None:
        pad = 16
        x0 = cam_w + pad
        inner_w = self._sidebar_w - 2 * pad
        y = self._header_h + pad

        y = self._draw_status_card(canvas, x0, y, inner_w, assessment)
        y += 18
        y = self._draw_angles_panel(canvas, x0, y, inner_w, metrics)
        y += 18
        self._draw_warnings_panel(canvas, x0, y, inner_w, assessment)

    def _draw_status_card(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        assessment: PostureAssessment,
    ) -> int:
        y = self._draw_section_title(canvas, x, y, "STATUS")
        severity = assessment.overall
        color = _SEVERITY_COLORS[severity]
        card_h = 64
        cv2.rectangle(canvas, (x, y), (x + w, y + card_h), color, -1)

        text = f"POSTURE: {_SEVERITY_LABELS[severity]}"
        (tw, _), _ = cv2.getTextSize(text, _FONT, 0.7, 2)
        cv2.putText(
            canvas, text, (x + (w - tw) // 2, y + 41),
            _FONT, 0.7, Palette.on_severity, 2, cv2.LINE_AA,
        )
        return y + card_h

    def _draw_angles_panel(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        metrics: PostureMetrics,
    ) -> int:
        y = self._draw_section_title(canvas, x, y, "ANGLES")
        rows = (
            ("Neck",       metrics.neck_angle),
            ("Shoulders",  metrics.shoulder_slope),
            ("Torso",      metrics.torso_inclination),
        )
        row_h = 30
        for i, (label, value) in enumerate(rows):
            top = y + i * row_h
            cv2.rectangle(
                canvas, (x, top), (x + w, top + row_h - 2),
                Palette.panel if i % 2 == 0 else Palette.panel_row, -1,
            )
            cv2.putText(
                canvas, label, (x + 12, top + 21),
                _FONT, 0.55, Palette.text_muted, 1, cv2.LINE_AA,
            )
            val_text = "  --" if math.isnan(value) else f"{value:6.1f} deg"
            (tw, _), _ = cv2.getTextSize(val_text, _FONT, 0.6, 1)
            cv2.putText(
                canvas, val_text, (x + w - tw - 12, top + 21),
                _FONT, 0.6, Palette.text, 1, cv2.LINE_AA,
            )
        return y + len(rows) * row_h

    def _draw_warnings_panel(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        assessment: PostureAssessment,
    ) -> int:
        y = self._draw_section_title(canvas, x, y, "WARNINGS")
        problems = [
            (label, sev) for label, sev in assessment.findings
            if sev in (Severity.MILD, Severity.SEVERE)
        ]
        if not problems:
            cv2.putText(
                canvas, "no issues detected", (x + 12, y + 22),
                _FONT, 0.55, Palette.text_muted, 1, cv2.LINE_AA,
            )
            return y + 32

        row_h = 32
        for i, (label, sev) in enumerate(problems):
            top = y + i * row_h
            cv2.rectangle(
                canvas, (x, top), (x + w, top + row_h - 4),
                Palette.panel, -1,
            )
            cv2.circle(canvas, (x + 14, top + 14), 5, _SEVERITY_COLORS[sev], -1, cv2.LINE_AA)
            cv2.putText(
                canvas, f"{label}  ({sev.value})", (x + 30, top + 19),
                _FONT, 0.55, Palette.text, 1, cv2.LINE_AA,
            )
        return y + len(problems) * row_h

    def _draw_section_title(self, canvas: np.ndarray, x: int, y: int, text: str) -> int:
        cv2.putText(
            canvas, text, (x, y + 14),
            _FONT, 0.48, Palette.text_muted, 1, cv2.LINE_AA,
        )
        cv2.line(canvas, (x, y + 22), (x + 40, y + 22), Palette.accent, 2)
        return y + 32
