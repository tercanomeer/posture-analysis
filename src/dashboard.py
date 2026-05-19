"""Composed real-time posture-analysis dashboard.

A single ``Dashboard`` instance owns the visible UI: a top header strip with
title and FPS, the camera area (where the skeleton overlay is already drawn
in-place), and a right-hand sidebar with three sectioned panels —
``STATUS``, ``ANGLES``, ``WARNINGS``.

Design goals (in priority order):

1. **Real-time.** A profile revealed that filling a 1620×776×3 canvas with a
   BG color via numpy broadcast cost ~3 ms — 94% of the dashboard render. To
   avoid that hit, all *static* chrome (background, header bg/title/accent,
   sidebar section titles + underlines, ANGLES row stripes + labels) is
   pre-rendered into a template at first use and brought back with a single
   ``np.copyto`` each frame. Only the *dynamic* parts (FPS value, STATUS
   card, angle values, warnings) are re-drawn per frame.
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
_ANGLE_LABELS = ("Neck", "Shoulders", "Torso")


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
        self._chrome: Optional[np.ndarray] = None
        self._cam_size: Optional[Tuple[int, int]] = None
        # Pre-computed layout coordinates; populated when the canvas is built.
        self._angle_row_tops: tuple[int, ...] = ()
        self._angles_value_anchor_x: int = 0
        self._status_card_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._warnings_top: int = 0
        self._sidebar_inner_x: int = 0
        self._sidebar_inner_w: int = 0
        self._fps_text_baseline_y: int = 0

    @property
    def title(self) -> str:
        return self._title

    def render(
        self,
        frame: np.ndarray,
        metrics: PostureMetrics,
        assessment: PostureAssessment,
        fps: float,
    ) -> np.ndarray:
        cam_h, cam_w = frame.shape[:2]
        self._ensure_canvas(cam_w, cam_h)
        canvas = self._canvas
        assert canvas is not None and self._chrome is not None

        # 1. Restore static chrome with a single fast memcpy.
        np.copyto(canvas, self._chrome)

        # 2. Composite the camera frame into the camera region.
        canvas[self._header_h:self._header_h + cam_h, :cam_w] = frame

        # 3. Camera-area border (drawn after frame copy so it isn't overwritten).
        cv2.rectangle(
            canvas,
            (0, self._header_h),
            (cam_w - 1, self._header_h + cam_h - 1),
            Palette.divider,
            1,
        )

        # 4. Dynamic elements only.
        self._draw_fps_value(canvas, fps)
        self._draw_status_card(canvas, assessment)
        self._draw_angle_values(canvas, metrics)
        self._draw_warnings(canvas, assessment)
        return canvas

    # ── canvas / chrome lifecycle ─────────────────────────────────────────

    def _ensure_canvas(self, cam_w: int, cam_h: int) -> None:
        if self._cam_size == (cam_w, cam_h) and self._canvas is not None:
            return
        out_h = cam_h + self._header_h
        out_w = cam_w + self._sidebar_w
        self._canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
        self._chrome = np.empty_like(self._canvas)
        self._cam_size = (cam_w, cam_h)
        self._compute_layout(cam_w)
        self._build_chrome(cam_w, cam_h)

    def _compute_layout(self, cam_w: int) -> None:
        pad = 16
        self._sidebar_inner_x = cam_w + pad
        self._sidebar_inner_w = self._sidebar_w - 2 * pad

        # Header
        self._fps_text_baseline_y = self._header_h - 20

        # Sidebar vertical flow: STATUS title → STATUS card → ANGLES title → 3 rows → WARNINGS title → list
        y = self._header_h + pad
        # STATUS title takes 32 px (title text + accent line).
        y_after_status_title = y + 32
        status_card_h = 64
        self._status_card_box = (
            self._sidebar_inner_x,
            y_after_status_title,
            self._sidebar_inner_x + self._sidebar_inner_w,
            y_after_status_title + status_card_h,
        )
        y = y_after_status_title + status_card_h + 18

        # ANGLES section
        y_after_angles_title = y + 32
        row_h = 30
        self._angle_row_tops = tuple(
            y_after_angles_title + i * row_h for i in range(len(_ANGLE_LABELS))
        )
        # Right-anchor X for value text:
        self._angles_value_anchor_x = (
            self._sidebar_inner_x + self._sidebar_inner_w - 12
        )
        y = y_after_angles_title + len(_ANGLE_LABELS) * row_h + 18

        # WARNINGS title + list start
        self._warnings_top = y + 32

    def _build_chrome(self, cam_w: int, cam_h: int) -> None:
        c = self._chrome
        assert c is not None
        # Solid background — broadcast assignment dominates render cost when
        # done every frame; here it's done once.
        c[:] = Palette.bg

        # Header background + accent line + title.
        cv2.rectangle(c, (0, 0), (c.shape[1], self._header_h), Palette.panel, -1)
        cv2.line(c, (0, self._header_h - 1), (c.shape[1], self._header_h - 1),
                 Palette.accent, 1)
        cv2.putText(c, self._title, (16, self._fps_text_baseline_y),
                    _FONT, 0.7, Palette.text, 1, cv2.LINE_AA)

        # Section titles & accent underlines.
        pad = 16
        x = self._sidebar_inner_x
        y = self._header_h + pad
        self._draw_section_title(c, x, y, "STATUS")

        y = y + 32 + 64 + 18  # below STATUS card
        self._draw_section_title(c, x, y, "ANGLES")

        # ANGLES row stripes + left-side labels (static).
        row_h = 30
        for i, label in enumerate(_ANGLE_LABELS):
            top = self._angle_row_tops[i]
            fill = Palette.panel if i % 2 == 0 else Palette.panel_row
            cv2.rectangle(c, (x, top), (x + self._sidebar_inner_w,
                                        top + row_h - 2), fill, -1)
            cv2.putText(c, label, (x + 12, top + 21),
                        _FONT, 0.55, Palette.text_muted, 1, cv2.LINE_AA)

        y = y + 32 + len(_ANGLE_LABELS) * row_h + 18
        self._draw_section_title(c, x, y, "WARNINGS")

    # ── static helpers ────────────────────────────────────────────────────

    def _draw_section_title(self, c: np.ndarray, x: int, y: int, text: str) -> None:
        cv2.putText(c, text, (x, y + 14),
                    _FONT, 0.48, Palette.text_muted, 1, cv2.LINE_AA)
        cv2.line(c, (x, y + 22), (x + 40, y + 22), Palette.accent, 2)

    # ── dynamic per-frame draws ───────────────────────────────────────────

    def _draw_fps_value(self, c: np.ndarray, fps: float) -> None:
        text = f"FPS  {fps:5.1f}"
        (tw, _), _ = cv2.getTextSize(text, _FONT, 0.7, 1)
        cv2.putText(c, text, (c.shape[1] - tw - 16, self._fps_text_baseline_y),
                    _FONT, 0.7, Palette.text, 1, cv2.LINE_AA)

    def _draw_status_card(self, c: np.ndarray, assessment: PostureAssessment) -> None:
        x1, y1, x2, y2 = self._status_card_box
        severity = assessment.overall
        cv2.rectangle(c, (x1, y1), (x2, y2), _SEVERITY_COLORS[severity], -1)
        text = f"POSTURE: {_SEVERITY_LABELS[severity]}"
        (tw, _), _ = cv2.getTextSize(text, _FONT, 0.7, 2)
        cv2.putText(c, text, (x1 + ((x2 - x1) - tw) // 2, y1 + 41),
                    _FONT, 0.7, Palette.on_severity, 2, cv2.LINE_AA)

    def _draw_angle_values(self, c: np.ndarray, metrics: PostureMetrics) -> None:
        values = (metrics.neck_angle, metrics.shoulder_slope, metrics.torso_inclination)
        for i, value in enumerate(values):
            text = "  --" if math.isnan(value) else f"{value:6.1f} deg"
            (tw, _), _ = cv2.getTextSize(text, _FONT, 0.6, 1)
            top = self._angle_row_tops[i]
            cv2.putText(c, text, (self._angles_value_anchor_x - tw, top + 21),
                        _FONT, 0.6, Palette.text, 1, cv2.LINE_AA)

    def _draw_warnings(self, c: np.ndarray, assessment: PostureAssessment) -> None:
        x = self._sidebar_inner_x
        w = self._sidebar_inner_w
        y = self._warnings_top
        problems = [
            (label, sev) for label, sev in assessment.findings
            if sev in (Severity.MILD, Severity.SEVERE)
        ]
        if not problems:
            cv2.putText(c, "no issues detected", (x + 12, y + 22),
                        _FONT, 0.55, Palette.text_muted, 1, cv2.LINE_AA)
            return
        row_h = 32
        for i, (label, sev) in enumerate(problems):
            top = y + i * row_h
            cv2.rectangle(c, (x, top), (x + w, top + row_h - 4), Palette.panel, -1)
            cv2.circle(c, (x + 14, top + 14), 5, _SEVERITY_COLORS[sev],
                       -1, cv2.LINE_AA)
            cv2.putText(c, f"{label}  ({sev.value})", (x + 30, top + 19),
                        _FONT, 0.55, Palette.text, 1, cv2.LINE_AA)
