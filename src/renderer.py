from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .biomechanics import PostureMetrics
from .landmarks import POSE_CONNECTIONS, Pose

Color = Tuple[int, int, int]


class PoseRenderer:
    """Draws a ``Pose`` skeleton and the FPS/metrics HUD onto frames in place.

    Depends only on the ``landmarks`` and ``biomechanics`` modules — never
    imports MediaPipe directly. Posture-feedback visuals (border, banner,
    warnings) live in ``FeedbackRenderer`` so this class stays focused on
    geometric/numeric overlays.
    """

    def __init__(
        self,
        landmark_color: Color = (0, 255, 0),
        connection_color: Color = (255, 255, 255),
        landmark_radius: int = 3,
        connection_thickness: int = 2,
        visibility_threshold: float = 0.5,
        hud_color: Color = (0, 255, 0),
        hud_top: int = 80,  # leave room for FeedbackRenderer's banner.
    ) -> None:
        self._landmark_color = landmark_color
        self._connection_color = connection_color
        self._landmark_radius = landmark_radius
        self._connection_thickness = connection_thickness
        self._visibility_threshold = visibility_threshold
        self._hud_color = hud_color
        self._hud_top = hud_top

    def draw_skeleton(self, frame: np.ndarray, pose: Optional[Pose]) -> np.ndarray:
        if pose is None:
            return frame
        points = pose.pixels
        landmarks = pose.landmarks
        threshold = self._visibility_threshold

        for start, end in POSE_CONNECTIONS:
            if (
                landmarks[start].visibility < threshold
                or landmarks[end].visibility < threshold
            ):
                continue
            cv2.line(
                frame,
                points[start],
                points[end],
                self._connection_color,
                self._connection_thickness,
                cv2.LINE_AA,
            )

        for (x, y), lm in zip(points, landmarks):
            if lm.visibility < threshold:
                continue
            cv2.circle(
                frame, (x, y), self._landmark_radius, self._landmark_color, -1, cv2.LINE_AA
            )
        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        return self._draw_hud_text(frame, f"FPS: {fps:5.1f}", origin=(12, self._hud_top))

    def draw_metrics(self, frame: np.ndarray, metrics: PostureMetrics) -> np.ndarray:
        lines = (
            f"Neck:     {self._fmt(metrics.neck_angle)}",
            f"Shoulder: {self._fmt(metrics.shoulder_slope)}",
            f"Torso:    {self._fmt(metrics.torso_inclination)}",
        )
        for i, line in enumerate(lines):
            self._draw_hud_text(
                frame, line, origin=(12, self._hud_top + 32 + i * 28), scale=0.65
            )
        return frame

    @staticmethod
    def _fmt(value: float) -> str:
        return "  --" if math.isnan(value) else f"{value:6.1f} deg"

    def _draw_hud_text(
        self,
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        scale: float = 0.8,
        color: Optional[Color] = None,
    ) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fill = color if color is not None else self._hud_color
        # Black outline first for legibility on bright backgrounds.
        cv2.putText(frame, text, origin, font, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, origin, font, scale, fill, 2, cv2.LINE_AA)
        return frame
