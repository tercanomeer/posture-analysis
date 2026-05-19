from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from mediapipe.tasks.python.vision import PoseLandmarksConnections

from .detector import PoseResult

Color = Tuple[int, int, int]

_POSE_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
)


class PoseRenderer:
    """Draws pose skeleton and HUD elements onto frames in place."""

    def __init__(
        self,
        landmark_color: Color = (0, 255, 0),
        connection_color: Color = (255, 255, 255),
        landmark_radius: int = 3,
        connection_thickness: int = 2,
        visibility_threshold: float = 0.5,
        hud_color: Color = (0, 255, 0),
    ) -> None:
        self._landmark_color = landmark_color
        self._connection_color = connection_color
        self._landmark_radius = landmark_radius
        self._connection_thickness = connection_thickness
        self._visibility_threshold = visibility_threshold
        self._hud_color = hud_color

    def draw_skeleton(self, frame: np.ndarray, result: PoseResult) -> np.ndarray:
        if not result.detected:
            return frame
        h, w = frame.shape[:2]
        landmarks = result.landmarks
        # Pre-project once; reuse for both edges and joints.
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        vis = [getattr(lm, "visibility", 1.0) for lm in landmarks]
        threshold = self._visibility_threshold

        for start, end in _POSE_CONNECTIONS:
            if vis[start] < threshold or vis[end] < threshold:
                continue
            cv2.line(
                frame,
                points[start],
                points[end],
                self._connection_color,
                self._connection_thickness,
                cv2.LINE_AA,
            )

        for (x, y), v in zip(points, vis):
            if v < threshold:
                continue
            cv2.circle(frame, (x, y), self._landmark_radius, self._landmark_color, -1, cv2.LINE_AA)

        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        text = f"FPS: {fps:5.1f}"
        origin = (12, 32)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        # Black outline first for legibility on bright backgrounds.
        cv2.putText(frame, text, origin, font, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, origin, font, scale, self._hud_color, 2, cv2.LINE_AA)
        return frame
