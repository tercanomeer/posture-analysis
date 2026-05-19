from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


@dataclass(frozen=True)
class RawPoseResult:
    """MediaPipe-native pose output for a single frame.

    Exists only at the detector boundary. Other layers should consume a
    ``landmarks.Pose`` produced by ``LandmarkExtractor``.
    """

    landmarks: Optional[List]
    image_size: tuple[int, int]

    @property
    def detected(self) -> bool:
        return self.landmarks is not None


class PoseDetector:
    """Thin wrapper around MediaPipe Tasks ``PoseLandmarker`` (VIDEO mode)."""

    def __init__(
        self,
        model_path: str,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        # VIDEO mode requires monotonically increasing millisecond timestamps.
        self._t0 = time.perf_counter()

    def detect(self, bgr_frame: np.ndarray) -> RawPoseResult:
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        return RawPoseResult(landmarks=landmarks, image_size=(w, h))

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
