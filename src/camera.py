from __future__ import annotations

from typing import Iterator, Optional

import cv2
import numpy as np


class Webcam:
    """Context-managed webcam capture with low-latency defaults."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "Webcam":
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source {self._source!r}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        # Minimize internal buffering so we always process the most recent frame.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("Webcam must be used as a context manager")
        ok, frame = self._cap.read()
        return frame if ok else None

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame
