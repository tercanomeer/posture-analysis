from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Iterator, Optional, Tuple

from mediapipe.tasks.python.vision import PoseLandmarksConnections


class PoseLandmark(IntEnum):
    """Names for the 33 landmarks emitted by MediaPipe Pose."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Pose topology pulled from MediaPipe once at import so downstream modules
# (renderer, analytics) don't need to import mediapipe themselves.
POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = tuple(
    (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
)


@dataclass(frozen=True, slots=True)
class Landmark:
    """A single landmark in image-normalized coordinates.

    ``x`` and ``y`` are in ``[0, 1]`` relative to the image; ``z`` shares the
    same scale as ``x`` and is roughly hip-centered. ``visibility`` is
    MediaPipe's predicted presence/visibility score in ``[0, 1]``.
    """

    x: float
    y: float
    z: float
    visibility: float

    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        return int(self.x * width), int(self.y * height)


class Pose:
    """An ordered collection of 33 landmarks plus the originating image size.

    Pixel projections are computed lazily on first access and cached so the
    renderer can fetch them in O(1) on subsequent calls within the same frame.
    """

    __slots__ = ("landmarks", "image_size", "_pixels")

    def __init__(
        self,
        landmarks: Tuple[Landmark, ...],
        image_size: Tuple[int, int],
    ) -> None:
        self.landmarks: Tuple[Landmark, ...] = landmarks
        self.image_size: Tuple[int, int] = image_size
        self._pixels: Optional[Tuple[Tuple[int, int], ...]] = None

    def __len__(self) -> int:
        return len(self.landmarks)

    def __iter__(self) -> Iterator[Landmark]:
        return iter(self.landmarks)

    def __getitem__(self, key) -> Landmark:
        if isinstance(key, PoseLandmark):
            return self.landmarks[int(key)]
        return self.landmarks[key]

    def get(self, lm: PoseLandmark) -> Landmark:
        return self.landmarks[int(lm)]

    @property
    def pixels(self) -> Tuple[Tuple[int, int], ...]:
        if self._pixels is None:
            w, h = self.image_size
            self._pixels = tuple(
                (int(lm.x * w), int(lm.y * h)) for lm in self.landmarks
            )
        return self._pixels

    def body_normalized(self, min_scale: float = 1e-3) -> Tuple[Landmark, ...]:
        """Return landmarks recentered at the hip midpoint and scaled by torso length.

        Output is translation- and scale-invariant — well-suited for posture
        analysis or downstream ML features. ``min_scale`` guards against
        degenerate torsos (e.g. when the subject is mostly off-frame).
        """
        lh = self.get(PoseLandmark.LEFT_HIP)
        rh = self.get(PoseLandmark.RIGHT_HIP)
        ls = self.get(PoseLandmark.LEFT_SHOULDER)
        rs = self.get(PoseLandmark.RIGHT_SHOULDER)

        cx = (lh.x + rh.x) * 0.5
        cy = (lh.y + rh.y) * 0.5
        cz = (lh.z + rh.z) * 0.5

        sx = (ls.x + rs.x) * 0.5
        sy = (ls.y + rs.y) * 0.5
        torso = ((sx - cx) ** 2 + (sy - cy) ** 2) ** 0.5
        scale = max(torso, min_scale)

        return tuple(
            Landmark(
                x=(lm.x - cx) / scale,
                y=(lm.y - cy) / scale,
                z=(lm.z - cz) / scale,
                visibility=lm.visibility,
            )
            for lm in self.landmarks
        )


class LandmarkExtractor:
    """Converts raw MediaPipe pose landmarks into framework-agnostic ``Pose`` objects.

    Stateless; safe to share across threads. Defined as a class to make the
    pipeline seam explicit (Detector → Extractor → Renderer/Analytics).
    """

    def extract(
        self,
        raw_landmarks: Optional[Iterable],
        image_size: Tuple[int, int],
    ) -> Optional[Pose]:
        if raw_landmarks is None:
            return None
        return Pose(
            landmarks=tuple(
                Landmark(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=float(getattr(lm, "visibility", 1.0)),
                )
                for lm in raw_landmarks
            ),
            image_size=image_size,
        )
