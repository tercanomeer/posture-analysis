from __future__ import annotations

import argparse
import sys

import cv2

from src.camera import Webcam
from src.detector import PoseDetector
from src.fps import FPSCounter
from src.models import MODEL_URLS, ensure_pose_model
from src.renderer import PoseRenderer

WINDOW_NAME = "Posture Analysis"
QUIT_KEYS = {ord("q"), ord("Q"), 27}  # 27 == ESC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time posture analysis.")
    parser.add_argument("--source", default=0, help="Camera index or video path (default: 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_URLS),
        default="full",
        help="MediaPipe pose model variant (default: full).",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory used to cache downloaded .task model files.",
    )
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal flip.")
    return parser.parse_args()


def _coerce_source(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def main() -> int:
    args = parse_args()
    source = _coerce_source(args.source)
    mirror = not args.no_mirror

    model_path = ensure_pose_model(args.model, args.models_dir)

    fps = FPSCounter()
    renderer = PoseRenderer()

    with Webcam(source=source, width=args.width, height=args.height) as cam, \
            PoseDetector(model_path=str(model_path)) as detector:
        for frame in cam.frames():
            if mirror:
                frame = cv2.flip(frame, 1)

            result = detector.detect(frame)
            renderer.draw_skeleton(frame, result)
            renderer.draw_fps(frame, fps.tick())

            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) in QUIT_KEYS:
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
