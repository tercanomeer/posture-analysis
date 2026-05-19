from __future__ import annotations

import argparse
import math
import sys

import cv2

from src.biomechanics import PostureAnalyzer, PostureMetrics
from src.camera import Webcam
from src.detector import PoseDetector
from src.fps import FPSCounter
from src.landmarks import LandmarkExtractor
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


def _fmt_deg(v: float) -> str:
    return "   -- " if math.isnan(v) else f"{v:6.1f}"


def _print_metrics(metrics: PostureMetrics, fps: float) -> None:
    line = (
        f"\rNeck: {_fmt_deg(metrics.neck_angle)}°  "
        f"Shoulder: {_fmt_deg(metrics.shoulder_slope)}°  "
        f"Torso: {_fmt_deg(metrics.torso_inclination)}°  "
        f"FPS: {fps:5.1f}"
    )
    # \r so an interactive terminal sees one self-updating line; flush so
    # consumers reading the pipe (e.g. background tasks) see each tick promptly.
    sys.stdout.write(line)
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    source = _coerce_source(args.source)
    mirror = not args.no_mirror

    model_path = ensure_pose_model(args.model, args.models_dir)

    fps = FPSCounter()
    extractor = LandmarkExtractor()
    analyzer = PostureAnalyzer()
    renderer = PoseRenderer()

    with Webcam(source=source, width=args.width, height=args.height) as cam, \
            PoseDetector(model_path=str(model_path)) as detector:
        for frame in cam.frames():
            if mirror:
                frame = cv2.flip(frame, 1)

            raw = detector.detect(frame)
            pose = extractor.extract(raw.landmarks, raw.image_size)
            metrics = analyzer.analyze(pose)
            current_fps = fps.tick()

            renderer.draw_skeleton(frame, pose)
            renderer.draw_fps(frame, current_fps)
            renderer.draw_metrics(frame, metrics)
            _print_metrics(metrics, current_fps)

            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) in QUIT_KEYS:
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    sys.stdout.write("\n")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
