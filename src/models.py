from __future__ import annotations

import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def _build_https_opener() -> urllib.request.OpenerDirector:
    # python.org's macOS builds ship without trusted roots; certifi is bundled
    # via mediapipe so we use it explicitly to keep behavior deterministic.
    ctx = ssl.create_default_context(cafile=certifi.where())
    handler = urllib.request.HTTPSHandler(context=ctx)
    return urllib.request.build_opener(handler)


def ensure_pose_model(variant: str = "full", dest_dir: Path | str = "models") -> Path:
    """Return the local path to a pose_landmarker .task file, downloading if absent."""
    if variant not in MODEL_URLS:
        raise ValueError(f"Unknown model variant {variant!r}. Choose from {list(MODEL_URLS)}.")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"pose_landmarker_{variant}.task"
    if target.exists() and target.stat().st_size > 0:
        return target

    url = MODEL_URLS[variant]
    print(f"Downloading pose model '{variant}' from {url}", file=sys.stderr)
    tmp = target.with_suffix(target.suffix + ".part")
    opener = _build_https_opener()
    try:
        with opener.open(url) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return target
