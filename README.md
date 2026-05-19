# posture-analysis

Real-time human pose detection from a webcam feed, built on MediaPipe + OpenCV.
Produces structured, framework-agnostic landmark objects (with optional
body-relative normalization) and renders a skeleton overlay with a live FPS
counter.

## File structure

```
posture-analysis/
├── main.py              # Entry point — argparse, capture/detect/extract/render loop
├── requirements.txt
├── README.md
├── .gitignore
├── models/              # Auto-populated cache for downloaded .task files (gitignored)
└── src/
    ├── __init__.py
    ├── camera.py        # Webcam capture (context-managed, low-latency defaults)
    ├── detector.py      # PoseDetector — MediaPipe Tasks PoseLandmarker wrapper
    ├── landmarks.py     # PoseLandmark enum, Landmark, Pose, LandmarkExtractor, POSE_CONNECTIONS
    ├── renderer.py      # PoseRenderer — skeleton overlay + HUD (MediaPipe-free)
    ├── fps.py           # FPSCounter — EMA-smoothed frame rate
    └── models.py        # ensure_pose_model — auto-downloads .task files
```

## Installation

Requires Python 3.10+ (tested on 3.14, Apple Silicon macOS).

```bash
git clone https://github.com/tercanomeer/posture-analysis.git
cd posture-analysis

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Grant your terminal camera access on first run (macOS: *System Settings →
Privacy & Security → Camera*).

## Execution

```bash
python main.py
```

Useful flags:

| Flag           | Default  | Description                                  |
| -------------- | -------- | -------------------------------------------- |
| `--source`     | `0`      | Camera index or video file path              |
| `--width`      | `1280`   | Capture width                                |
| `--height`     | `720`    | Capture height                               |
| `--model`      | `full`   | `lite` / `full` / `heavy`                    |
| `--models-dir` | `models` | Where to cache downloaded `.task` files      |
| `--no-mirror`  | off      | Disable horizontal flip of the webcam feed   |

Press **Q** or **Esc** to quit; closing the window also exits cleanly.

## Architecture

The pipeline is split into single-responsibility stages so each piece is
independently testable, replaceable, and reusable.

```
   ┌────────┐  BGR frame   ┌──────────┐  RawPoseResult   ┌──────────────────┐  Pose   ┌────────────┐
   │ Webcam │ ───────────► │ Detector │ ───────────────► │ LandmarkExtractor│ ──────► │ Renderer   │ ──► imshow
   └────────┘              └──────────┘                  └──────────────────┘         └────────────┘
                                                                  │
                                                                  ▼
                                                          downstream analytics
                                                          (e.g. body_normalized())
```

The key architectural rule: **MediaPipe lives behind the detector/extractor
boundary**. The renderer and any analytics modules consume `Pose` objects
defined in `src/landmarks.py` and never import `mediapipe` directly. Swapping
the detector for a different pose backend would require no changes elsewhere.

### `src/camera.py` — `Webcam`
Context-managed `cv2.VideoCapture` wrapper. Sets `CAP_PROP_BUFFERSIZE=1` so the
loop always processes the freshest frame instead of draining a backlog under
load.

### `src/detector.py` — `PoseDetector`, `RawPoseResult`
Wraps `mediapipe.tasks.vision.PoseLandmarker` in `RunningMode.VIDEO`
(synchronous per-frame inference with internal temporal smoothing). Returns
`RawPoseResult`, which carries the raw MediaPipe landmark list plus the source
image size. This type exists only at the MediaPipe boundary — downstream
modules consume the structured `Pose` produced by the extractor.

### `src/landmarks.py` — landmark processing
The reusable structured-output layer.

- **`PoseLandmark`** — `IntEnum` of all 33 named MediaPipe pose joints
  (`NOSE`, `LEFT_SHOULDER`, `RIGHT_HIP`, …) for self-documenting access.
- **`Landmark`** — `@dataclass(frozen=True, slots=True)` value type carrying
  `x`, `y`, `z` (image-normalized) and `visibility`. `slots=True` keeps
  per-frame allocations tight.
- **`Pose`** — ordered collection of 33 `Landmark`s plus the originating image
  size. Supports `len()`, iteration, integer indexing, *and* `pose[PoseLandmark.LEFT_HIP]`
  /  `pose.get(PoseLandmark.LEFT_HIP)`. Pixel projections are computed lazily
  on first access and cached for O(1) reuse within a frame.
- **`Pose.body_normalized()`** — returns a tuple of `Landmark`s recentered at
  the hip midpoint and scaled by torso length. Output is translation- and
  scale-invariant, ideal for posture metrics, classifiers, or temporal
  comparisons. A `min_scale` floor guards against degenerate torsos when the
  subject is partially out of frame.
- **`LandmarkExtractor`** — stateless converter from `RawPoseResult.landmarks`
  → `Pose`. Returns `None` when no pose was detected.
- **`POSE_CONNECTIONS`** — the 35-edge pose topology, extracted once from
  MediaPipe at module load and re-exported as a plain tuple so consumers
  (renderer, custom overlays) don't need to import MediaPipe.

### `src/renderer.py` — `PoseRenderer`
Stateless drawing of skeleton + HUD onto frames *in place*. Imports only from
`landmarks.py`. Uses the cached `pose.pixels` projection and visibility-gates
both edges and joints. FPS text is rendered twice (black stroke, then colored
fill) for legibility on bright backgrounds.

### `src/fps.py` — `FPSCounter`
Exponential-moving-average frame-rate estimator over `time.perf_counter()`
deltas. The `alpha` weight trades responsiveness for stability; default `0.1`
produces a readable HUD without per-frame jitter.

### `src/models.py` — `ensure_pose_model`
Lazy downloader for the pose `.task` files. Routes downloads through
`certifi`'s CA bundle because python.org's macOS builds ship without a trust
store. Writes to a `.part` temp file and atomically renames on success so an
interrupted download never leaves a corrupt model on disk.

### `main.py` — composition root
Parses CLI args, ensures the model file, then composes the four collaborators
inside a single `with` block so camera and detector are deterministically
released on exit (including on Ctrl-C). The per-frame loop is intentionally
allocation-light: capture → mirror → detect → extract → draw → show → key check.

## Using the landmark system in your own code

```python
from src.detector import PoseDetector
from src.landmarks import LandmarkExtractor, PoseLandmark
from src.models import ensure_pose_model

extractor = LandmarkExtractor()
with PoseDetector(model_path=str(ensure_pose_model("full"))) as det:
    raw = det.detect(bgr_frame)
    pose = extractor.extract(raw.landmarks, raw.image_size)
    if pose is not None:
        # Named access for self-documenting analytics:
        left_shoulder = pose[PoseLandmark.LEFT_SHOULDER]
        right_hip = pose[PoseLandmark.RIGHT_HIP]

        # Translation/scale-invariant features for ML or rules-based posture analysis:
        features = pose.body_normalized()
```

## Performance notes

- **Single BGR→RGB conversion** per frame; the array is wrapped in `mp.Image`
  with no extra copy.
- **`CAP_PROP_BUFFERSIZE = 1`** trades occasional dropped frames for low
  end-to-end latency — the right trade-off for a live HUD.
- **VIDEO running mode** beats `IMAGE` mode in cost (it reuses tracker state)
  and beats `LIVE_STREAM` in simplicity (sync return, no callback marshalling).
- **Lazy pixel cache** in `Pose.pixels` means each landmark is projected to
  pixel space exactly once per frame and reused across edge + joint drawing.
- **`slots=True`** on the `Landmark` dataclass keeps the per-frame 33-landmark
  allocation compact (no per-instance `__dict__`).
- **Model choice**: `--model lite` for low-end CPUs, `full` (default) for
  Apple Silicon / discrete GPUs, `heavy` for offline/quality-first work.
