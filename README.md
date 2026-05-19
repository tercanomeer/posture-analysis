# posture-analysis

Real-time human pose detection from a webcam feed, built on MediaPipe + OpenCV.
Draws a 33-point skeleton overlay and a live FPS counter.

## File structure

```
posture-analysis/
├── main.py              # Entry point — argparse, capture/detect/render loop
├── requirements.txt
├── .gitignore
├── README.md
├── models/              # Auto-populated cache for downloaded .task files (gitignored)
└── src/
    ├── __init__.py
    ├── camera.py        # Webcam capture (context-managed, low-latency defaults)
    ├── detector.py      # PoseDetector — MediaPipe Tasks PoseLandmarker wrapper
    ├── renderer.py      # PoseRenderer — skeleton overlay + HUD
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

Grant your terminal camera access on first run (macOS: *System Settings → Privacy
& Security → Camera*).

## Execution

```bash
python main.py
```

Useful flags:

| Flag                | Default | Description                                          |
| ------------------- | ------- | ---------------------------------------------------- |
| `--source`          | `0`     | Camera index or video file path                      |
| `--width`           | `1280`  | Capture width                                        |
| `--height`          | `720`   | Capture height                                       |
| `--model`           | `full`  | Pose model variant: `lite`, `full`, `heavy`          |
| `--models-dir`      | `models`| Where to cache downloaded `.task` files              |
| `--no-mirror`       | off     | Disable horizontal flip of the webcam feed           |

The first run downloads the chosen `.task` model (~6 MB lite, ~9 MB full, ~30 MB
heavy) into `models/`. Subsequent runs reuse the cached file.

Press **Q** or **Esc** to quit; closing the window also exits cleanly.

## Architecture

The pipeline is split into single-responsibility modules so each piece is
independently testable and replaceable.

```
   ┌──────────┐    BGR frame    ┌────────────┐    PoseResult    ┌────────────┐
   │  Webcam  │ ──────────────► │PoseDetector│ ───────────────► │PoseRenderer│ ──► imshow
   └──────────┘                 └────────────┘                  └────────────┘
                                                                       ▲
                                                                       │ fps
                                                                  ┌─────────┐
                                                                  │FPSCounter│
                                                                  └─────────┘
```

### `src/camera.py` — `Webcam`
Context-managed `cv2.VideoCapture` wrapper. Sets `CAP_PROP_BUFFERSIZE=1` so the
loop always processes the freshest frame instead of draining a backlog under
load. Exposes a `frames()` generator and a single-shot `read()` for flexibility.

### `src/detector.py` — `PoseDetector`, `PoseResult`
Wraps `mediapipe.tasks.vision.PoseLandmarker` in `RunningMode.VIDEO` (synchronous
per-frame inference with internal temporal smoothing). Converts BGR → RGB once
per frame, wraps it in `mp.Image`, and supplies a monotonic millisecond
timestamp (required by VIDEO mode). The `PoseResult` dataclass hides the
MediaPipe-specific shape so downstream consumers see a simple
`landmarks: list | None` plus a `detected` flag.

Tunables: `model_path`, `num_poses`, the three confidence thresholds.

### `src/renderer.py` — `PoseRenderer`
Stateless drawing of skeleton + HUD onto frames *in place* (no allocations per
frame). Edges come from `PoseLandmarksConnections.POSE_LANDMARKS`; each landmark
is visibility-gated against a configurable threshold so noisy off-screen joints
don't get drawn. FPS text is rendered twice — black stroke, then colored fill —
for legibility on bright backgrounds.

### `src/fps.py` — `FPSCounter`
Exponential-moving-average frame-rate estimator over `time.perf_counter()`
deltas. Smoothing factor (`alpha`) trades responsiveness for stability; default
0.1 produces a readable HUD without per-frame jitter.

### `src/models.py` — `ensure_pose_model`
Lazy downloader for the `.task` model files. Uses `certifi`'s CA bundle for SSL
verification (python.org macOS builds ship without trusted roots). Writes to a
`.part` temp file and atomically renames on success so an interrupted download
never leaves a corrupt model on disk.

### `main.py` — composition root
Parses CLI args, ensures the model file, then composes the four collaborators
inside a single `with` block so camera and detector are deterministically
released on exit (including on Ctrl-C). The loop is intentionally
allocation-light: capture → mirror → detect → draw → show → key check.

## Performance notes

- **Single BGR→RGB conversion** per frame; the same array is wrapped in
  `mp.Image` directly.
- **`CAP_PROP_BUFFERSIZE = 1`** keeps end-to-end latency low at the cost of
  occasional dropped frames under back-pressure — the right trade-off for a
  live HUD.
- **VIDEO running mode** beats `IMAGE` mode in cost (it reuses tracker state)
  and beats `LIVE_STREAM` in simplicity (sync return, no callback marshalling).
- **Pre-projected landmark points** are reused for edges and joints inside
  `draw_skeleton` so each landmark is converted to pixel space exactly once.
- **Model choice**: start with `--model lite` on CPU-only machines; `full`
  (default) is a good balance on Apple Silicon / discrete GPUs; `heavy` is for
  offline/quality-first work.
