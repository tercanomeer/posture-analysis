# posture-analysis

Real-time human pose detection from a webcam feed, built on MediaPipe + OpenCV.
Produces structured, framework-agnostic landmark objects (with optional
body-relative normalization), runs a biomechanics analysis engine on them
(neck angle, shoulder slope, torso inclination), classifies the result against
configurable thresholds (forward head, shoulder asymmetry, slouching), and
renders a skeleton overlay with a live FPS + metrics + verdict HUD.

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
    ├── smoother.py      # LandmarkSmoother — per-coordinate EMA on Pose, jitter reduction
    ├── biomechanics.py  # angle_between primitive + neck/shoulder/torso analyzers
    ├── classifier.py    # Rule-based posture classifier (FHP, asymmetry, slouch)
    ├── renderer.py      # PoseRenderer — skeleton overlay (also has FPS/metrics HUD methods)
    ├── feedback.py      # FeedbackRenderer — full-frame overlay alternative (border + banner + chips)
    ├── dashboard.py     # Dashboard — composed academic-style UI (header + camera + sidebar)
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
| `--smoothing-alpha` | `0.5` | EMA weight on the newest landmark sample. `1.0` = no smoothing. |
| `--visibility-alpha` | `0.7` | EMA weight on landmark visibility scores. |

Press **Q** or **Esc** to quit; closing the window also exits cleanly.

## Architecture

The pipeline is split into single-responsibility stages so each piece is
independently testable, replaceable, and reusable.

```
  Webcam ─► Detector ─► LandmarkExtractor ─► LandmarkSmoother ─► PostureAnalyzer ─► PostureClassifier ─► PoseRenderer.draw_skeleton ─► Dashboard.render ─► imshow
   (BGR)     (Raw)        (Pose)               (Pose, smoothed)    (Metrics)          (Assessment)             (camera frame, in place)        ▲
                                                                                                                                                 │
                                                                                       (metrics, assessment, fps) ──────────────────────────────┘
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

### `src/smoother.py` — `LandmarkSmoother`
Per-coordinate exponential moving average on `Pose` objects. State is a
single `(33, 4)` ndarray (`x, y, z, visibility`) so memory and per-frame
work are O(1) in framerate. Two parameters:

| Param | Default | Effect |
| --- | ---: | --- |
| `alpha` | `0.5` | Weight on the *newest* landmark sample for `x, y, z`. `1.0` = pass-through, `0.5` = balanced, `0.2` = heavy smoothing with perceptible lag. |
| `visibility_alpha` | `0.7` | Weight for visibility scores. Higher than `alpha` so visibility-gated logic (e.g. skeleton drawing, NaN propagation in the analyzer) reacts quickly when a landmark leaves the frame. |

`smooth(None)` clears internal state, so a long occlusion gap won't drag the
next real detection toward stale positions.

**Measured impact** on a synthetic 380-frame noisy stream (0.5% normalized
Gaussian jitter per coordinate):

| Setting | Neck-angle stdev | vs. raw |
| --- | ---: | ---: |
| raw | 1.22° | — |
| `alpha=0.5` (default) | 0.71° | −42% |
| `alpha=0.2` | 0.40° | −67% |

Step-response time at `alpha=0.5` is ~5 frames (~167 ms at 30 FPS) to reach
within 1% of a new target value — fast enough to feel live while killing
most of the per-frame flicker that drove classifier oscillation.

### `src/biomechanics.py` — geometry primitives + posture analyzer
Two layers, both reusable on their own.

**Pure-NumPy primitives** (no project dependencies):

- `angle_between(a, b, c)` — angle ∠ABC in degrees at vertex `b`. Accepts any
  array-like (tuple, list, ndarray) in 2D or 3D. Clamps cosine to `[-1, 1]`
  before `acos` and returns NaN on degenerate segments — never raises.
- `angles_batch(triplets)` — vectorized form for `(N, 3, D)` arrays.
- `signed_angle_from_horizontal(p_from, p_to)` — useful for shoulder slope.
- `signed_angle_from_vertical(p_from, p_to)` — useful for torso lean.

**Posture-specific helpers** that consume a `Pose`:

- `neck_angle(pose)` — ∠(hip-mid, shoulder-mid, ear-mid). 180° = head stacked
  above torso (ideal); smaller values indicate forward-head flexion.
- `shoulder_slope(pose)` — signed angle of the shoulder line from horizontal.
  Positive = right shoulder higher in the image.
- `torso_inclination(pose)` — signed angle of hip-mid → shoulder-mid from
  vertical-up. Positive = leaning right.

**Aggregation**: `PostureAnalyzer.analyze(pose) → PostureMetrics` produces a
frozen dataclass with all three metrics. Any field is `math.nan` whenever the
required landmarks fall below the visibility threshold or `pose is None`, so
the realtime loop never crashes on partially-occluded subjects.

### `src/classifier.py` — rule-based posture classification
Consumes a `PostureMetrics` and emits a `PostureAssessment` with a readable
label. Designed to be lightweight (O(1) per rule, no allocations per call)
and pluggable.

- **`Severity`** — `OK` / `MILD` / `SEVERE` / `UNKNOWN`. `UNKNOWN` is returned
  when the input metric is NaN (occluded landmarks), so callers can distinguish
  "no data" from "good posture".
- **`PostureThresholds`** — frozen dataclass exposing every tunable knob:

  | Field | Default | Meaning |
  | --- | ---: | --- |
  | `forward_head_mild_deg` | `160.0` | neck angle below this → mild forward head |
  | `forward_head_severe_deg` | `145.0` | neck angle below this → severe forward head |
  | `shoulder_mild_deg` | `5.0` | shoulder slope magnitude above this → mild asymmetry |
  | `shoulder_severe_deg` | `10.0` | shoulder slope magnitude above this → severe asymmetry |
  | `slouch_mild_deg` | `10.0` | torso inclination magnitude above this → mild slouch |
  | `slouch_severe_deg` | `20.0` | torso inclination magnitude above this → severe slouch |

- **`PostureRule`** — base class with a single `evaluate(metrics) → Severity`
  method. Three concrete rules ship with the engine: `ForwardHeadRule`,
  `ShoulderAsymmetryRule`, `SlouchingRule`.
- **`PostureClassifier`** — composes rules into an assessment. The default
  constructor wires up the three built-in rules from a `PostureThresholds`
  instance; you can also pass your own rule list to extend or replace them.
- **`PostureAssessment`** — frozen dataclass with `findings` (tuple of
  `(rule_label, severity)`), `.overall` (worst severity across rules), and
  `.label` (human-readable summary like
  `"Forward head (severe); Slouching (mild)"`).

The `ShoulderAsymmetryRule` folds slopes into `[-90°, +90°]` before comparing
magnitudes, so a mirrored-view artifact reporting `±178°` (LEFT/RIGHT
shoulder labels swapped) is correctly treated as a level shoulder line, not a
severe asymmetry.

### `src/renderer.py` — `PoseRenderer`
Stateless drawing of skeleton + numeric HUD onto frames *in place*. Imports
only from `landmarks.py` and `biomechanics.py`. Uses the cached `pose.pixels`
projection and visibility-gates both edges and joints. HUD text is rendered
twice (black stroke, then colored fill) for legibility on bright backgrounds.
`draw_fps` and `draw_metrics` deliberately position below `hud_top` (default
80px) so they sit beneath `FeedbackRenderer`'s status banner.

### `src/feedback.py` — `FeedbackRenderer`
Full-frame overlay alternative used when you want a single, undivided camera
view. Three severity-colored components: a thick border around the frame, a
status banner across the top, and warning chips stacked bottom-right. All
draws are plain `cv2.rectangle` / `cv2.putText` — no alpha blending. Use
this instead of `Dashboard` for fullscreen presentations where the camera
should fill the whole window.

### `src/dashboard.py` — `Dashboard`
The default UI for the live demo: a composed academic-style layout assembled
on a pre-allocated canvas.

```
┌─────────────────────────────────────────────────────────┐
│ Posture Analysis  |  Realtime Demo         FPS  29.7    │ ← Header strip (56 px)
├─────────────────────────────────────────┬───────────────┤
│                                         │ STATUS        │
│                                         │ ┌───────────┐ │
│           camera feed +                 │ │POSTURE:   │ │ ← severity-colored card
│           skeleton overlay              │ │  GOOD     │ │
│                                         │ └───────────┘ │
│                                         │ ANGLES        │
│                                         │  Neck   178.0°│ ← striped rows
│                                         │  Shoul.   1.0°│
│                                         │  Torso    2.0°│
│                                         │ WARNINGS      │
│                                         │  • Forward …  │
└─────────────────────────────────────────┴───────────────┘
```

Layout: a 56-px header strip with title + FPS, the raw camera frame copied
into the left area (with skeleton already drawn on it by `PoseRenderer`),
and a 340-px sidebar on the right with three sectioned panels — `STATUS`,
`ANGLES`, `WARNINGS`. Each section title is underlined with the gold accent.

Performance: the canvas is allocated once and reused; per-frame work is
one `np.ndarray` solid fill + one memcpy of the camera area + a handful of
`cv2.rectangle` / `cv2.putText` / `cv2.line` / `cv2.circle` calls.
Benchmarked at **3.3 ms / frame** on a 1280×720 stream (~300 fps headroom).

Both `sidebar_width`, `header_height`, and `title` are constructor args so a
thesis demo can swap in the speaker's preferred wording.

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

## Using the smoother in your own pipeline

```python
from src.smoother import LandmarkSmoother

smoother = LandmarkSmoother(alpha=0.5, visibility_alpha=0.7)

# Per frame:
pose = extractor.extract(raw.landmarks, raw.image_size)
pose = smoother.smooth(pose)            # → smoothed Pose, or None if input was None
metrics = analyzer.analyze(pose)        # downstream consumers don't change
```

The smoother is stateful (one instance per video stream). Call
`smoother.reset()` if you want to discard history without passing a `None`
detection — e.g. when switching cameras.

To run the live demo without smoothing for comparison:

```bash
python main.py --smoothing-alpha 1.0
```

## Using the biomechanics engine

The `angle_between` primitive is standalone — it works on any 2D/3D points
and has no project dependencies beyond NumPy:

```python
from src.biomechanics import angle_between, angles_batch

# Right angle between three 2D points (B is the vertex).
angle_between((1, 0), (0, 0), (0, 1))   # → 90.0

# Vectorized: shape (N, 3, D) → (N,) angles in degrees.
import numpy as np
triplets = np.array([
    [[1, 0], [0, 0], [0, 1]],
    [[1, 0, 0], [0, 0, 0], [-1, 0, 0]],
])
angles_batch(triplets)                  # → [90., 180.]
```

Pose-aware helpers and the aggregate analyzer:

```python
from src.biomechanics import (
    PostureAnalyzer,
    neck_angle, shoulder_slope, torso_inclination,
)

# Compute one metric ad hoc:
neck_angle(pose)                        # → e.g. 168.4 (or NaN if occluded)

# Or get all three at once:
analyzer = PostureAnalyzer(visibility_threshold=0.5)
metrics = analyzer.analyze(pose)
print(metrics.neck_angle, metrics.shoulder_slope, metrics.torso_inclination)
```

All helpers return `math.nan` when required landmarks fall below the
visibility threshold or when `pose is None`, so they're safe to call every
frame without try/except.

## Using the posture classifier

```python
from src.classifier import PostureClassifier, PostureThresholds, Severity

# Default thresholds:
classifier = PostureClassifier()
assessment = classifier.classify(metrics)
print(assessment.overall.value, "|", assessment.label)
# → "ok      | Good posture"
# → "severe  | Forward head (severe); Slouching (mild)"
# → "unknown | No detection"

# Custom thresholds — pass any subset; the rest stay at defaults:
strict = PostureClassifier(PostureThresholds(
    forward_head_mild_deg=170,   # tighter neck tolerance
    forward_head_severe_deg=160,
    shoulder_mild_deg=2,
    shoulder_severe_deg=5,
))
strict.classify(metrics)

# Custom rule set — replace or extend the defaults:
from src.classifier import PostureRule, ForwardHeadRule, ShoulderAsymmetryRule

class HeadTiltRule(PostureRule):
    name = "head_tilt"; label = "Head tilt"
    def evaluate(self, metrics):
        return Severity.OK  # your logic here

custom = PostureClassifier(rules=[
    ForwardHeadRule(mild_deg=160, severe_deg=145),
    HeadTiltRule(),
])
```

### Example outputs

| Metrics (neck / shoulder / torso, all degrees) | Overall | Label |
| --- | :---: | --- |
| 178 / 1 / 2 | OK | `Good posture` |
| 155 / 3 / 4 | MILD | `Forward head (mild)` |
| 140 / 12 / 25 | SEVERE | `Forward head (severe); Shoulder asymmetry (severe); Slouching (severe)` |
| NaN / NaN / NaN | UNKNOWN | `No detection` |

By default the live demo uses `Dashboard` (composed sidebar UI). For a
fullscreen-camera presentation, swap in `FeedbackRenderer` instead — its
border + banner + chips sit directly on the frame.

## Integration: building a dashboard-aware loop

```python
from src.dashboard import Dashboard

dashboard = Dashboard(title="My Thesis Demo")  # sidebar/header sizes also configurable

# Per frame:
renderer.draw_skeleton(frame, pose)            # mutates the camera frame in place
canvas = dashboard.render(frame, metrics, assessment, current_fps)
cv2.imshow("Posture Analysis", canvas)
```

The draw order is fixed:

1. **Skeleton first** — drawn into the raw camera frame so it appears
   inside the dashboard's camera area.
2. **Dashboard second** — composites the frame plus all panels onto its
   pre-allocated canvas in a single `render()` call.

### Alternative: fullscreen overlay

If you'd rather present the camera at full window size, replace `Dashboard`
with `FeedbackRenderer`:

```python
from src.feedback import FeedbackRenderer
feedback = FeedbackRenderer()
# ...
renderer.draw_skeleton(frame, pose)
feedback.draw(frame, assessment)
renderer.draw_fps(frame, current_fps)
renderer.draw_metrics(frame, metrics)
cv2.imshow("Posture Analysis", frame)
```

## Optimization notes

### Profile-driven analysis

Measured per-stage latency on a 1280×720 stream (200 iters, MediaPipe `full`
model, M4 CPU). Stages that early-exit on `pose is None` are profiled
separately with synthetic landmarks injected.

```
stage              p50      p95     mean    share
─────────────────────────────────────────────────
detector          7.04    7.33    7.04    pipeline-bound  ← MediaPipe inference
extractor         0.012   0.012   0.012   trivial
smoother          0.026   0.026   0.025   trivial
analyzer          0.006   0.006   0.006   trivial
classifier        0.001   0.001   0.001   trivial
skeleton draw     0.085   0.089   0.085   trivial
dashboard         0.197   0.206   0.197   ← after fix; was 3.336
─────────────────────────────────────────────────
total                                      ~7.4 ms  →  ~135 fps headroom
```

### Applied fix: dashboard chrome cache

The first profile showed the dashboard at 3.34 ms — 96% of all
post-detection cost. Drilling in, the *entire* time came from one line:
`canvas[:] = Palette.bg`, a numpy broadcast that touches every pixel of the
1620×776×3 canvas. The actual `cv2` draw calls were only 0.13 ms combined.

The dashboard now pre-renders all static chrome (background fill, header
bg/title/accent, sidebar section titles + accent underlines, ANGLES row
stripes + left-side labels) into a template at first use. Per-frame
`Dashboard.render`:

1. `np.copyto(canvas, chrome)` — one fast memcpy.
2. Camera frame copy into the camera region.
3. Camera-area divider line.
4. Dynamic only: FPS value, STATUS card, ANGLES values, WARNINGS list.

Result: **dashboard 3.34 → 0.20 ms (−94%)**, total downstream **3.47 → 0.33 ms (−91%)**.

### What was NOT worth optimizing

After the chrome cache the pipeline is dominated by MediaPipe (7 ms);
everything else combined is ~0.33 ms. Specifically:

- **Skeleton draw (0.086 ms)** — 35 lines + 33 circles via individual cv2
  calls. Could batch with `cv2.polylines` but the pose topology is a graph,
  not chains. Save would be sub-100 µs at best. Skip.
- **Smoother Python loop (0.026 ms)** — reading 33 landmarks into the EMA
  state array. Vectorizable, but a 0.026 ms ceiling on the win isn't worth
  the API churn. Skip.
- **Extractor (0.012 ms)** — same reasoning.

### Profiling suggestions for future changes

1. **Profile first, optimize second.** The chrome-cache win was a 17×
   speedup on the dashboard — and we'd never have guessed `canvas[:] = bg`
   was a 3 ms line without measurement. The harnesses used here are in this
   PR's commit; they take ~5 seconds to run on a fresh checkout.
2. **`time.perf_counter()`** with `n ≥ 200` iterations and a warmup phase
   gives stable percentiles; mean alone hides p99 tails.
3. For MediaPipe inference specifically, try `--model lite` if CPU bound;
   it costs ~3 ms vs ~7 ms for `full` with modest accuracy loss.

### Architectural improvements considered

- **Threaded capture-detect split**: run `cv2.VideoCapture.read()` on a
  background thread feeding a single-slot queue, so the inference loop never
  waits on the next frame. At our current 30 FPS / 10 ms pipeline this saves
  little, but on a smaller-budget device (RPi, mobile) it can recover
  effective FPS. Not implemented because the current bottleneck is solidly
  MediaPipe, not capture latency.
- **In-place `Pose` mutation**: smoother currently builds 33 new `Landmark`
  dataclasses every frame. Switching `Pose` to carry a `(33, 4)` numpy array
  internally — with `Landmark` as a thin view — would eliminate that. But
  the allocations cost 0.026 ms, so this is API-churn-for-no-gain unless
  the language runtime regresses.
- **Resize before inference**: downsampling the camera frame to 640×360
  before handing it to MediaPipe can roughly halve inference time on
  CPU-only hardware. Quality impact varies with distance to camera. Add as
  a `--inference-size` flag if needed.

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
