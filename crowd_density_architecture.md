# Crowd Density Estimation — Architecture
**OpenVINO · Density Map Regression · Temporal Aggregation · Zone Alerting**

---

## Pipeline Overview

```
Video File (WorldExpo'10 / ShanghaiTech / UCF-CC-50)
       |
       v
Frame Preprocessing
  - BGR → RGB
  - Resize to model input dims
  - Normalize (ImageNet mean/std)
  - Transpose HWC → CHW → NCHW (float32)
       |
       v
CSRNet Inference (OpenVINO IR)
  - VGG-16 frontend → dilated conv backend
  - Output: density map at H/8 × W/8 resolution
       |
       v
Density Map Postprocessing
  - Upsample to frame resolution (bilinear)
  - Global pixel sum → total count estimate
  - Apply per-zone polygon masks → per-zone density
       |
       v
Temporal Aggregation (per zone)
  - Circular buffer of last N density estimates
  - Smoothed density (rolling mean)
  - Rate of change (delta between smoothed values)
  - Trend (linear regression slope over buffer)
       |
       v
Anomaly Flagging
  - Sustained high density → overcrowding alert
  - Sudden spike / drop → surge or stampede signal
  - Accumulation trend → proactive warning
       |
       v
Output
  - Annotated frame: heatmap overlay + zone labels + alert state
  - Event log: timestamped alerts per zone (JSON / CSV)
```

---

## Stage 01 — Input

**Video source used:** Kaggle dataset — `ubaydulloasatullaev/crowd-detection-video`. Overhead shot of a moving, mostly packed crowd. Frame resolution: 1920×1080. Camera is fully static with no panning or drift.

The overhead angle eliminates depth distortion — every person occupies roughly the same pixel area regardless of position in the frame, making CSRNet density estimates spatially consistent without geometric correction.

**Other dataset options for validation or benchmarking:**

**WorldExpo'10** — surveillance video sequences with ground truth count annotations per ROI per frame. Best fit for quantitative evaluation because it has fixed overhead cameras and per-frame ground truth.

**ShanghaiTech Part A / Part B** — Part A is ultra-dense (concerts, rallies), Part B is street-level and sparser. Both have dot annotations for density map generation.

**UCF-CC-50** — 50 images, extreme crowd densities. Good for stress-testing the upper bound of the estimator.

Camera constraint: overhead or high-angle fixed camera strongly preferred. A perspective-view camera introduces depth distortion — a person at 10m occupies far fewer pixels than one at 2m, which corrupts density estimates without geometric correction.

---

## Stage 02 — Preprocessing

```
Frame (BGR, HWC)
  → cv2.cvtColor BGR → RGB
  → cv2.resize to model input dims
  → subtract ImageNet mean [0.485, 0.456, 0.406]
  → divide by ImageNet std  [0.229, 0.224, 0.225]
  → transpose HWC → CHW
  → np.expand_dims → NCHW (batch=1)
  → cast float32
  → assign to OpenVINO infer request input tensor
```

---

## Stage 03 — Inference: CSRNet

CSRNet is a density map regression model, not a detector. It does not localize individual people. It outputs a spatial density map where integrating (summing) pixel values gives the crowd count estimate.

**Architecture:**
- Frontend: VGG-16 conv layers 1–10, no fully connected layers
- Backend: dilated convolutional layers (dilation rates 2, 4) — avoids spatial downsampling while maintaining large receptive fields

**Why not detection-based:**
In dense crowds, occlusion causes person detectors to miss 40–60% of individuals. CSRNet is trained with Gaussian-blurred dot annotations (one dot per head), so it learns to model the distribution of heads rather than localize each one. It degrades gracefully at high densities where detectors fail.

**Model conversion path (offline, one-time):**

```
Pretrained PyTorch checkpoint (rootstrap-org/crowd-counting on Hugging Face)
  → ov.convert_model(model, example_input=dummy_input)
  → ov.save_model(ov_model, "assets/models/csrnet.xml")
  → csrnet.xml + csrnet.bin
  → core.compile_model("csrnet.xml", "AUTO")
```

`ov.convert_model()` reads the PyTorch graph directly via `torch.jit.trace`, eliminating the ONNX intermediate step. This avoids operator compatibility issues with CSRNet's dilated convolutions and keeps the entire conversion pipeline in Python with no CLI switching.

**Pretrained weights note:** Weights sourced from `rootstrap-org/crowd-counting` (trained on ShanghaiTech Part B — sparse street scenes). For a dense overhead crowd, Part A weights give better calibrated counts. Part B weights are sufficient for a POC pipeline.

---

## Stage 04 — Postprocessing

**Density map processing:**

```
Raw output: single-channel map at H/8 × W/8
  → upsample to frame resolution (bilinear interpolation)
  → optional Gaussian smoothing (reduces pixelation in overlay)
  → np.sum over full map → global count estimate
  → apply jet colormap → RGB heatmap for overlay
```

Note: sum before upsampling for the count estimate. Summing after upsampling inflates the value by 64× (8×8 interpolation factor).

**Per-zone density:**

```
For each zone:
  → define polygon vertices in pixel space
  → cv2.fillPoly → binary mask (frame resolution)
  → density_map * mask → masked density
  → np.sum(masked_density) → zone count estimate
  → zone count / zone pixel area → normalized density value
```

Zone types to define:
- Entry/exit zones — compare adjacent zone densities over time to infer flow direction
- Bottleneck corridors — narrow paths where dangerous compression builds
- Capacity zones — large areas with defined safe person-count limits

---

## Stage 05 — Temporal Aggregation

This is the core reasoning layer. A single-frame density estimate is noisy — lighting changes, motion blur, and partial occlusion all cause transient spikes. Temporal smoothing separates real crowd changes from sensor artifacts.

Directly analogous to the sliding window and frame-counter aggregation in the fatigue detection pipeline.

**Per zone, maintain:**

```
circular_buffer[zone_id]     # last N density estimates (e.g. N=30 at 10fps = 3 seconds)
smoothed_density[zone_id]    # rolling mean of buffer
rate_of_change[zone_id]      # delta between current and previous smoothed value
trend_slope[zone_id]         # linear regression slope over buffer
alert_state[zone_id]         # NORMAL | WARNING | ALERT | CRITICAL
```

**Second longer buffer for trend detection:**

```
trend_buffer[zone_id]        # last M density estimates (e.g. M=300 at 1/sec = 5 minutes)
trend_r_squared[zone_id]     # goodness of fit — reject weak trends
```

---

## Stage 06 — Anomaly Flagging

Three independent alert types, each with its own state machine per zone.

**Sustained High Density**

Trigger: smoothed density exceeds capacity threshold for K consecutive seconds (e.g. K=5).

Parameters: density threshold (defined per zone based on physical capacity), persistence window K.

This is the most interpretable alert — maps directly to a physical overcrowding scenario.

**Sudden Spike / Drop**

Trigger: rate_of_change exceeds a physically plausible bound — faster than a crowd could organically fill or empty a zone.

- Spike (rapid increase): possible crowd surge, panic convergence, emergency ingress
- Drop (rapid decrease): possible stampede, evacuation, crowd scatter

Disambiguation from camera artifacts:
- Require spike to persist for at least 2 frames before flagging
- Cross-check spatially adjacent zones — a real surge affects correlated zones together
- Monitor global frame brightness as a confound indicator (a light switching on/off causes a density map artifact that is spatially uniform, not zone-specific)

This is the hardest alert type to tune correctly.

**Accumulation Trend**

Trigger: linear regression slope over the trend_buffer is positive and exceeds a configured rate, even if current density is below the high-density threshold. Require R² > 0.6 to reject weak fits.

Value: proactive alerting — flags a zone that is gradually filling before it reaches a dangerous level, giving operators time to intervene.

---

## Stage 07 — Output

**Visual output (per frame):**

```
original frame
  + alpha-blend heatmap overlay (~40% opacity, jet colormap)
  + zone polygon outlines (color-coded: green / yellow / red by alert state)
  + per-zone count label at zone centroid
  + global count in corner HUD
  + alert state text indicator per zone
  → cv2.VideoWriter or cv2.imshow
```

**Event log (per alert event):**

```json
{
  "timestamp_frame": 1240,
  "timestamp_wall": "2026-03-14T10:23:41",
  "zone_id": "zone_B",
  "alert_type": "SUSTAINED",
  "density": 4.72,
  "duration_frames": 53,
  "resolved": false
}
```

Accumulate and export as JSON (streaming append) or CSV at end of run. Use against WorldExpo'10 ground truth for quantitative evaluation.

---

## Design Constraints and Known Challenges

**Camera geometry** — density per pixel is depth-dependent in perspective views. A proper implementation either uses flat overhead cameras (preferred) or applies a geometry correction map derived from camera calibration.

**Ground truth scarcity** — if fine-tuning CSRNet on a custom scene, dot annotations (one dot per head) are required. Tedious to produce at scale.

**Heatmap overlay latency** — keep colormap application vectorized (numpy operations, not pixel loops). At HD resolution this matters for real-time display.

**Zone calibration** — the density threshold per zone needs to be set based on physical zone area and expected safe occupancy. This is a one-time setup step per deployment scene.

---

## Hardware Targeting

```
core.compile_model("csrnet.xml", "AUTO")
```

AUTO device selection lets OpenVINO pick between CPU, iGPU, and NPU based on availability. For better throughput on HD video, target GPU or NPU explicitly:

```
core.compile_model("csrnet.xml", "GPU")
core.compile_model("csrnet.xml", "NPU")
```

---

## Zone Configuration — zones.json

All zone definitions and runtime parameters live in `config/zones.json`.

### Zone Parameters

**`zone_id`** — unique identifier used in alert logs and visualization labels.

**`label`** — human-readable name displayed on the annotated frame.

**`polygon`** — list of `[x, y]` pixel coordinates defining the zone boundary. Defined in frame pixel space. Extract a reference frame to measure coordinates before setting these.

**`capacity`** — maximum safe person count for the zone. Used to compute `occupancy_ratio = zone_count / capacity`. This value must be calibrated to the model's actual output density scale, not real-world person counts. With Part B weights on a dense scene, capacity values in the thousands (10000, 4000, 5000) are appropriate.

### Temporal Parameters

**`short_window`** — sliding buffer size for smoothing and rate-of-change. At 30fps, a value of 30 = 1 second of history. Increase to make the system less reactive to sudden changes.

**`trend_window`** — longer buffer size for trend detection. At 30fps, a value of 300 = 10 seconds. Linear regression is fit over this buffer to detect gradual accumulation.

### Anomaly Parameters

**`sustained_density_threshold`** — density per pixel value above which a zone is considered overcrowded. Must be tuned after observing actual smoothed density values in the terminal output. Too low and the system fires SUSTAINED alerts continuously on a packed scene.

**`sustained_persistence_frames`** — how many consecutive frames density must stay above threshold before firing a SUSTAINED alert. At 30fps, a value of 30 = 1 second.

**`spike_roc_threshold`** — maximum plausible rate-of-change per frame for normal crowd movement. If `rate_of_change > threshold`, a SPIKE is flagged. Tune by observing rate-of-change values during normal playback.

**`drop_roc_threshold`** — same as above for sudden density drops. If `rate_of_change < threshold` (negative value), a DROP is flagged.

**`spike_persist_frames`** — a spike or drop must persist for at least this many frames before alerting. Filters single-frame noise from lighting changes or motion blur.

**`trend_slope_threshold`** — minimum linear regression slope over the trend buffer to flag an accumulation trend. Raise this if too many TREND alerts are firing.

**`trend_r_squared_threshold`** — minimum R² goodness of fit for the trend regression. Values below this mean the density is not changing consistently enough to be a real trend. Raise to make trend detection stricter.

### Example Configuration (1920×1080 frame, Part B weights)

```json
{
  "zones": [
    {
      "zone_id": "zone_A",
      "label": "Main Crowd Area",
      "polygon": [[50, 50], [1270, 50], [1270, 700], [50, 700]],
      "capacity": 10000
    },
    {
      "zone_id": "zone_B",
      "label": "Upper Right",
      "polygon": [[1280, 50], [1870, 50], [1870, 500], [1280, 500]],
      "capacity": 4000
    },
    {
      "zone_id": "zone_C",
      "label": "Lower Strip",
      "polygon": [[50, 720], [1870, 720], [1870, 1030], [50, 1030]],
      "capacity": 5000
    }
  ],
  "temporal": {
    "short_window": 30,
    "trend_window": 300
  },
  "anomaly": {
    "sustained_density_threshold": 0.005,
    "sustained_persistence_frames": 30,
    "spike_roc_threshold": 0.002,
    "drop_roc_threshold": -0.002,
    "spike_persist_frames": 2,
    "trend_slope_threshold": 0.00001,
    "trend_r_squared_threshold": 0.6
  }
}
```
