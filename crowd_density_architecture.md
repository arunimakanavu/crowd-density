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

Dataset options, in order of preference for this pipeline:

**WorldExpo'10** — surveillance video sequences with ground truth count annotations per ROI per frame. Best fit because it has fixed overhead cameras and per-frame ground truth you can validate against.

**ShanghaiTech Part A / Part B** — Part A is ultra-dense (concerts, rallies), Part B is street-level and sparser. Both have dot annotations for density map generation.

**UCF-CC-50** — 50 images, extreme crowd densities. Good for stress-testing the upper bound of the estimator.

For quick prototyping without ground truth: Pexels/Pixabay stock footage or yt-dlp downloads are fine for visual heatmap validation.

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
Pretrained PyTorch checkpoint (ShanghaiTech weights)
  → torch.onnx.export(model, dummy_input, "csrnet.onnx", opset_version=11)
  → ovc csrnet.onnx --output_model csrnet
  → csrnet.xml + csrnet.bin
  → core.compile_model("csrnet.xml", "AUTO")
```

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

AUTO device selection lets OpenVINO pick between CPU, iGPU, and NPU based on availability — same pattern used in the fatigue detection deployment. For an edge deployment on OpenEdge Platform hardware, GPU or NPU will give better throughput on HD video than CPU alone.
