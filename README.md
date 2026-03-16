# Crowd Density Estimation with OpenVINO

Real-time crowd density estimation using CSRNet and OpenVINO. Processes an overhead video, generates per-frame density maps, applies spatial zone analysis, performs temporal aggregation, and flags anomalies.

---

## How It Works

The pipeline runs seven stages per frame:

```
Video Input → Preprocessing → CSRNet Inference (OpenVINO IR)
    → Density Map Postprocessing → Temporal Aggregation
    → Anomaly Detection → Annotated Output
```

CSRNet is a density map regression model — it does not detect individuals. It outputs a spatial heatmap where integrating pixel values gives the crowd count estimate. This handles dense, occluded crowds where person detectors fail.

---

## Project Structure

```
crowd-density/
├── assets/
│   ├── sample-video.mp4          # input video
│   └── models/
│       ├── csrnet.xml            # OpenVINO IR model (generated)
│       └── csrnet.bin            # IR weights (generated)
├── config/
│   └── zones.json                # zone polygons, thresholds
├── src/
│   ├── preprocess.py             # frame preprocessing
│   ├── inference.py              # OpenVINO inference wrapper
│   ├── postprocess.py            # density map processing, zone masking
│   ├── temporal.py               # sliding window aggregation
│   ├── anomaly.py                # alert state machines
│   └── visualize.py              # heatmap overlay, HUD, zone labels
├── output/
│   ├── annotated_output.mp4      # rendered output (generated)
│   └── event_log.json            # alert events (generated)
├── convert_model.py              # one-time model conversion script
├── requirements.txt
└── main.py
```

---

## Setup

**1. Clone and create environment**

```bash
git clone https://github.com/arunimakanavu/crowd-density.git
cd crowd-density
python -m venv crowd_env
source crowd_env/bin/activate
pip install -r requirements.txt
```

**2. Add your video**

Place your overhead crowd video at:

```
assets/sample-video.mp4
```

**3. Download pretrained weights**

```python
from huggingface_hub import hf_hub_download
import shutil

path = hf_hub_download(repo_id='rootstrap-org/crowd-counting', filename='weights.pth')
shutil.copy(path, 'assets/models/pretrained.pth')
```

**4. Convert model to OpenVINO IR**

```bash
python convert_model.py --checkpoint assets/models/pretrained.pth
```

This generates `assets/models/csrnet.xml` and `assets/models/csrnet.bin`. PyTorch is not needed after this step.

**5. Configure zones**

Edit `config/zones.json` to define polygon zones that match your video's frame layout. The default config divides a 1920×1080 frame into 6 equal zones (3 columns × 2 rows).

To extract a reference frame:

```bash
python -c "
import cv2
cap = cv2.VideoCapture('assets/sample-video.mp4')
ret, frame = cap.read()
if ret:
    cv2.imwrite('output/first_frame.png', frame)
    print(frame.shape)
cap.release()
"
```

**6. Run**

```bash
python main.py
```

A live window opens showing the annotated output. Press `q` to quit. The annotated video and event log are saved to `output/`.

---

## Configuration

All runtime parameters are in `config/zones.json`:

```json
{
  "zones": [
    { "zone_id": "zone_1", "label": "Top Left",      "polygon": [[0,0],[640,0],[640,540],[0,540]],            "capacity": 6000 },
    { "zone_id": "zone_2", "label": "Top Center",    "polygon": [[640,0],[1280,0],[1280,540],[640,540]],      "capacity": 6000 },
    { "zone_id": "zone_3", "label": "Top Right",     "polygon": [[1280,0],[1920,0],[1920,540],[1280,540]],    "capacity": 6000 },
    { "zone_id": "zone_4", "label": "Bottom Left",   "polygon": [[0,540],[640,540],[640,1080],[0,1080]],      "capacity": 6000 },
    { "zone_id": "zone_5", "label": "Bottom Center", "polygon": [[640,540],[1280,540],[1280,1080],[640,1080]],"capacity": 6000 },
    { "zone_id": "zone_6", "label": "Bottom Right",  "polygon": [[1280,540],[1920,540],[1920,1080],[1280,1080]],"capacity": 6000 }
  ],
  "temporal": {
    "short_window": 30,
    "trend_window": 300
  },
  "anomaly": {
    "sustained_density_threshold": 0.01,
    "sustained_persistence_frames": 30,
    "spike_roc_threshold": 0.0003,
    "drop_roc_threshold": -0.0003,
    "spike_persist_frames": 2,
    "trend_slope_threshold": 0.00001,
    "trend_r_squared_threshold": 0.6
  }
}
```

**Tuning note:** Thresholds were calibrated against observed smoothed density values (0.010–0.018 range for this scene). If your video produces different density ranges, adjust `sustained_density_threshold` and the `roc` thresholds accordingly by printing the smoothed density values during a first run.

---

## Alert Types

| Alert | Trigger |
|-------|---------|
| `SUSTAINED` | Zone density exceeds threshold for N consecutive seconds |
| `SPIKE` | Density increases faster than physically plausible crowd movement |
| `DROP` | Density decreases faster than physically plausible crowd movement |
| `TREND` | Gradual accumulation trend detected over a longer window |

---

## Requirements

```
openvino>=2023.1.0
torch
torchvision
opencv-python
numpy
scipy
huggingface_hub
```

---

## Notes

- **Camera angle:** Overhead or high-angle fixed cameras work best. Perspective-view cameras introduce depth distortion that inflates density estimates for distant regions.
- **Model weights:** Default weights are from `rootstrap-org/crowd-counting` trained on ShanghaiTech Part B (sparse scenes). For dense crowds, Part A weights give better calibrated absolute counts.
- **Device targeting:** Change `DEVICE = "AUTO"` in `main.py` to `"CPU"`, `"GPU"`, or `"NPU"` to target specific Intel hardware.
- **Count scale:** With Part B weights on a dense scene, absolute counts will be inflated. The relative density distribution across zones is spatially correct and the temporal signals behave as expected.

---

## Author

Arunima Surendran
