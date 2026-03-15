import cv2
import json
import dataclasses
from datetime import datetime
from pathlib import Path

from src.preprocess  import preprocess_frame, get_video_properties
from src.inference   import CrowdDensityInference
from src.postprocess import process_density_map, apply_zone_masks, build_heatmap, overlay_heatmap
from src.temporal    import TemporalAggregator
from src.anomaly     import AnomalyDetector
from src.visualize   import draw_zones, draw_hud, draw_alerts, create_video_writer


CONFIG_PATH  = "config/zones.json"
VIDEO_PATH   = "assets/sample-video.mp4"
MODEL_PATH   = "assets/models/csrnet.xml"
OUTPUT_VIDEO = "output/annotated_output.mp4"
OUTPUT_LOG   = "output/event_log.json"
DEVICE       = "AUTO"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    # --- load config ---
    config = load_config(CONFIG_PATH)
    zones  = config["zones"]
    t_cfg  = config["temporal"]
    a_cfg  = config["anomaly"]

    # --- init modules ---
    model    = CrowdDensityInference(MODEL_PATH, DEVICE)
    temporal = TemporalAggregator(
                   zones,
                   short_window=t_cfg["short_window"],
                   trend_window=t_cfg["trend_window"],
               )
    detector = AnomalyDetector(
                   zones,
                   sustained_density_threshold=a_cfg["sustained_density_threshold"],
                   sustained_persistence_frames=a_cfg["sustained_persistence_frames"],
                   spike_roc_threshold=a_cfg["spike_roc_threshold"],
                   drop_roc_threshold=a_cfg["drop_roc_threshold"],
                   spike_persist_frames=a_cfg["spike_persist_frames"],
                   trend_slope_threshold=a_cfg["trend_slope_threshold"],
                   trend_r_squared_threshold=a_cfg["trend_r_squared_threshold"],
               )

    # --- open video ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    props = get_video_properties(cap)
    print(f"Video: {props['width']}x{props['height']} @ {props['fps']} fps | {props['frame_count']} frames")

    # --- output setup ---
    Path("output").mkdir(exist_ok=True)
    writer = create_video_writer(
        OUTPUT_VIDEO,
        props["fps"],
        props["width"],
        props["height"],
    )

    all_events   = []
    frame_number = 0

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- preprocess ---
        tensor = preprocess_frame(frame, props["height"], props["width"])

        # --- inference ---
        density_map_raw = model.infer(tensor)

        # --- postprocess ---
        density_map, global_count = process_density_map(
            density_map_raw,
            props["height"],
            props["width"],
        )
        zone_results = apply_zone_masks(density_map, zones)

        # --- temporal update ---
        temporal.update(zone_results)
        temporal_states = temporal.get_all_states()
        
        for zone_id, state in temporal_states.items():
            print(f"{zone_id} | smoothed: {state.smoothed_density:.6f} | roc: {state.rate_of_change:.6f}")

        # --- anomaly detection ---
        events = detector.evaluate(temporal_states, frame_number)
        if events:
            all_events.extend(events)
            for e in events:
                print(f"  [ALERT] frame {frame_number} | {e.zone_id} | {e.alert_type} | {e.alert_state} | density {e.density:.4f}")

        # --- visualize ---
        heatmap   = build_heatmap(density_map)
        annotated = overlay_heatmap(frame, heatmap)
        annotated = draw_zones(annotated, zones, zone_results, temporal_states)
        annotated = draw_hud(annotated, global_count, frame_number, props["fps"])
        annotated = draw_alerts(annotated, all_events)

        writer.write(annotated)

        # --- live visualization ---
        cv2.imshow("Crowd Density Estimation", annotated)
        wait_ms = max(1, int(1000 / props["fps"]))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break

        if frame_number % 100 == 0:
            print(f"  processed frame {frame_number} / {props['frame_count']} | count: {global_count:.1f}")

        frame_number += 1

    # --- cleanup ---
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to {OUTPUT_VIDEO}")

    # --- save event log ---
    log = [dataclasses.asdict(e) for e in all_events]
    with open(OUTPUT_LOG, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"Event log saved to {OUTPUT_LOG} | {len(log)} events recorded")


if __name__ == "__main__":
    main()
