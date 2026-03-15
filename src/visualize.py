import cv2
import numpy as np
from src.temporal import ZoneTemporalState, NORMAL, WARNING, ALERT, CRITICAL
from src.anomaly import AlertEvent


# alert state → BGR color mapping
STATE_COLORS = {
    NORMAL:   (0, 255, 0),    # green
    WARNING:  (0, 255, 255),  # yellow
    ALERT:    (0, 128, 255),  # orange
    CRITICAL: (0, 0, 255),    # red
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2


def draw_zones(
    frame: np.ndarray,
    zones: list[dict],
    zone_results: list[dict],
    temporal_states: dict[str, ZoneTemporalState],
) -> np.ndarray:
    """
    Draw zone polygon outlines and per-zone count labels on the frame.

    Args:
        frame:           BGR frame
        zones:           zone dicts from zones.json (for polygon vertices)
        zone_results:    per-zone count/density from postprocess.py
        temporal_states: alert states from temporal.py

    Returns:
        annotated BGR frame
    """
    output = frame.copy()

    # build a lookup from zone_id → result
    result_map = {r["zone_id"]: r for r in zone_results}

    for zone in zones:
        zone_id = zone["zone_id"]
        polygon = np.array(zone["polygon"], dtype=np.int32)

        t_state = temporal_states.get(zone_id)
        alert_state = t_state.alert_state if t_state else NORMAL
        color = STATE_COLORS.get(alert_state, STATE_COLORS[NORMAL])

        # draw zone polygon outline
        cv2.polylines(output, [polygon], isClosed=True, color=color, thickness=2)

        # semi-transparent fill
        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon], color)
        output = cv2.addWeighted(overlay, 0.08, output, 0.92, 0)

        # label position — centroid of polygon
        cx = int(np.mean(polygon[:, 0]))
        cy = int(np.mean(polygon[:, 1]))

        result = result_map.get(zone_id)
        if result:
            count_text = f"{zone_id}: {result['count']:.1f}"
            occ_text   = f"occ: {result['occupancy_ratio'] * 100:.0f}%"

            # count label
            cv2.putText(output, count_text, (cx - 40, cy - 8),
                        FONT, FONT_SCALE, color, THICKNESS)

            # occupancy label
            cv2.putText(output, occ_text, (cx - 40, cy + 16),
                        FONT, FONT_SCALE, color, 1)

            # alert state badge
            cv2.putText(output, alert_state, (cx - 40, cy + 36),
                        FONT, 0.4, color, 1)

    return output


def draw_hud(
    frame: np.ndarray,
    global_count: float,
    frame_number: int,
    fps: float,
) -> np.ndarray:
    """
    Draw global count and frame info in the top-left HUD.

    Args:
        frame:        BGR frame
        global_count: total crowd count estimate for the frame
        frame_number: current frame index
        fps:          video FPS for timestamp calculation

    Returns:
        annotated BGR frame
    """
    output = frame.copy()

    timestamp_sec = frame_number / fps if fps > 0 else 0
    minutes = int(timestamp_sec // 60)
    seconds = int(timestamp_sec % 60)

    hud_lines = [
        f"Total Count : {global_count:.1f}",
        f"Frame       : {frame_number}",
        f"Time        : {minutes:02d}:{seconds:02d}",
    ]

    # dark background rect for readability
    cv2.rectangle(output, (8, 8), (260, 80), (0, 0, 0), -1)
    cv2.rectangle(output, (8, 8), (260, 80), (255, 255, 255), 1)

    for i, line in enumerate(hud_lines):
        y = 28 + i * 18
        cv2.putText(output, line, (14, y), FONT, 0.45, (255, 255, 255), 1)

    return output


def draw_alerts(
    frame: np.ndarray,
    active_events: list[AlertEvent],
    max_display: int = 4,
) -> np.ndarray:
    """
    Draw recent alert events in the bottom-left corner of the frame.

    Args:
        frame:         BGR frame
        active_events: recent AlertEvents from anomaly.py
        max_display:   maximum number of alerts to show on screen

    Returns:
        annotated BGR frame
    """
    output = frame.copy()
    h = frame.shape[0]

    recent = active_events[-max_display:] if len(active_events) > max_display else active_events

    if not recent:
        return output

    # background rect
    cv2.rectangle(output, (8, h - 20 - len(recent) * 20), (360, h - 8), (0, 0, 0), -1)
    cv2.rectangle(output, (8, h - 20 - len(recent) * 20), (360, h - 8), (255, 255, 255), 1)

    for i, event in enumerate(reversed(recent)):
        color = STATE_COLORS.get(event.alert_state, STATE_COLORS[NORMAL])
        text  = f"[{event.alert_type}] {event.zone_id} | density: {event.density:.4f}"
        y     = h - 12 - i * 20
        cv2.putText(output, text, (14, y), FONT, 0.4, color, 1)

    return output


def create_video_writer(
    output_path: str,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> cv2.VideoWriter:
    """
    Create an OpenCV VideoWriter for saving annotated output.

    Args:
        output_path:  path to output .mp4
        fps:          frames per second
        frame_width:  frame width in pixels
        frame_height: frame height in pixels

    Returns:
        cv2.VideoWriter instance
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
