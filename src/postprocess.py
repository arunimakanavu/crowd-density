import cv2
import numpy as np


def process_density_map(
    density_map: np.ndarray,
    frame_height: int,
    frame_width: int,
    smooth: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Upsample raw density map output to frame resolution and compute global count.

    Args:
        density_map:  raw model output, shape (1, 1, H/8, W/8)
        frame_height: original frame height to upsample to
        frame_width:  original frame width to upsample to
        smooth:       apply Gaussian smoothing to reduce pixelation

    Returns:
        upsampled_map: float32 density map at frame resolution (H, W)
        global_count:  total crowd count estimate for the frame
    """

    # squeeze batch and channel dims → (H/8, W/8)
    raw = density_map.squeeze()

    # global count — sum BEFORE upsampling to avoid 64x inflation
    global_count = float(np.sum(raw))

    # upsample to frame resolution
    upsampled = cv2.resize(raw, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

    # optional Gaussian smoothing
    if smooth:
        upsampled = cv2.GaussianBlur(upsampled, (15, 15), sigmaX=0, sigmaY=0)

    return upsampled, global_count


def apply_zone_masks(
    density_map: np.ndarray,
    zones: list[dict],
) -> list[dict]:
    """
    Apply polygon ROI masks to the upsampled density map for per-zone counts.

    Args:
        density_map: float32 density map at frame resolution (H, W)
        zones:       list of zone dicts from zones.json, each with:
                       - zone_id:   str
                       - polygon:   list of [x, y] vertices
                       - capacity:  int (max safe person count)

    Returns:
        list of dicts with zone_id, density, count, capacity, occupancy_ratio
    """
    h, w = density_map.shape
    results = []

    for zone in zones:
        polygon = np.array(zone["polygon"], dtype=np.int32)

        # binary mask for this zone
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(mask, [polygon], 1.0)

        # masked density
        masked = density_map * mask

        # zone count and normalized density
        zone_count    = float(np.sum(masked))
        zone_area     = float(np.sum(mask))
        zone_density  = zone_count / zone_area if zone_area > 0 else 0.0
        occupancy     = zone_count / zone["capacity"] if zone["capacity"] > 0 else 0.0

        results.append({
            "zone_id":         zone["zone_id"],
            "count":           zone_count,
            "density":         zone_density,
            "capacity":        zone["capacity"],
            "occupancy_ratio": occupancy,
        })

    return results


def build_heatmap(density_map: np.ndarray) -> np.ndarray:
    """
    Convert a float32 density map to a jet colormap BGR image for overlay.

    Args:
        density_map: float32 array (H, W)

    Returns:
        heatmap: uint8 BGR image (H, W, 3)
    """
    # normalize to [0, 255]
    d_min = density_map.min()
    d_max = density_map.max()

    if d_max - d_min > 0:
        normalized = (density_map - d_min) / (d_max - d_min)
    else:
        normalized = np.zeros_like(density_map)

    scaled = (normalized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

    return heatmap


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Alpha-blend heatmap over the original frame.

    Args:
        frame:   original BGR frame (H, W, 3)
        heatmap: jet colormap BGR image (H, W, 3)
        alpha:   heatmap opacity

    Returns:
        blended BGR frame
    """
    return cv2.addWeighted(heatmap, alpha, frame, 1 - alpha, 0)
