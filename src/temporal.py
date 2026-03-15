import numpy as np
from collections import deque


# alert state levels
NORMAL   = "NORMAL"
WARNING  = "WARNING"
ALERT    = "ALERT"
CRITICAL = "CRITICAL"


class ZoneTemporalState:
    """
    Maintains temporal density history and derived signals for a single zone.
    """

    def __init__(
        self,
        zone_id: str,
        short_window: int = 30,   # frames — for smoothing and rate-of-change
        trend_window: int = 300,  # frames — for accumulation trend detection
    ):
        self.zone_id      = zone_id
        self.alert_state  = NORMAL

        # short buffer — smoothing and rate-of-change
        self._short_buffer = deque(maxlen=short_window)

        # long buffer — trend detection
        self._trend_buffer = deque(maxlen=trend_window)

        # derived signals (updated on each push)
        self.smoothed_density  = 0.0
        self.rate_of_change    = 0.0
        self.trend_slope       = 0.0
        self.trend_r_squared   = 0.0

        # previous smoothed value for rate-of-change
        self._prev_smoothed    = 0.0

    def push(self, density: float) -> None:
        """
        Add a new density estimate and recompute all derived signals.

        Args:
            density: per-zone density value from postprocess.py
        """
        self._short_buffer.append(density)
        self._trend_buffer.append(density)

        self._update_smoothed()
        self._update_rate_of_change()
        self._update_trend()

    def _update_smoothed(self) -> None:
        self._prev_smoothed   = self.smoothed_density
        self.smoothed_density = float(np.mean(self._short_buffer))

    def _update_rate_of_change(self) -> None:
        self.rate_of_change = self.smoothed_density - self._prev_smoothed

    def _update_trend(self) -> None:
        """
        Fit a linear regression over the trend buffer.
        Computes slope and R² to assess trend strength.
        """
        n = len(self._trend_buffer)

        if n < 10:
            # not enough data yet
            self.trend_slope     = 0.0
            self.trend_r_squared = 0.0
            return

        y = np.array(self._trend_buffer, dtype=np.float32)
        x = np.arange(n, dtype=np.float32)

        # linear regression coefficients
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)

        self.trend_slope = float(ss_xy / ss_xx) if ss_xx > 0 else 0.0

        # R² — goodness of fit
        y_pred  = self.trend_slope * (x - x_mean) + y_mean
        ss_res  = np.sum((y - y_pred) ** 2)
        ss_tot  = np.sum((y - y_mean) ** 2)

        self.trend_r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def is_buffer_ready(self) -> bool:
        """
        Returns True once the short buffer has enough frames to be meaningful.
        """
        return len(self._short_buffer) == self._short_buffer.maxlen


class TemporalAggregator:
    """
    Manages ZoneTemporalState instances for all zones.
    """

    def __init__(self, zones: list[dict], short_window: int = 30, trend_window: int = 300):
        """
        Args:
            zones:        list of zone dicts from zones.json
            short_window: sliding window size for smoothing (frames)
            trend_window: sliding window size for trend detection (frames)
        """
        self.states = {
            zone["zone_id"]: ZoneTemporalState(
                zone_id=zone["zone_id"],
                short_window=short_window,
                trend_window=trend_window,
            )
            for zone in zones
        }

    def update(self, zone_results: list[dict]) -> None:
        """
        Push new per-zone density estimates into their respective state objects.

        Args:
            zone_results: output from postprocess.apply_zone_masks()
        """
        for result in zone_results:
            zone_id = result["zone_id"]
            if zone_id in self.states:
                self.states[zone_id].push(result["density"])

    def get_state(self, zone_id: str) -> ZoneTemporalState:
        return self.states[zone_id]

    def get_all_states(self) -> dict[str, ZoneTemporalState]:
        return self.states
