from dataclasses import dataclass, field
from datetime import datetime
from src.temporal import ZoneTemporalState, NORMAL, WARNING, ALERT, CRITICAL


# alert type constants
SUSTAINED   = "SUSTAINED"
SPIKE       = "SPIKE"
DROP        = "DROP"
TREND       = "TREND"


@dataclass
class AlertEvent:
    """
    A single alert event emitted by the anomaly detector.
    """
    timestamp_frame: int
    timestamp_wall:  str
    zone_id:         str
    alert_type:      str
    alert_state:     str
    density:         float
    duration_frames: int
    resolved:        bool = False


@dataclass
class ZoneAnomalyState:
    """
    Tracks active alert conditions for a single zone.
    """
    zone_id:                  str
    sustained_frame_count:    int   = 0
    spike_frame_count:        int   = 0
    drop_frame_count:         int   = 0
    active_alerts:            list  = field(default_factory=list)


class AnomalyDetector:
    """
    Evaluates temporal signals per zone and emits AlertEvents.
    """

    def __init__(
        self,
        zones: list[dict],

        # sustained high density
        sustained_density_threshold: float = 0.005,  # density per pixel — tune per scene
        sustained_persistence_frames: int  = 30,     # ~3 seconds at 10fps

        # spike / drop
        spike_roc_threshold: float  =  0.002,        # max plausible density increase per frame
        drop_roc_threshold:  float  = -0.002,        # max plausible density decrease per frame
        spike_persist_frames: int   = 2,             # require persistence to reject glitches

        # trend
        trend_slope_threshold:    float = 0.00001,   # minimum slope to flag
        trend_r_squared_threshold: float = 0.6,      # minimum R² to accept trend as real
    ):
        self.thresholds = {
            "sustained_density":    sustained_density_threshold,
            "sustained_persistence": sustained_persistence_frames,
            "spike_roc":            spike_roc_threshold,
            "drop_roc":             drop_roc_threshold,
            "spike_persist":        spike_persist_frames,
            "trend_slope":          trend_slope_threshold,
            "trend_r_squared":      trend_r_squared_threshold,
        }

        self.zone_states = {
            zone["zone_id"]: ZoneAnomalyState(zone_id=zone["zone_id"])
            for zone in zones
        }

    def evaluate(
        self,
        temporal_states: dict[str, ZoneTemporalState],
        frame_number: int,
    ) -> list[AlertEvent]:
        """
        Evaluate all zones and return any new AlertEvents for this frame.

        Args:
            temporal_states: output of TemporalAggregator.get_all_states()
            frame_number:    current frame index

        Returns:
            list of AlertEvent — may be empty if no anomalies detected
        """
        events = []

        for zone_id, t_state in temporal_states.items():
            if not t_state.is_buffer_ready():
                continue

            a_state = self.zone_states[zone_id]

            events += self._check_sustained(t_state, a_state, frame_number)
            events += self._check_spike_drop(t_state, a_state, frame_number)
            events += self._check_trend(t_state, a_state, frame_number)

            # update alert state on temporal state object
            t_state.alert_state = self._resolve_alert_state(a_state)

        return events

    def _check_sustained(
        self,
        t: ZoneTemporalState,
        a: ZoneAnomalyState,
        frame: int,
    ) -> list[AlertEvent]:
        events = []
        threshold = self.thresholds["sustained_density"]
        persistence = self.thresholds["sustained_persistence"]

        if t.smoothed_density > threshold:
            a.sustained_frame_count += 1
        else:
            # condition cleared
            if a.sustained_frame_count >= persistence:
                events.append(self._make_event(
                    frame, t, SUSTAINED, NORMAL,
                    a.sustained_frame_count, resolved=True
                ))
            a.sustained_frame_count = 0

        if a.sustained_frame_count == persistence:
            events.append(self._make_event(
                frame, t, SUSTAINED, ALERT,
                a.sustained_frame_count
            ))
        elif a.sustained_frame_count > persistence:
            events.append(self._make_event(
                frame, t, SUSTAINED, CRITICAL,
                a.sustained_frame_count
            ))

        return events

    def _check_spike_drop(
        self,
        t: ZoneTemporalState,
        a: ZoneAnomalyState,
        frame: int,
    ) -> list[AlertEvent]:
        events = []
        spike_thresh = self.thresholds["spike_roc"]
        drop_thresh  = self.thresholds["drop_roc"]
        persist      = self.thresholds["spike_persist"]

        if t.rate_of_change > spike_thresh:
            a.spike_frame_count += 1
            a.drop_frame_count   = 0
            if a.spike_frame_count >= persist:
                events.append(self._make_event(
                    frame, t, SPIKE, ALERT,
                    a.spike_frame_count
                ))
        elif t.rate_of_change < drop_thresh:
            a.drop_frame_count  += 1
            a.spike_frame_count  = 0
            if a.drop_frame_count >= persist:
                events.append(self._make_event(
                    frame, t, DROP, ALERT,
                    a.drop_frame_count
                ))
        else:
            a.spike_frame_count = 0
            a.drop_frame_count  = 0

        return events

    def _check_trend(
        self,
        t: ZoneTemporalState,
        a: ZoneAnomalyState,
        frame: int,
    ) -> list[AlertEvent]:
        events = []
        slope_thresh = self.thresholds["trend_slope"]
        r2_thresh    = self.thresholds["trend_r_squared"]

        if (
            t.trend_slope       > slope_thresh
            and t.trend_r_squared > r2_thresh
        ):
            events.append(self._make_event(
                frame, t, TREND, WARNING,
                len(a.active_alerts)
            ))

        return events

    def _resolve_alert_state(self, a: ZoneAnomalyState) -> str:
        """
        Derive the current alert state for a zone based on active conditions.
        """
        if a.sustained_frame_count > self.thresholds["sustained_persistence"] * 2:
            return CRITICAL
        if a.sustained_frame_count >= self.thresholds["sustained_persistence"]:
            return ALERT
        if a.spike_frame_count >= self.thresholds["spike_persist"] \
        or a.drop_frame_count  >= self.thresholds["spike_persist"]:
            return ALERT
        return NORMAL

    @staticmethod
    def _make_event(
        frame: int,
        t: ZoneTemporalState,
        alert_type: str,
        alert_state: str,
        duration_frames: int,
        resolved: bool = False,
    ) -> AlertEvent:
        return AlertEvent(
            timestamp_frame=frame,
            timestamp_wall=datetime.now().isoformat(),
            zone_id=t.zone_id,
            alert_type=alert_type,
            alert_state=alert_state,
            density=t.smoothed_density,
            duration_frames=duration_frames,
            resolved=resolved,
        )
