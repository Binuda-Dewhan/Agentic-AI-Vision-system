import logging

from sleep_monitor.config.settings import SafetyRulesConfig
from sleep_monitor.engine.timeline_engine import TimelineEngine
from sleep_monitor.schemas.state import ActivityState, BedContext, SafetyDecision

logger = logging.getLogger(__name__)


class SafetyEngine:
    """
    Evaluates current state and timeline history to determine the SafetyDecision
    (NORMAL, MONITOR, ALERT) according to configurable rules.

    Core assignment rules:
      1. Low confidence / UNKNOWN → MONITOR
      2. Prolonged bed-edge / uncertain sitting → MONITOR
      3. Prolonged out-of-bed → ALERT
      4. Extended UNKNOWN → ALERT

    Optional safety extensions:
      5. Fall-like state (lying + out-of-bed context) → ALERT
      6. Multiple bed exits → ALERT

    All thresholds are configurable engineering parameters for this prototype.
    They are not clinical recommendations.
    """

    def __init__(self, config: SafetyRulesConfig):
        self.config = config
        self.current_decision = SafetyDecision.NORMAL

    def evaluate(
        self,
        current_activity: ActivityState,
        current_context: BedContext,
        timeline_engine: TimelineEngine,
    ) -> SafetyDecision:
        """Evaluate all safety rules and return the highest-severity decision."""

        # --- ALERT-level rules (highest priority) ---

        # Rule 5 (extension): Fall / Floor Detection
        # If the posture is lying but the spatial context is out of bed, they may be on the floor.
        if (
            current_activity == ActivityState.LYING_IN_BED
            and current_context == BedContext.OUT_OF_BED
        ):
            if self.current_decision != SafetyDecision.ALERT:
                logger.warning(
                    "SAFETY ALERT: Person appears to be lying on the floor outside the bed!"
                )
            self.current_decision = SafetyDecision.ALERT
            return self.current_decision

        # Rule 6 (extension): Multiple Exits
        exit_count = sum(
            1 for e in timeline_engine.bed_events if e.event_type == "BED_EXIT"
        )
        if exit_count > self.config.max_exits_before_alert:
            if self.current_decision != SafetyDecision.ALERT:
                logger.warning(
                    f"SAFETY ALERT: Multiple bed exits detected ({exit_count} > {self.config.max_exits_before_alert})"
                )
            self.current_decision = SafetyDecision.ALERT
            return self.current_decision

        # Rule 3 (core): Prolonged Out-of-Bed → ALERT
        if current_context == BedContext.OUT_OF_BED:
            current_out_duration = timeline_engine.current_out_duration

            if current_out_duration > self.config.alert_out_of_bed_duration_sec:
                if self.current_decision != SafetyDecision.ALERT:
                    logger.warning(
                        f"SAFETY ALERT: Person out of bed for too long ({current_out_duration:.0f}s > {self.config.alert_out_of_bed_duration_sec}s)"
                    )
                self.current_decision = SafetyDecision.ALERT
                return self.current_decision

        # Rule 4 (core): Extended UNKNOWN → ALERT
        unknown_duration = timeline_engine.activity_durations.get(
            ActivityState.UNKNOWN.value, 0.0
        )
        if unknown_duration > self.config.alert_unknown_duration_sec:
            if self.current_decision != SafetyDecision.ALERT:
                logger.warning(
                    f"SAFETY ALERT: Extended UNKNOWN state ({unknown_duration:.0f}s > {self.config.alert_unknown_duration_sec}s)"
                )
            self.current_decision = SafetyDecision.ALERT
            return self.current_decision

        # --- MONITOR-level rules ---

        # Rule 1 (core): Current state is UNKNOWN → MONITOR
        if current_activity == ActivityState.UNKNOWN:
            if self.current_decision == SafetyDecision.NORMAL:
                logger.info(
                    "SAFETY MONITOR: Activity cannot be confidently determined."
                )
            self.current_decision = SafetyDecision.MONITOR
            return self.current_decision

        # Rule 2 (core): Prolonged bed-edge sitting → MONITOR
        sitting_on_bed_duration = timeline_engine.activity_durations.get(
            ActivityState.SITTING_ON_BED.value, 0.0
        )
        if sitting_on_bed_duration > self.config.monitor_bed_edge_duration_sec:
            if self.current_decision == SafetyDecision.NORMAL:
                logger.info(
                    f"SAFETY MONITOR: Person sitting on bed edge for unusually long ({sitting_on_bed_duration:.0f}s > {self.config.monitor_bed_edge_duration_sec}s)"
                )
            self.current_decision = SafetyDecision.MONITOR
            return self.current_decision

        # Rule 3 partial: Moderate out-of-bed → MONITOR (before escalating to ALERT)
        if current_context == BedContext.OUT_OF_BED:
            current_out_duration = timeline_engine.current_out_duration
            if current_out_duration > self.config.out_of_bed_monitor_sec:
                if self.current_decision == SafetyDecision.NORMAL:
                    logger.info(
                        f"SAFETY MONITOR: Person out of bed for ({current_out_duration:.0f}s > {self.config.out_of_bed_monitor_sec}s)"
                    )
                self.current_decision = SafetyDecision.MONITOR
                return self.current_decision

        # --- NORMAL ---

        # If person is in bed and no rules fired, recover to NORMAL
        if current_context in (BedContext.IN_BED, BedContext.UNSET):
            if exit_count <= self.config.max_exits_before_alert:
                if self.current_decision != SafetyDecision.NORMAL:
                    logger.info("SAFETY NORMAL: Person is safely back in bed.")
                self.current_decision = SafetyDecision.NORMAL

        return self.current_decision
