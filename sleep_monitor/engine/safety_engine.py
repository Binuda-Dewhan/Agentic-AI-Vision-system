import logging

from sleep_monitor.config.settings import SafetyRulesConfig
from sleep_monitor.engine.timeline_engine import TimelineEngine
from sleep_monitor.schemas.state import ActivityState, BedContext, SafetyDecision

logger = logging.getLogger(__name__)


class SafetyEngine:
    """
    Evaluates current state and timeline history to determine the SafetyDecision
    (NORMAL, MONITOR, ALERT) according to the business rules.
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
        """Evaluate the safety rules and update the current safety decision."""

        # 1. Fall / Floor Detection (Highest Priority)
        # If the posture is lying but the spatial context is out of bed, they are on the floor.
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

        # 2. Multiple Exits Rule
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

        # 3. Time Out Of Bed Rules
        if current_context == BedContext.OUT_OF_BED:
            current_out_duration = timeline_engine.current_out_duration

            if current_out_duration > self.config.alert_out_of_bed_duration_sec:
                if self.current_decision != SafetyDecision.ALERT:
                    logger.warning(
                        f"SAFETY ALERT: Person out of bed for too long ({current_out_duration}s > {self.config.alert_out_of_bed_duration_sec}s)"
                    )
                self.current_decision = SafetyDecision.ALERT

            elif current_out_duration > self.config.out_of_bed_monitor_sec:
                if self.current_decision == SafetyDecision.NORMAL:
                    logger.info(
                        f"SAFETY MONITOR: Person out of bed for ({current_out_duration}s > {self.config.out_of_bed_monitor_sec}s)"
                    )
                    self.current_decision = SafetyDecision.MONITOR
        else:
            # 4. Return to Bed Recovery
            # If they are in bed, and haven't exceeded the max exits, they are NORMAL.
            # (If they exceeded max exits, the alert persists for the shift).
            if exit_count <= self.config.max_exits_before_alert:
                if self.current_decision != SafetyDecision.NORMAL:
                    logger.info("SAFETY NORMAL: Person is safely back in bed.")
                self.current_decision = SafetyDecision.NORMAL

        return self.current_decision
