import logging

from sleep_monitor.config.settings import BedEventsConfig
from sleep_monitor.schemas.events import BedEvent
from sleep_monitor.schemas.perception import Observation
from sleep_monitor.schemas.state import ActivityState, BedContext

logger = logging.getLogger(__name__)


class BedEventEngine:
    """
    Detects BED_EXIT and RETURN_TO_BED events using temporal sequence monitoring
    and hysteresis to prevent false exits.
    """

    def __init__(self, config: BedEventsConfig):
        self.config = config

        # State tracking for event sequences
        self.potential_exit_start_time: float | None = None
        self.potential_return_start_time: float | None = None
        self.state_before_exit: str = ""  # Tracks the state before a potential exit

    def process(
        self,
        current_activity: ActivityState,
        current_context: BedContext,
        observation: Observation,
        previous_activity: ActivityState | None = None,
    ) -> tuple[BedContext, BedEvent | None]:
        """
        Process current state and observation to detect bed events.
        Returns updated BedContext and a BedEvent if one occurred.
        """
        new_context = current_context
        event = None

        # 1. BED_EXIT Logic
        if current_context == BedContext.IN_BED:
            is_out_posture = current_activity in [
                ActivityState.STANDING,
                ActivityState.WALKING,
                ActivityState.SITTING_OUTSIDE_BED,
            ]
            is_out_spatial = observation.majority_spatial_position == "OUTSIDE"

            if is_out_posture and is_out_spatial:
                if self.potential_exit_start_time is None:
                    self.potential_exit_start_time = observation.start_time_sec
                    self.state_before_exit = (
                        previous_activity.value if previous_activity else "UNKNOWN"
                    )
                    logger.debug(
                        f"Potential BED_EXIT started at {self.potential_exit_start_time}s"
                    )

                time_outside = observation.end_time_sec - self.potential_exit_start_time
                if time_outside >= self.config.bed_exit_hysteresis_sec:
                    new_context = BedContext.OUT_OF_BED
                    event = BedEvent(
                        event_type="BED_EXIT",
                        timestamp_sec=self.potential_exit_start_time,
                        confirmed_at_sec=observation.end_time_sec,
                        previous_state=self.state_before_exit,
                        current_state=current_activity.value,
                        confidence=0.9,
                    )
                    logger.info(
                        f"BED_EXIT confirmed! Occurred at {self.potential_exit_start_time}s"
                    )
                    self.potential_exit_start_time = None
            else:
                # False exit prevention: person sat/lay back down before hysteresis threshold
                if self.potential_exit_start_time is not None:
                    logger.debug(
                        f"Potential BED_EXIT aborted at {observation.start_time_sec}s (False exit prevented)"
                    )
                    self.potential_exit_start_time = None

        # 2. RETURN_TO_BED Logic
        elif current_context == BedContext.OUT_OF_BED:
            is_in_posture = current_activity in [
                ActivityState.LYING_IN_BED,
                ActivityState.SITTING_ON_BED,
            ]
            is_in_spatial = observation.majority_spatial_position in [
                "INSIDE",
                "ON_EDGE",
            ]

            if is_in_posture and is_in_spatial:
                if self.potential_return_start_time is None:
                    self.potential_return_start_time = observation.start_time_sec
                    logger.debug(
                        f"Potential RETURN_TO_BED started at {self.potential_return_start_time}s"
                    )

                time_inside = (
                    observation.end_time_sec - self.potential_return_start_time
                )
                if time_inside >= self.config.return_hysteresis_sec:
                    new_context = BedContext.IN_BED
                    event = BedEvent(
                        event_type="RETURN_TO_BED",
                        timestamp_sec=self.potential_return_start_time,
                        confirmed_at_sec=observation.end_time_sec,
                        previous_state=previous_activity.value
                        if previous_activity
                        else "UNKNOWN",
                        current_state=current_activity.value,
                        confidence=0.9,
                    )
                    logger.info(
                        f"RETURN_TO_BED confirmed! Occurred at {self.potential_return_start_time}s"
                    )
                    self.potential_return_start_time = None
            else:
                # Person stood back up or moved outside before hysteresis threshold
                if self.potential_return_start_time is not None:
                    logger.debug(
                        f"Potential RETURN_TO_BED aborted at {observation.start_time_sec}s"
                    )
                    self.potential_return_start_time = None

        return new_context, event
