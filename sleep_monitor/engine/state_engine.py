import logging
from collections import deque

from sleep_monitor.config.settings import TemporalStateConfig
from sleep_monitor.schemas.perception import Observation
from sleep_monitor.schemas.state import ActivityState, BedContext

logger = logging.getLogger(__name__)


class StateEngine:
    """
    Determines the current activity state from accumulated Observations.
    Implements sliding window evidence accumulation and hysteresis.
    """

    def __init__(self, config: TemporalStateConfig):
        self.config = config
        self.current_activity = ActivityState.UNKNOWN
        self.current_bed_context = BedContext.UNSET

        # Sliding window of recent observations
        self.observation_window: deque[Observation] = deque()

    def process_observation(self, obs: Observation) -> tuple[ActivityState, BedContext]:
        """
        Process a new observation, update internal state, and return the new state.
        """
        self.observation_window.append(obs)

        # 1. Prune old observations outside the confirmation window
        while self.observation_window:
            window_duration = (
                obs.end_time_sec - self.observation_window[0].start_time_sec
            )
            if (
                window_duration > self.config.state_confirmation_window_sec
                and len(self.observation_window) > 1
            ):
                self.observation_window.popleft()
            else:
                break

        # 2. Accumulate evidence over the window
        scores = {state: 0.0 for state in ActivityState}

        for w_obs in self.observation_window:
            ev = w_obs.average_evidence
            scores[ActivityState.LYING_IN_BED] += ev.lying

            if w_obs.majority_spatial_position in ["INSIDE", "ON_EDGE"]:
                scores[ActivityState.SITTING_ON_BED] += ev.sitting
            elif w_obs.majority_spatial_position == "OUTSIDE":
                scores[ActivityState.SITTING_OUTSIDE_BED] += ev.sitting
            else:
                # Unconfigured fallback
                scores[ActivityState.SITTING_OUTSIDE_BED] += ev.sitting

            scores[ActivityState.STANDING] += ev.standing
            scores[ActivityState.WALKING] += ev.walking

        # 3. Apply hysteresis to provide stability to the current state
        if self.current_activity != ActivityState.UNKNOWN:
            scores[self.current_activity] *= self.config.state_transition_hysteresis

        # 4. Determine new activity state
        best_state = max(scores, key=scores.get)
        best_score = scores[best_state]
        avg_score = (
            best_score / len(self.observation_window)
            if self.observation_window
            else 0.0
        )

        # If no target frames or score is too low, fall back to UNKNOWN
        valid_frames = sum(1 for frame in obs.frames if frame.bbox is not None)
        if valid_frames == 0 or avg_score < self.config.confidence_threshold:
            new_activity = ActivityState.UNKNOWN
        else:
            new_activity = best_state

        # 5. Log state transitions
        if new_activity != self.current_activity:
            logger.info(
                f"Activity transition: {self.current_activity} -> {new_activity} (score: {avg_score:.2f})"
            )
            self.current_activity = new_activity

        # 6. Set initial bed context if unset
        # (BedEventEngine will handle actual transitions between IN_BED and OUT_OF_BED later)
        if (
            self.current_bed_context == BedContext.UNSET
            and self.current_activity != ActivityState.UNKNOWN
        ):
            if self.current_activity in [
                ActivityState.LYING_IN_BED,
                ActivityState.SITTING_ON_BED,
            ]:
                self.current_bed_context = BedContext.IN_BED
            else:
                self.current_bed_context = BedContext.OUT_OF_BED

        return self.current_activity, self.current_bed_context
