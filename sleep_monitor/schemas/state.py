from enum import Enum


class ActivityState(str, Enum):
    """Activity states representing what the person is doing."""

    LYING_IN_BED = "LYING_IN_BED"
    SITTING_ON_BED = "SITTING_ON_BED"
    SITTING_OUTSIDE_BED = "SITTING_OUTSIDE_BED"
    STANDING = "STANDING"
    WALKING = "WALKING"
    UNKNOWN = "UNKNOWN"


class BedContext(str, Enum):
    """Context regarding the person's relationship to the bed."""

    IN_BED = "IN_BED"
    OUT_OF_BED = "OUT_OF_BED"
    UNSET = "UNSET"  # Used when bed region is not configured or initially


class SafetyDecision(str, Enum):
    """Safety decisions for the current observation or event."""

    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    ALERT = "ALERT"
