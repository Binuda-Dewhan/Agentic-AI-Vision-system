# Safety Alert Rules — Rationale

This document explains the logic behind each alert rule as required by Assignment §6.

---

## Decision Levels

| Decision | Meaning |
|----------|---------|
| **NORMAL** | Recognized activity state with sufficient confidence. No monitoring condition is active. |
| **MONITOR** | A condition exists that warrants continued observation. |
| **ALERT** | A configured safety threshold has been exceeded. |

---

## Rule 1: Fall / Floor Detection

**Condition:** `ActivityState == LYING_IN_BED` AND `BedContext == OUT_OF_BED`

**Classification:** ALERT (highest priority)

**Rationale:** If the pose estimation detects a horizontal body orientation (lying) but the spatial analysis shows the person is outside the bed region, this strongly suggests the person is lying on the floor — a potential fall. This is the most safety-critical scenario and takes priority over all other rules.

**Configurable:** Not threshold-based; this is a logical condition. The bed polygon and pose orientation thresholds indirectly control sensitivity.

**Real-world consideration:** A real monitoring system would immediately notify a caregiver and potentially trigger an emergency response. False positives (e.g., the person deliberately lying on the floor to exercise) would be handled by caregiver override.

---

## Rule 2: Multiple Bed Exits

**Condition:** `bed_exit_count > max_exits_before_alert`

**Default threshold:** `max_exits_before_alert = 3`

**Classification:** ALERT (persists even after return to bed)

**Rationale:** Frequent bed exits during a monitoring period (e.g., overnight) can indicate restlessness, discomfort, or confusion — all of which are clinically relevant for elderly care. The threshold of 3 was chosen as a reasonable default: 1–2 exits per night are common (e.g., bathroom visits), but 4+ exits suggest an abnormal pattern.

**Configurable:** `safety_rules.max_exits_before_alert` in YAML.

**Real-world consideration:** The threshold should be personalized per resident. Some elderly individuals routinely wake more frequently. A real system would learn baseline patterns.

---

## Rule 3: Prolonged Out-of-Bed Duration — ALERT

**Condition:** `BedContext == OUT_OF_BED` AND `current_out_duration > alert_out_of_bed_duration_sec`

**Default threshold:** `alert_out_of_bed_duration_sec = 900` (15 minutes)

**Classification:** ALERT

**Rationale:** If an elderly person leaves their bed and does not return within 15 minutes during a monitoring period (e.g., nighttime), this may indicate they are lost, confused, or have fallen in another room. 15 minutes provides sufficient time for a normal bathroom visit while catching genuinely concerning absences.

**Configurable:** `safety_rules.alert_out_of_bed_duration_sec` in YAML.

**Real-world consideration:** This threshold should account for time of day. A 15-minute absence at 3 AM is more concerning than at 7 AM. This system does not implement time-of-day awareness but documents it as a limitation.

---

## Rule 4: Prolonged Out-of-Bed Duration — MONITOR

**Condition:** `BedContext == OUT_OF_BED` AND `current_out_duration > out_of_bed_monitor_sec`

**Default threshold:** `out_of_bed_monitor_sec = 300` (5 minutes)

**Classification:** MONITOR

**Rationale:** This is the early warning stage before the ALERT threshold. After 5 minutes out of bed, the system flags the situation for monitoring. This gives caregivers awareness without generating a full alert for what may be a normal activity.

**Configurable:** `safety_rules.out_of_bed_monitor_sec` in YAML.

---

## Rule 5: Return to Bed Recovery

**Condition:** `BedContext == IN_BED` AND `exit_count <= max_exits_before_alert`

**Classification:** NORMAL (resets from MONITOR)

**Rationale:** When the person safely returns to bed and hasn't exceeded the maximum exit count, the safety status resets to NORMAL. This prevents lingering MONITOR states after resolved situations. However, if the multiple-exit threshold has been exceeded (Rule 2), the ALERT persists for the remainder of the monitoring period — this is intentional to ensure the pattern is reviewed by a caregiver.

---

## Rule Evaluation Order

Rules are evaluated in priority order:

1. **Fall detection** (immediate ALERT, highest priority)
2. **Multiple exits** (persistent ALERT)
3. **Duration ALERT** (> 15 min out of bed)
4. **Duration MONITOR** (> 5 min out of bed)
5. **Return to bed recovery** (reset to NORMAL)

This ordering ensures that the most critical conditions are never masked by lower-priority rules.
