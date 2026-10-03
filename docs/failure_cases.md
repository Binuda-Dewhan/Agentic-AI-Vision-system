# Failure Case Analysis

This document details three difficult scenarios (as requested in Assignment §8 and §11) that stress the system's perception and temporal tracking logic. For each case, we analyze the scenario, the expected ground truth, the potential system prediction, the root cause of the failure, and how the architecture is designed to handle or degrade gracefully under these conditions.

---

## Case 1: The "False Bed Exit" (Brief Standing / Repositioning)

**Scenario:** 
An elderly person is lying in bed. They wake up, sit on the edge of the bed for 5 seconds, stand up right next to the bed for 4 seconds (perhaps to stretch or adjust clothing), and then sit back down and lie back in bed.

**Ground Truth:**
- `00:00 - 00:05`: SITTING_ON_BED
- `00:05 - 00:09`: STANDING (In-bed context)
- `00:09 - 00:15`: SITTING_ON_BED / LYING_IN_BED
- **Events:** No BED_EXIT should be generated.

**Potential System Prediction (Failure Mode):**
- State sequence matches ground truth.
- **Events:** `BED_EXIT` generated at 00:05, `RETURN_TO_BED` generated at 00:09.

**Why it fails:**
If the person stands near the edge of the bed, the bounding box might temporarily shift outside the strict bed polygon, and the pose estimation confidently detects `STANDING`. If the temporal hysteresis window for bed exits is set too short (e.g., 3 seconds), the system will prematurely confirm a bed exit before the person sits back down.

**How the architecture handles it:**
The architecture mitigates this using a **dual-threshold approach**:
1. **Spatial Hysteresis:** The person must move a configurable distance away from the bed center (`bed_exit_spatial_threshold`). Standing next to the bed usually does not exceed this distance.
2. **Temporal Hysteresis:** The `bed_exit_hysteresis_sec` config (default 10s) requires the person to remain out-of-bed for a sustained period before the event is confirmed. In this 4-second standing scenario, the exit timer starts, but aborts when they sit back down.

*Improvement:* If the hysteresis is too long, real rapid bed exits (e.g. rushing to the bathroom) might have delayed detection. In a production system, this could be tuned per resident or augmented with a depth camera to better judge floor contact.

---

## Case 2: Heavy Occlusion (Thick Blankets)

**Scenario:**
The person is lying in bed, but pulls a thick, patterned comforter entirely over themselves, leaving perhaps only the top of their head visible. They turn over under the blanket.

**Ground Truth:**
- State: `LYING_IN_BED` (continuously)

**Potential System Prediction (Failure Mode):**
- State transitions to `UNKNOWN`.
- Target tracker loses the person entirely, dropping the ByteTrack track ID.

**Why it fails:**
YOLO and YOLO-Pose rely on visual features of the human body (limbs, torso, face). Under heavy occlusion, the detection confidence drops below `detection_confidence_threshold`, or keypoints become invisible (`keypoint_visibility_threshold` is not met). The deterministic perception pipeline fails to extract meaningful features, resulting in low activity evidence.

**How the architecture handles it:**
Instead of guessing or hallucinating states, the system fails safely:
1. **UNKNOWN Fallback:** When evidence is low, the Temporal State Engine transitions to `UNKNOWN`.
2. **LangGraph Agent Invocation:** The state engine's low confidence triggers the Investigation Agent. 
3. **VLM Semantic Analysis:** The agent uses the `analyze_with_vlm` tool, sending the frames to Claude. While YOLO fails on the blob-like blanket, the VLM can contextually understand "A bed with a person under a thick blanket" and return a confident `LYING_IN_BED` decision, overriding the deterministic failure.

*Improvement:* If the VLM also fails, the system logs the duration of the `UNKNOWN` state. The Safety Engine monitors this duration (`alert_unknown_duration_sec`). If the person is "lost" for too long (e.g. 5 minutes), it escalates to an `ALERT` to notify caregivers to check the room.

---

## Case 3: The Caregiver Intrusion (Multi-Person Confusion)

**Scenario:**
The elderly person is sitting on the edge of the bed (`SITTING_ON_BED`). A caregiver enters the room, walks over to the bed, stands directly in front of the elderly person (partially occluding them), hands them a glass of water, and walks away.

**Ground Truth:**
- Elderly Person State: `SITTING_ON_BED` (continuously).
- No bed exit.

**Potential System Prediction (Failure Mode):**
- The tracker switches IDs. The system starts tracking the caregiver as the primary target.
- State sequence: `SITTING_ON_BED` → `STANDING` → `WALKING` → `OUT_OF_BED`.
- **Events:** `BED_EXIT` incorrectly generated.

**Why it fails:**
When two people are in close proximity and one occludes the other, bounding box intersection-over-union (IoU) tracking can swap identities. If the caregiver walks away, the tracker might assign the elderly person's track ID to the caregiver. The system then analyzes the caregiver's actions (walking away) and incorrectly attributes them to the elderly person.

**How the architecture handles it:**
1. **ByteTrack Persistence:** ByteTrack is specifically chosen over simple SORT because it recovers better from partial occlusions using visual appearance features, not just kalman-filter motion.
2. **Target Selector Disambiguation:** The `TargetSelector` component monitors the scene. When multiple people are detected (`other_persons_present=True`), it uses spatial anchors. It knows the primary target's last known state was `SITTING_ON_BED` (inside the bed region). When the occlusion clears, it re-associates the primary track ID with the bounding box that is still sitting on the bed, rather than the one walking towards the door.
3. **Event Hysteresis:** Even if a track swap briefly occurs, the `bed_exit_hysteresis_sec` provides a buffer window for the tracker to correct itself before committing a false `BED_EXIT` to the timeline.

*Improvement:* For persistent multi-person tracking issues, the pipeline could be augmented with ReID (Re-Identification) models that extract stronger appearance embeddings (clothing color, height) to strictly separate the caregiver from the resident.
