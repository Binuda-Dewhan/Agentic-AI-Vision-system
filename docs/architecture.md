# Architecture — Elderly Sleep Monitoring System

## 1. Overview

An Agentic AI + Vision system that analyzes continuous indoor video of an elderly person to:

1. Determine what the person is doing over time (activity states).
2. Detect whether the person leaves or returns to bed (events).
3. Calculate how long the person spends in each activity state (durations).
4. Determine whether an observation requires NORMAL, MONITOR, or ALERT handling.

The system prioritizes **temporal understanding** over per-frame classification, using a deterministic computer vision pipeline for continuous perception and a single LangGraph investigation agent for uncertainty resolution.

This is an **offline video analysis** system. The entire video is available before processing begins. This means the system can use both past and future temporal context when resolving ambiguous observations. The architecture does not claim or imply real-time capability.

---

## 2. Assignment Requirements Mapping

| # | Assignment Requirement (§) | Architecture Component | Notes |
|---|---------------------------|----------------------|-------|
| 1 | Activity/state recognition — 7 states (§1) | Temporal State Engine | All 7 states: LYING_IN_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING, WALKING, OUT_OF_BED, UNKNOWN |
| 2 | Transitions, not per-frame classification (§1) | Temporal State Engine | State changes require sustained temporal evidence |
| 3 | Bed exit detection — multi-step sequence (§2) | Bed Event Engine | In-bed → standing → moving away → BED_EXIT event |
| 4 | Return to bed detection — multi-step sequence (§2) | Bed Event Engine | Out-of-bed → approaches → sits/lies on bed → RETURN_TO_BED event |
| 5 | Sitting up ≠ bed exit (§2) | Bed Event Engine | Temporal hysteresis + spatial movement required |
| 6 | Activity duration calculation (§3) | Timeline & Duration Engine | Durations per state, must approximately sum to video duration |
| 7 | Temporal activity timeline (§4) | Timeline & Duration Engine | State transition–based timeline |
| 8 | Agentic temporal context analysis (§5) | LangGraph Investigation Agent | Agent retrieves past/future context for ambiguous observations |
| 9 | NORMAL / MONITOR / ALERT decisions (§6) | Safety Decision Engine | Rule-based, configurable thresholds with documented rationale |
| 10 | Bed-exit event output format (§7) | Pydantic schemas + report | Matches assignment JSON format: event, start_time, confirmed_time, previous_state, current_state, confidence, decision |
| 11 | Complete summary output format (§7) | Pydantic schemas + report | observation_duration_sec, activity_duration_sec, bed_exit_count, bed_return_count, total_in_bed_sec, total_out_of_bed_sec, longest_out_of_bed_period_sec, final_state |
| 12 | Difficult/failure cases — ≥3 (§8) | Failure-case analysis | At least 3 documented failure cases with ground truth, prediction, explanation |
| 13 | UNKNOWN for insufficient evidence (§8) | Temporal State Engine | UNKNOWN is used when evidence is genuinely insufficient, not as an error state |
| 14 | Evaluation: activity accuracy, confusion, failures (§10) | Evaluation Pipeline | Accuracy, precision, recall, F1, confusion matrix |
| 15 | Evaluation: bed-exit precision/recall/false positives (§10) | Evaluation Pipeline | Event-level precision, recall, false positives/negatives |
| 16 | Evaluation: predicted vs ground-truth duration error (§10) | Evaluation Pipeline | Per-activity absolute duration error |
| 17 | Deliverables: source, README, diagram, instructions, timeline, durations, events, evaluation, failures (§11) | All components | See Deliverables Mapping section |
| 18 | Explain alert rule logic (§6) | Safety Decision Engine docs | Documented rationale for each rule |
| 19 | Interview defensibility (§11) | Architecture simplicity | Codebase stays small, every component explainable |

---

## 3. Core Design Principles

1. **Deterministic-first perception.** Computer vision (person detection, pose estimation, spatial analysis) handles continuous perception. The VLM is NOT invoked for every frame.

2. **Temporal state reasoning.** Activity states are determined from accumulated evidence over time. State transitions require sustained, consistent evidence — not a single frame.

3. **Uncertainty-driven agentic investigation.** The LangGraph agent is invoked ONLY when the deterministic pipeline produces ambiguous or low-confidence observations. The agent retrieves temporal context and optionally uses VLM semantic analysis to resolve uncertainty.

4. **Events are not states.** BED_EXIT and RETURN_TO_BED are temporal events confirmed by observing a sequence of state transitions. They are not activity states. OUT_OF_BED is the activity state that becomes active after a BED_EXIT event is confirmed.

5. **Configurable thresholds.** All temporal parameters, confidence thresholds, spatial thresholds, and safety rules are configurable via YAML. Nothing is hard-coded.

6. **Modular VLM interface.** The VLM provider is abstracted behind an interface. Claude is used during development, but the provider can be substituted without architectural changes.

7. **Graceful degradation.** The system produces a complete (if potentially less accurate) output even when VLM access is unavailable. The deterministic pipeline is the backbone; the VLM is an enhancement.

8. **Offline analysis.** The system processes a complete video file. Both past and future temporal context are available. The architecture explicitly does not claim real-time capability.

---

## 4. High-Level Architecture

```
Video File
  → Video Ingestion (OpenCV: decode, metadata, frame sampling)
  → Temporal Segments (configurable window of sampled frames)
  → Perception Pipeline (person detection → tracking → target selection → pose estimation → spatial analysis)
  → Observation (structured per-segment data)
  → Temporal State Engine (evidence scoring → confidence check)
      ├─ sufficient confidence → state update
      └─ insufficient confidence → LangGraph Investigation Agent
                                      → temporal context retrieval
                                      → optional VLM semantic analysis
                                      → structured decision
                                      → state update
  → Bed Event Engine (detect BED_EXIT / RETURN_TO_BED from state transition sequences)
  → Timeline & Duration Engine (timeline, durations, bed summary, final_state)
  → Safety Decision Engine (NORMAL / MONITOR / ALERT from configurable rules)
  → Structured JSON Report
```

See [architecture_diagram.md](architecture_diagram.md) for Mermaid diagrams.

---

## 5. Component Descriptions

### 5.1 Video Ingestion

**Purpose:** Decode a video file, extract metadata, and produce sampled frames grouped into temporal segments.

- Accepts arbitrary video files via OpenCV.
- Extracts metadata: resolution, FPS, duration, codec, total frames.
- Samples frames at a **configurable rate**. The default will be determined during benchmarking against evaluation videos. The rate is NOT locked to any specific value during architecture. Short transitions (e.g., briefly standing) can be missed at very low sampling rates, so the rate must balance computational cost against temporal resolution.
- Groups sampled frames into **temporal segments** — fixed-duration windows (e.g., 2–5 seconds) containing multiple sampled frames. Each segment becomes one unit of observation for downstream processing.
- Preserves frame timestamps relative to video start.

### 5.2 Perception Pipeline

**Purpose:** Extract structured visual features from each temporal segment.

The pipeline runs in sequence for each segment's frames:

1. **Person Detection** — Ultralytics YOLO (model checkpoint configurable, not locked to a specific YOLO version). Detects all persons in each frame with bounding boxes and confidence scores.

2. **Person Tracking** — ByteTrack (via Ultralytics tracker integration). Assigns persistent track IDs across frames. Required because the assignment includes a caregiver-enters-the-scene scenario — the system must distinguish the monitored person from visitors.

3. **Target Person Selection** — See §6 below.

4. **Pose Estimation** — YOLO-Pose or equivalent pretrained pose model (checkpoint configurable). Extracts body keypoints with per-keypoint visibility/confidence scores. Used for body-position classification (lying, sitting, standing) and movement analysis.

5. **Spatial Analysis** — Computes the relationship between the target person and the configured bed region: bounding-box overlap with bed polygon, distance from person center to bed center, whether the person is inside/outside/on-edge of the bed region.

6. **Observation Builder** — Aggregates frame-level detections within a segment into a single structured Observation (see §7).

**Implementation note — Detection + Pose:** YOLO-Pose models can produce both bounding boxes and keypoints in a single inference pass. If a YOLO-Pose model provides sufficient detection quality, a separate detection-only model is unnecessary. If detection quality from the pose model is insufficient, a separate detection model feeds into tracking, and pose estimation runs on the tracked crops. This decision will be benchmarked during implementation, not locked now.

### 5.3 Temporal State Engine

See §8.

### 5.4 LangGraph Investigation Agent

See §11–13.

### 5.5 Bed Event Engine

See §9–10.

### 5.6 Timeline & Duration Engine

See §14.

### 5.7 Safety Decision Engine

See §15.

---

## 6. Target Person Handling

**Problem:** The assignment lists "caregiver entering the scene" as a difficult case. The system must identify and maintain the correct elderly person's track and not confuse it with a visitor.

**Strategy:**

Target person selection is **configurable** via the YAML config:

```yaml
target_person:
  mode: auto    # auto-select first stable subject
```

The `auto` mode is the development default. If the actual evaluation video requires it, a manual target initialization mode can be added later without architectural changes. No face recognition is needed.

1. **Auto mode — primary track selection.** At video start, the first consistently detected person is designated as the primary target. Their ByteTrack ID becomes the monitored track.

2. **Track persistence.** ByteTrack maintains the primary track across frames, handling brief occlusions and re-appearances.

3. **Multi-person disambiguation.** When multiple persons are detected:
   - The system continues tracking the primary target by their established track ID.
   - If the primary track ID is temporarily lost, the system identifies the most likely re-appearance using spatial proximity to the last known position and visual similarity (bounding-box size, position relative to bed).
   - Other detected persons are flagged as `other_persons_present` in the Observation but are NOT analyzed for activity states.

4. **No face recognition.** Face recognition is not used. It is unnecessary complexity for this assignment. Track continuity and spatial reasoning are sufficient for a single-camera, single-subject scenario with occasional visitors.

5. **Edge case — primary person leaves camera view.** When the primary track is lost entirely:
   - The system enters UNKNOWN activity state.
   - If a person reappears and no other person was present, the reappearing track is tentatively reassigned as the primary target.
   - If a different person was present (caregiver), the system uses spatial heuristics (position relative to bed, body size) to distinguish the returning subject from the visitor.

---

## 7. Observation Model

Each temporal segment produces one structured **Observation** containing:

| Field | Type | Description |
|-------|------|-------------|
| `segment_id` | int | Sequential segment identifier |
| `timestamp_start` | float | Start time of segment (seconds from video start) |
| `timestamp_end` | float | End time of segment |
| `target_track_id` | int/None | ByteTrack ID of the monitored person (None if not detected) |
| `target_detected` | bool | Whether the target person was detected in this segment |
| `bbox` | tuple/None | Bounding box of target person (aggregated or representative frame) |
| `detection_confidence` | float | Detection confidence (0–1) |
| `keypoints` | list/None | Pose keypoints with per-keypoint confidence |
| `keypoint_visibility` | float | Fraction of keypoints with sufficient confidence (0–1) |
| `body_orientation` | str/None | Estimated body orientation: horizontal, upright, intermediate |
| `bed_region_overlap` | float | Fraction of person bbox overlapping the bed region (0–1). -1 if bed region not configured |
| `distance_to_bed_center` | float/None | Pixel distance from person center to bed center. None if bed region not configured |
| `position_relative_to_bed` | str/None | Categorical: "on_bed", "bed_edge", "near_bed", "away_from_bed", None if bed region not configured |
| `movement_magnitude` | float | Estimated movement between frames within the segment (pixel displacement) |
| `other_persons_present` | bool | Whether other tracked persons are in the scene |
| `frame_quality` | str | Estimated quality: "good", "low_light", "partially_occluded", "poor" |
| `activity_evidence` | dict | Per-state evidence scores computed from the above features |
| `segment_confidence` | float | Overall confidence in this observation (0–1) |

**Design rationale:**
- Fields that depend on the bed region return None or -1 when no bed region is configured, allowing development without a configured bed polygon.
- `activity_evidence` is computed by the observation builder from the raw features. It provides candidate scores for each possible activity state, which the temporal state engine uses for state determination.
- `frame_quality` provides a simple indicator that downstream components can use to weight observations.

---

## 8. Temporal State Engine

**Purpose:** Determine the current activity state and bed context from accumulated Observations, using temporal evidence rather than per-frame classification.

### Dual-Dimension Internal State Model

Internally, the system tracks **two independent dimensions** to avoid ambiguity:

**Activity State** (what the person is doing):

| State | Definition |
|-------|-----------|
| `LYING_IN_BED` | Person is in a horizontal/reclined position within the bed region |
| `SITTING_ON_BED` | Person is in an upright sitting position within or on the edge of the bed region |
| `SITTING_OUTSIDE_BED` | Person is sitting outside the bed region (e.g., on a chair) |
| `STANDING` | Person is upright and stationary or with minimal movement |
| `WALKING` | Person is upright and moving with significant displacement |
| `UNKNOWN` | Insufficient evidence to determine state |

**Bed Context** (relationship to bed):

| Context | Definition |
|---------|----------|
| `IN_BED` | Person is in the bed region (LYING_IN_BED or SITTING_ON_BED) |
| `OUT_OF_BED` | Person has exited the bed (post–BED_EXIT event, pre–RETURN_TO_BED) |

This dual-tracking prevents the impossible situation where a Pydantic model must hold both `state = WALKING` and `state = OUT_OF_BED` simultaneously. Instead, the person has `activity = WALKING` and `bed_context = OUT_OF_BED`.

### Mapping to Assignment Output

The assignment lists 7 states including OUT_OF_BED. The **report/evaluation output layer** maps the internal representation to the assignment-required format:

- For **timeline output**: the specific activity state is reported (WALKING, STANDING, etc.).
- For **bed summary output**: time is aggregated using bed_context (IN_BED / OUT_OF_BED).
- For **activity duration output**: OUT_OF_BED time is reported as its own entry, computed from bed_context tracking.
- For **state classification evaluation**: the system can produce the assignment's 7-state representation by combining activity + bed_context (e.g., when `bed_context = OUT_OF_BED` and the specific activity is ambiguous, report `OUT_OF_BED`).

### Without Bed Region Configuration

If the bed region is not configured, the system cannot distinguish SITTING_ON_BED from SITTING_OUTSIDE_BED or determine bed context. In this case, it falls back to pose-only state classification (LYING, SITTING, STANDING, WALKING) without bed qualification, and bed events are not detected. Bed context remains unset.

### State Determination Logic

1. **Evidence accumulation.** Each Observation's `activity_evidence` scores are accumulated over a sliding temporal window.

2. **State scoring.** The accumulated evidence produces a confidence-weighted score for each candidate state.

3. **Transition rules.** Not all state transitions are equally likely. The engine uses a transition probability matrix to bias toward plausible transitions (e.g., LYING_IN_BED → SITTING_ON_BED is plausible; LYING_IN_BED → WALKING is implausible without intermediate states).

4. **Confirmation threshold.** A state change is only committed when the new state's score exceeds a configurable confirmation threshold for a configurable minimum duration (temporal smoothing). This prevents rapid flickering between states.

5. **Hysteresis.** Once a state is confirmed, it requires stronger counter-evidence to transition away. This provides stability.

---

## 9. Bed Exit Logic

**BED_EXIT** is a temporal **event**, not a state. It is confirmed when the system observes a specific sequence of state transitions with sufficient temporal evidence.

### Required Sequence

```
IN_BED state (LYING_IN_BED or SITTING_ON_BED)
  → STANDING (within or near bed region)
  → spatial movement away from bed region
  → sustained presence outside bed region
  → BED_EXIT event confirmed
  → activity state continues as STANDING / WALKING / SITTING_OUTSIDE_BED
  → bed-context state becomes OUT_OF_BED
```

### Confirmation Requirements

- The person must have been in an IN_BED state (LYING_IN_BED or SITTING_ON_BED) before the transition.
- The person must be detected as upright (STANDING or WALKING).
- The person must have moved away from the bed region (spatial displacement beyond a configurable threshold, OR bed-region overlap drops below a configurable threshold).
- The out-of-bed state must persist for a **configurable minimum duration** (hysteresis window) before the event is confirmed. This is the primary mechanism for preventing false exits.

### False Exit Prevention

The following scenarios must NOT produce a BED_EXIT event:

| Scenario | Why It Doesn't Trigger BED_EXIT |
|----------|-------------------------------|
| Turning/repositioning while lying | Body remains horizontal, bed overlap remains high, no upright transition |
| Sitting up in bed | Person transitions to SITTING_ON_BED but remains within bed region, no STANDING transition |
| Sitting on the edge of the bed | Person is SITTING_ON_BED (bed-edge), may partially overlap bed region. No STANDING transition or spatial movement away |
| Brief standing beside bed | Person stands but sits/lies back down within the hysteresis window. Timer resets before BED_EXIT is confirmed |
| Standing and sitting back down | Same as above — the sustained-outside-bed requirement prevents premature confirmation |

### BED_EXIT Event Output

```json
{
  "event": "bed_exit",
  "start_time": "00:05:08",
  "confirmed_time": "00:05:20",
  "previous_state": "sitting_on_bed",
  "current_state": "walking",
  "confidence": 0.92,
  "decision": "MONITOR"
}
```

- `start_time`: when the transition away from in-bed state began
- `confirmed_time`: when the hysteresis window elapsed and the event was confirmed
- `decision`: safety classification at the time of event confirmation

---

## 10. Return-to-Bed Logic

**RETURN_TO_BED** is a temporal event confirmed when a person who was out of bed re-enters the bed region and assumes an in-bed state.

### Required Sequence

```
OUT_OF_BED (person was confirmed as having exited bed)
  → approaches bed region (decreasing distance / increasing overlap)
  → enters bed region
  → assumes in-bed state (SITTING_ON_BED or LYING_IN_BED)
  → sustained in-bed evidence
  → RETURN_TO_BED event confirmed
  → bed-context state returns to IN_BED
```

### Important Distinction

The assignment's return sequence shows: approaches → sits on bed → lies down → RETURN_TO_BED. However, the RETURN_TO_BED event is confirmed when the person is established in an in-bed state — this includes SITTING_ON_BED. The person does not need to lie down before the return is confirmed.

After RETURN_TO_BED is confirmed, subsequent transitions (SITTING_ON_BED → LYING_IN_BED) are normal activity state changes, not part of the return event.

### Confirmation Requirements

- The person must have been in OUT_OF_BED (a prior BED_EXIT must exist).
- The person must be detected within the bed region (bed-region overlap above threshold).
- The person must be in SITTING_ON_BED or LYING_IN_BED state.
- The in-bed state must persist for a configurable minimum duration before confirmation.

---

## 11. LangGraph Investigation Agent

**Purpose:** Resolve ambiguous or low-confidence observations by retrieving temporal context and optionally using VLM semantic analysis.

**Invocation trigger:** The agent is invoked ONLY when the Temporal State Engine's confidence check falls below a configurable threshold. It is NOT invoked for every observation.

### Responsibilities

The agent answers questions such as:

- "Is this person actually leaving the bed, or only sitting up / standing briefly?"
- "Is this person lying on the bed or on the floor?"
- "Was this a bed exit that was missed due to occlusion?"
- "Has the person returned to bed or are they sitting near the bed?"

### Graph Workflow

```
START
  → ASSESS_UNCERTAINTY (determine what is ambiguous)
  → GATHER_CONTEXT (retrieve past/future segments, state history)
  → REASON (use temporal context + optionally VLM to resolve)
  → DECIDE (produce structured AgentDecision)
  → END
```

This is a simple, linear, single-pass workflow. It does NOT loop. If the agent cannot resolve the ambiguity, it returns a decision with low confidence and the state remains UNKNOWN.

### Agent Output

The agent returns a structured Pydantic `AgentDecision`:

```
AgentDecision:
  resolved_state: ActivityState       # the agent's best determination
  confidence: float                   # 0.0–1.0
  decision_type: str                  # CONFIRM_STATE, CONFIRM_BED_EXIT, CONFIRM_RETURN, UNABLE_TO_RESOLVE
  evidence: list[str]                 # reasons supporting the decision
  uncertainty: list[str]              # remaining ambiguities
  vlm_used: bool                      # whether VLM was invoked
```

### When VLM Is Used Within the Agent

The VLM is NOT always invoked when the agent runs. The agent first attempts resolution using temporal context (state history, adjacent segment observations). The VLM is invoked only when:

- Temporal context alone is insufficient.
- The ambiguity is semantic (e.g., lying on bed vs. floor, person vs. blanket shape).
- The agent determines that visual inspection of specific frames would resolve the question.

---

## 12. LangChain Responsibilities

LangChain provides:

| Responsibility | Usage |
|---------------|-------|
| **Model abstraction** | Unified interface to Claude / other VLM providers via `langchain-anthropic` or equivalent |
| **Tool definitions** | LangChain Tool wrappers for agent tools (see §13) |
| **Structured output** | Pydantic-validated VLM responses via LangChain's structured output support |
| **Prompt management** | Prompt templates for VLM queries |

LangChain is NOT used for:
- Deterministic state logic (that belongs in the Temporal State Engine)
- Safety rule evaluation (that belongs in the Safety Decision Engine)
- Any processing that doesn't involve LLM/VLM interaction

---

## 13. Agent Tools

The agent has access to the following tools, defined as LangChain tools:

| Tool | Purpose | Justification |
|------|---------|--------------|
| `get_temporal_context` | Retrieve observations from previous, current, and following segments in a single call | The agent needs adjacent temporal context to reason about transitions. Combined into one tool because the agent almost always needs all three. Offline mode allows future context. |
| `get_state_history` | Retrieve the last N confirmed state transitions with timestamps | The agent needs to know what states were recently confirmed to evaluate whether a transition is plausible |
| `analyze_with_vlm` | Send representative frame(s) from a specified segment to the VLM with a focused question | Semantic visual analysis for cases where numerical features are insufficient |

**Removed from previous design:**

- `get_current_segment` / `get_previous_segment` / `get_following_segment` — merged into `get_temporal_context` to reduce unnecessary tool-calling overhead.
- `get_visual_observations` — the observation data is already included in `get_temporal_context`.

**Design rationale:** Three focused tools. The agent can resolve most ambiguities with `get_temporal_context` + `get_state_history`. VLM is invoked only when those are insufficient.

---

## 14. Timeline & Duration Engine

**Purpose:** Produce the temporal activity timeline, activity durations, bed summary, and final state from the sequence of confirmed state transitions.

### Outputs

**Timeline** — ordered list of state segments:

```
00:00 – 04:32  LYING_IN_BED
04:32 – 05:08  SITTING_ON_BED
05:08 – 05:20  STANDING
05:20 – 07:41  WALKING
...
```

**Activity Duration Summary** — total time per state:

```json
{
  "activity_duration_sec": {
    "lying_in_bed": 702,
    "sitting_on_bed": 128,
    "sitting_outside_bed": 95,
    "standing": 63,
    "walking": 167,
    "unknown": 45
  }
}
```

**Bed Summary:**

```json
{
  "total_in_bed_sec": 830,
  "total_out_of_bed_sec": 370,
  "bed_exit_count": 2,
  "bed_return_count": 2,
  "longest_out_of_bed_period_sec": 241
}
```

**Final state:** The last confirmed activity state at video end.

**Constraint:** Activity durations must approximately sum to the total video observation duration.

---

## 15. Safety Decision Engine

**Purpose:** Classify each observation or event as NORMAL, MONITOR, or ALERT based on configurable rules.

### Definitions

| Decision | Definition |
|----------|-----------|
| **NORMAL** | Recognized activity state with sufficient confidence. No configured monitoring condition is active. |
| **MONITOR** | A condition exists that warrants continued observation according to configured rules. Either low-confidence classification, or a potentially relevant temporal pattern. |
| **ALERT** | A configured event or condition has exceeded its threshold. |

These are **assignment-level monitoring rule classifications**. They are NOT medical recommendations.

### Rule Examples

The following are example rules. All thresholds are configurable via YAML:

| Rule | Condition | Classification | Rationale |
|------|-----------|---------------|-----------|
| Low confidence | `segment_confidence < threshold` | MONITOR | Activity cannot be confidently determined |
| Prolonged bed-edge sitting | SITTING_ON_BED for > duration threshold | MONITOR | Prolonged edge sitting may indicate difficulty/hesitation |
| Prolonged out-of-bed | OUT_OF_BED for > duration threshold | ALERT | Unexpected prolonged absence from bed |
| Extended UNKNOWN | UNKNOWN for > duration threshold | ALERT | Extended inability to determine activity |
| Normal activity | All else | NORMAL | Recognized activity, no concerning pattern |

### Rule Logic Documentation

The candidate is required to explain the logic behind alert rules (assignment §6). Each rule will be documented with:

1. What condition triggers it.
2. What the configurable threshold is.
3. Why this condition was selected.
4. What a real monitoring system might do differently.

---

## 16. Evaluation Architecture

The evaluation pipeline is **separate** from the runtime pipeline. It compares system predictions against ground-truth annotations.

### Annotation Format

A simple JSON/YAML annotation format per video:

```yaml
video_file: "example.mp4"
annotator: "manual"
segments:
  - start_time: 0.0
    end_time: 272.0
    state: "lying_in_bed"
  - start_time: 272.0
    end_time: 308.0
    state: "sitting_on_bed"
  # ...
events:
  - type: "bed_exit"
    time: 320.0
  - type: "return_to_bed"
    time: 555.0
```

### Evaluation Metrics

**Activity Classification (§10):**

| Metric | Description |
|--------|------------|
| Accuracy | Overall frame-level or segment-level state classification accuracy |
| Per-state precision/recall/F1 | Per activity state |
| Confusion matrix | All states × all states, with attention to similar-state confusion (LYING_IN_BED ↔ SITTING_ON_BED, SITTING_ON_BED ↔ SITTING_OUTSIDE_BED, STANDING ↔ WALKING) |
| Failure cases | Specific examples with ground truth, prediction, explanation |

**Bed Events (§10):**

| Metric | Description |
|--------|------------|
| BED_EXIT precision | Fraction of predicted bed exits that were correct |
| BED_EXIT recall | Fraction of actual bed exits that were detected |
| False positives | Predicted bed exits that did not occur |
| False negatives | Actual bed exits that were missed |
| RETURN_TO_BED evaluation | Same metrics where ground-truth return events are annotated |
| Temporal tolerance | Events are matched with a configurable time tolerance (e.g., ±10 seconds) |

**Duration (§10):**

| Metric | Description |
|--------|------------|
| Per-activity predicted vs ground-truth duration | Side-by-side comparison |
| Absolute duration error | |predicted - ground_truth| per activity |
| Relative error | Optional: absolute error / ground_truth duration |

### Evaluation Tools

- scikit-learn for classification metrics and confusion matrix
- Matplotlib for visualizations (confusion matrix heatmap, timeline comparison plot)
- pandas for tabular reporting

---

## 17. Failure-Case Strategy

The architecture must support identification and documentation of at least 3 failure cases (assignment §8, §11).

### Planned Failure Cases

| # | Scenario | Expected Difficulty | How Architecture Handles It |
|---|----------|--------------------|-----------------------------|
| 1 | Sitting up / brief standing incorrectly interpreted as bed exit | The most important false-positive scenario. Hysteresis and spatial movement requirements should prevent this, but edge cases may fail. | Temporal hysteresis + spatial displacement threshold. If both are insufficient, the failure is documented. |
| 2 | Partial occlusion by blankets | Pose estimation degrades. Keypoints become invisible. Body orientation may be misclassified. | `keypoint_visibility` and `frame_quality` fields trigger UNKNOWN or agent investigation. If agent + VLM cannot resolve, state remains UNKNOWN. |
| 3 | Caregiver entering the scene | A second person may confuse tracking. The caregiver's track may be misidentified as the target. | ByteTrack maintains the primary track. `other_persons_present` flag is set. If tracking fails, this becomes a documented failure. |

### Additional Cases (If Time Permits)

| # | Scenario |
|---|----------|
| 4 | Person leaves camera view entirely |
| 5 | Poor/changing lighting conditions |
| 6 | Very short transition (e.g., stands for 2 seconds, sits back) |
| 7 | Ambiguous bed-edge sitting (is the person on the bed or beside it?) |

### Documentation Format

Each failure case will include:

1. **Scenario description** — what happens in the video
2. **Ground truth** — what the correct classification/event should be
3. **System prediction** — what the system actually produced
4. **Why it failed** — root cause analysis
5. **How the architecture handles uncertainty** — whether UNKNOWN was used, whether the agent was invoked, whether the VLM helped
6. **Potential improvements** — what could be done with more time/data

---

## 18. Offline vs. Real-Time Distinction

This system is designed for **offline video analysis**. The entire video file is available before processing begins.

### Implications

| Aspect | Offline (This System) | Real-Time (Not Implemented) |
|--------|----------------------|---------------------------|
| Future context | Available — the agent can retrieve following segments | Not available — reasoning must be causal |
| Latency | Not a constraint — thorough analysis is prioritized | Critical — must produce results within seconds |
| VLM usage | Can re-analyze segments | Must decide in real-time, cannot wait for VLM |
| Bed event confirmation | Can use look-ahead to confirm exits/returns | Must use only past + current evidence |
| Frame sampling | Can be adjusted after initial analysis | Must be decided a priori |

### Future Real-Time Adaptation

If the system were adapted for real-time:
- Remove `get_following_segment` / future context from agent tools.
- Increase hysteresis windows (slower confirmation without look-ahead).
- Use VLM asynchronously with delayed confirmation.
- This is NOT implemented in this assignment.

---

## 19. Technology Stack

| Category | Technology | Justification |
|----------|-----------|--------------|
| Language | Python 3.11+ | Assignment standard |
| Video | OpenCV | Reliable, standard video decoding |
| Detection + Pose | Ultralytics YOLO (version benchmarked during implementation) | Pretrained, fast, provides both detection and pose. Model checkpoint configurable. |
| Tracking | ByteTrack (via Ultralytics) | Handles multi-person disambiguation, occlusion recovery. Required for caregiver scenario. |
| Agent orchestration | LangGraph | Stateful workflow with conditional branching for investigation |
| LLM/VLM abstraction | LangChain | Model abstraction, tool definitions, structured output |
| VLM provider | Claude (primary, via langchain-anthropic) | Multimodal, strong reasoning. Interface abstracted for substitution. |
| Schemas | Pydantic v2 | Validated structured data throughout |
| Configuration | PyYAML + pydantic-settings | YAML config files with env-var overrides |
| Numerical | NumPy | Array operations, spatial math |
| Data analysis | pandas | Tabular evaluation data, duration tables |
| Evaluation | scikit-learn, Matplotlib | Classification metrics, confusion matrix, plots |
| Testing | pytest | Unit testing |
| CLI | Click | Simple command-line interface |

### Not Included

| Technology | Reason for Exclusion |
|-----------|---------------------|
| FastAPI | CLI is sufficient. No evaluation value for this assignment. |
| Frontend | Explicitly not required. |
| Database | No persistence needed beyond JSON output files. |
| Docker | Not required for submission. Can be added trivially later. |
| Multi-agent framework | Over-engineering. One agent is sufficient. |

---

## 20. Configuration Strategy

All configurable parameters are stored in a YAML configuration file with environment variable overrides for secrets (API keys).

### Configuration Categories

**Video processing:**
- `frame_sample_rate`: Frames per second to sample (benchmarked during implementation)
- `segment_duration_sec`: Temporal segment window size

**Perception:**
- `detection_model`: Path/name of YOLO detection model checkpoint
- `pose_model`: Path/name of YOLO-Pose model checkpoint
- `detection_confidence_threshold`: Minimum detection confidence
- `keypoint_visibility_threshold`: Minimum fraction of visible keypoints

**Target person:**
- `target_person.mode`: Target selection mode (`auto` default). Extensible to manual initialization later.

**Bed region:**
- `bed_region_polygon`: List of (x, y) vertices defining the bed region polygon. **This is scene-specific configuration supplied when a compatible video is available.** It is NOT hard-coded and does NOT need to be known during architecture or implementation. The system supports an empty/unconfigured bed region during development. When ready to configure, a utility extracts a representative frame from the video, and the user defines the polygon on that frame (e.g., via a simple OpenCV click-to-define tool or manual coordinate entry in the YAML).
- `bed_edge_margin`: Pixel margin around bed polygon for "on edge" classification

**Temporal state engine:**
- `state_confirmation_window_sec`: Minimum duration before committing a state change
- `state_transition_hysteresis`: Counter-evidence multiplier for leaving current state
- `confidence_threshold`: Below this, the agent is invoked

**Bed events:**
- `bed_exit_hysteresis_sec`: Minimum out-of-bed duration before confirming BED_EXIT
- `bed_exit_spatial_threshold`: Minimum distance from bed before exit is considered
- `return_hysteresis_sec`: Minimum in-bed duration before confirming RETURN_TO_BED

**Safety rules:**
- `monitor_confidence_threshold`: Below this, classify as MONITOR
- `monitor_bed_edge_duration_sec`: Edge-sitting duration threshold for MONITOR
- `alert_out_of_bed_duration_sec`: Out-of-bed duration threshold for ALERT
- `alert_unknown_duration_sec`: Unknown duration threshold for ALERT

**VLM:**
- `vlm_provider`: Provider name (e.g., "anthropic")
- `vlm_model`: Model name (e.g., "claude-sonnet-4-20250514")
- `vlm_max_frames_per_query`: Maximum frames sent per VLM call
- `vlm_enabled`: Boolean — allows disabling VLM for degraded mode

**Environment variables (.env):**
- `ANTHROPIC_API_KEY`

---

## 21. Deliverables Mapping

| Assignment Deliverable (§11) | System Output |
|-----------------------------|---------------|
| Source code | `sleep_monitor/` Python package |
| README | `README.md` — overview, setup, run instructions, example commands |
| Simple architecture diagram | `docs/architecture_diagram.png` and `docs/architecture_diagram.md` |
| Instructions to run the system | `README.md` + `python -m sleep_monitor --help` |
| Activity timeline | JSON output: ordered state segments with timestamps |
| Activity-duration summary | JSON output: per-state durations + bed summary |
| Bed-exit/return events | JSON output: event list with timestamps, states, confidence, decision |
| Evaluation results | `docs/evaluation_report.md` + generated metrics/plots |
| At least 3 failure-case examples | `docs/failure_cases.md` — documented with ground truth, predictions, analysis |

---

## 22. Known Assumptions and Limitations

### Assumptions

1. **Single primary subject.** One elderly person is the monitoring target. Others may appear but are not analyzed.
2. **Fixed camera.** The camera is stationary. No camera-motion compensation is implemented.
3. **Indoor environment.** A room with a bed visible in most frames.
4. **Continuous video.** A single continuous recording, not multiple clips.
5. **Standard video.** Standard frame rates (15–30 fps), standard codecs.
6. **Bed region supplied per-scene.** The bed polygon is configured manually when a video is available. Not auto-detected.
7. **VLM API access.** An API key is available at runtime (but the system degrades gracefully without one).
8. **No dataset provided.** The assignment does not specify a dataset. The system accepts arbitrary compatible videos. Evaluation annotations are created manually for available test videos.

### Limitations

1. **No real-time capability.** The system processes completed video files.
2. **No automatic bed detection.** Bed region must be manually configured.
3. **Single camera only.** No multi-view reasoning.
4. **Pose estimation under occlusion.** Heavy blanket coverage degrades pose accuracy. The system falls back to UNKNOWN rather than guessing.
5. **VLM latency and cost.** Each VLM call has latency and monetary cost. The architecture minimizes calls but cannot eliminate them for ambiguous cases.
6. **Tracking across long occlusions.** If the primary person is fully occluded for extended periods, track re-association may fail.
7. **No training.** The system uses pretrained models only. Performance is bounded by those models' capabilities on this domain.
