# Architecture Review — Phase 0 Final

This document records what was changed, why, and what remains to be decided during implementation.

---

## 1. Changes Made

### 1.1 — Target Person Handling (NEW — §6 in architecture.md)

**Problem:** The previous architecture did not address how the system identifies the primary subject when a caregiver enters the scene.

**Change:** Added explicit target person selection strategy: first consistently detected person becomes the primary track. ByteTrack maintains the track. Multi-person disambiguation uses spatial proximity to last known position and body size. No face recognition.

**Why:** The assignment lists "caregiver entering the scene" as a difficult case. Without explicit target handling, the system could silently track the wrong person.

### 1.2 — Activity State vs. Event Distinction (REVISED — §8, §9, §10)

**Problem:** The previous architecture was ambiguous about whether BED_EXIT is a state or an event. The state diagram mixed states and events.

**Change:**
- Clearly defined: BED_EXIT and RETURN_TO_BED are **events**, not states.
- OUT_OF_BED is a **bed-context meta-state** active from BED_EXIT to RETURN_TO_BED.
- The specific activity state (WALKING, STANDING, SITTING_OUTSIDE_BED) is tracked independently.
- Timeline reports specific activity states; bed summary reports in-bed/out-of-bed time.

**Why:** The assignment lists OUT_OF_BED as one of the 7 core states AND defines BED_EXIT as a detected event. Both must coexist without confusion.

### 1.3 — Return-to-Bed Logic (REVISED — §10)

**Problem:** The previous architecture required the person to lie down before confirming RETURN_TO_BED, matching the assignment's full sequence literally.

**Change:** RETURN_TO_BED is confirmed when the person is in an in-bed state — which includes SITTING_ON_BED. The person does not need to lie down before the return event is confirmed. Subsequent SITTING_ON_BED → LYING_IN_BED is a normal activity transition.

**Why:** The assignment sequence shows "sits on bed → lies down → RETURN_TO_BED" as an example flow, but the meaningful event is re-entering the bed, not specifically lying down. Requiring lying down would delay confirmation unnecessarily and could miss returns where the person sits on the bed for a while.

### 1.4 — False Exit Prevention (EXPANDED — §9)

**Problem:** The previous architecture mentioned hysteresis but did not specify which scenarios it prevents or how.

**Change:** Added explicit table mapping each false-exit scenario to its prevention mechanism: turning (no upright transition), sitting up (no STANDING), edge sitting (no STANDING + partial overlap), brief standing (hysteresis window), stand + sit back (hysteresis window).

**Why:** The assignment specifically lists these as difficult cases. The architecture must demonstrate how each is handled.

### 1.5 — Observation Model (NEW — §7)

**Problem:** The previous architecture did not define the Observation data model.

**Change:** Added complete Observation model with all fields, types, and descriptions. Includes explicit handling of unconfigured bed region (None/-1 values).

**Why:** The Observation is the central data structure connecting perception to reasoning. Its fields determine what the state engine and agent can work with.

### 1.6 — UNKNOWN State Handling (REVISED — §8)

**Problem:** The previous architecture listed UNKNOWN but did not define when it's used or what happens when it persists.

**Change:** UNKNOWN is used when evidence is genuinely insufficient after both the state engine and (optionally) the agent have attempted resolution. If UNKNOWN persists beyond a configurable threshold, it escalates to ALERT via the safety engine.

**Why:** The assignment explicitly requires using UNKNOWN rather than forcing a classification.

### 1.7 — Agent Tools (SIMPLIFIED — §13)

**Problem:** The previous architecture defined 6 tools, several of which overlap.

**Change:** Reduced to 3 tools:
- `get_temporal_context` (merged previous/current/following + visual observations)
- `get_state_history`
- `analyze_with_vlm`

**Why:** The agent almost always needs all adjacent segments. Separate tools increase LLM tool-calling overhead without benefit. Three focused tools are sufficient.

### 1.8 — Agent Workflow (SIMPLIFIED — §11)

**Problem:** The previous LangGraph workflow was too complex for its purpose.

**Change:** Simplified to: ASSESS_UNCERTAINTY → GATHER_CONTEXT → [optional VLM] → DECIDE. Single-pass, no loop. If the agent cannot resolve, it returns low confidence and state remains UNKNOWN.

**Why:** The agent resolves one ambiguity per invocation. It does not need a complex multi-step loop. Simpler graph = more explainable in an interview.

### 1.9 — Bed Region Configuration (REVISED — §20)

**Problem:** The previous architecture implied the bed region is pre-configured or auto-detected.

**Change:** The bed region polygon is scene-specific configuration supplied when a compatible video is available. It is NOT hard-coded and does NOT need to be known during architecture or implementation. The system supports an empty/unconfigured bed region during development. A simple utility (OpenCV click-to-define or manual YAML entry) allows configuration from a representative video frame.

**Why:** Per user requirement — bed region should not be assumed during development.

### 1.10 — Offline vs. Real-Time (NEW — §18)

**Problem:** The previous architecture did not distinguish between offline and real-time modes. Using "following segment" in agent tools implies future context availability, which is only valid in offline mode.

**Change:** Added explicit section documenting that this is offline analysis. Future context is available. Real-time adaptation is noted as future work.

**Why:** The architecture must not imply capabilities it doesn't have.

### 1.11 — Frame Sampling (REVISED — §5.1)

**Problem:** The previous architecture locked frame sampling to "1-2 fps."

**Change:** Frame sampling rate is configurable and will be benchmarked during implementation. No default is locked. Short transitions may require higher rates.

**Why:** Per review requirement — do not prematurely optimize.

### 1.12 — Detection + Pose Model Strategy (REVISED — §5.2)

**Problem:** The previous architecture assumed separate YOLO detection + YOLO-Pose models without evaluating whether a single pose model could serve both purposes.

**Change:** Documented that YOLO-Pose can produce both detections and keypoints. If detection quality from the pose model is sufficient, a separate detection model is unnecessary. Decision deferred to implementation benchmarking. Model checkpoints are configurable.

**Why:** Avoid unnecessary complexity if one model does both jobs well.

### 1.13 — Safety Engine Definitions (REVISED — §15)

**Problem:** The previous architecture used subjective terms like "safe" in safety definitions.

**Change:** Revised to neutral technical definitions: NORMAL = recognized state + sufficient confidence + no monitoring condition; MONITOR = condition warranting observation per configured rules; ALERT = configured threshold exceeded. Explicitly stated these are NOT medical recommendations.

**Why:** Per review requirement — avoid implying medical judgment.

### 1.14 — Evaluation Architecture (NEW — §16)

**Problem:** The previous architecture listed evaluation metrics but did not define the evaluation pipeline, annotation format, or temporal alignment strategy.

**Change:** Added complete evaluation section: annotation format (YAML), temporal alignment, event-level matching with configurable tolerance, separate evaluation pipeline, specific tools (scikit-learn, Matplotlib).

**Why:** The assignment requires evaluation as a deliverable. The architecture must support it structurally.

### 1.15 — Removed FastAPI

**Problem:** FastAPI was listed as optional in the previous architecture.

**Change:** Removed entirely from technology stack.

**Why:** CLI is sufficient. FastAPI adds no evaluation value.

---

## 2. Important Architectural Decisions

| Decision | Rationale |
|----------|----------|
| One agent, not multi-agent | Assignment asks for agentic reasoning, not a multi-agent system. One temporal investigation agent is sufficient and explainable. |
| VLM only for ambiguity | Cost, latency, and architectural principle. Deterministic CV handles the continuous perception workload. |
| ByteTrack for tracking | Specifically justified by the caregiver scenario. Without tracking, multi-person disambiguation is impossible. |
| OUT_OF_BED as meta-state | Reconciles the assignment listing OUT_OF_BED as a state with the need to track specific activities (WALKING, etc.) while out of bed. |
| RETURN_TO_BED on SITTING_ON_BED | Pragmatic: returning to bed is meaningfully confirmed when the person is seated on the bed, not only when lying down. |
| No auto bed detection | Manual configuration is more reliable for this assignment. Auto-detection is unnecessary complexity. |
| Offline-first design | The assignment is video evaluation, not real-time monitoring. Using future context improves accuracy. |
| 3 agent tools, not 6 | Fewer tools = fewer LLM decisions = more predictable agent behavior. |
| Single YOLO-Pose if sufficient | Decision deferred to benchmarking. May eliminate one model entirely. |

---

## 3. Remaining Assumptions

1. Single primary subject per video.
2. Fixed, stationary camera.
3. Indoor room with bed visible in most frames.
4. Continuous single recording.
5. Standard video format and frame rate.
6. Bed region will be configured manually per video.
7. VLM API key available (but system degrades gracefully without).
8. No official dataset — system accepts arbitrary videos, annotations created manually.

---

## 4. Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| No ground-truth dataset | HIGH — cannot evaluate without labels | Create annotation format, manually label short segments |
| VLM cost/latency | MEDIUM — agent invocations have real cost | Minimize invocations, cache results, support VLM-off mode |
| Pose estimation under occlusion | MEDIUM — blankets degrade keypoints | Use visibility scores, fall back to UNKNOWN, document as failure case |
| Tracking failure during long occlusion | MEDIUM — track re-association may fail | Document as limitation, use spatial heuristics for re-association |
| Frame sampling too aggressive | MEDIUM — may miss short transitions | Make rate configurable, benchmark during implementation |
| Interview defensibility | HIGH — candidate must explain everything | Keep codebase small, avoid unnecessary abstraction |
| Temporal thresholds not tunable without data | MEDIUM — all thresholds are guesses initially | Make all configurable, document default rationale |

---

## 5. What Will Be Benchmarked Later

These decisions are intentionally deferred to implementation:

1. **Frame sampling rate** — depends on video content and transition speed.
2. **YOLO model version** — YOLOv8 vs. v11, nano vs. small vs. medium.
3. **Separate detection + pose vs. single YOLO-Pose** — depends on detection quality.
4. **Segment duration** — 2 seconds, 3 seconds, 5 seconds.
5. **Confidence thresholds** — state confirmation, agent invocation.
6. **Hysteresis windows** — bed exit/return confirmation durations.
7. **VLM model selection** — Claude Sonnet vs. Haiku vs. Opus for the investigation agent.

---

## 6. What Should NOT Be Implemented Yet

- Frontend
- Database
- Docker/deployment
- Model training
- Real-time streaming
- Multi-camera support
- Face recognition
- Automatic bed detection
- Complex multi-agent orchestration

---

## 7. Architecture Status

**✅ Ready for implementation review.**

The architecture covers all 19 assignment requirements, defines clear component boundaries, addresses all 15 critical review points (A–O), and maintains implementation feasibility for a single engineer within the assignment timeframe.
