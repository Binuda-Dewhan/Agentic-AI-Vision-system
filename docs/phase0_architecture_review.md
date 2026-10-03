# Phase 0 — Architecture Review & Validation

## 1. Assignment vs. Proposed Architecture — Requirement Coverage Matrix

| # | Assignment Requirement | Proposed Coverage | Status | Notes |
|---|------------------------|-------------------|--------|-------|
| 1 | Activity/state recognition (7 states) | ✅ All 7 states defined | ✅ COVERED | States match exactly |
| 2 | Temporal transitions (not per-frame) | ✅ Temporal state engine | ✅ COVERED | Core architectural principle |
| 3 | Bed exit detection (multi-step sequence) | ✅ Bed event engine (Phase 6) | ✅ COVERED | Sequence: lying/sitting → standing → moving away → BED_EXIT |
| 4 | Return to bed detection (multi-step) | ✅ Bed event engine (Phase 6) | ✅ COVERED | Sequence: out → approach → sit → lie → RETURN |
| 5 | Sitting up ≠ bed exit | ✅ Hysteresis + temporal evidence | ✅ COVERED | Explicit false-exit prevention |
| 6 | Activity duration calculation | ✅ Timeline engine (Phase 7) | ✅ COVERED | Durations must sum to video duration |
| 7 | Temporal activity timeline | ✅ Timeline engine (Phase 7) | ✅ COVERED | State-transition-based, not per-frame |
| 8 | Agentic analysis (temporal context) | ✅ LangGraph agent (Phase 9) | ✅ COVERED | Uncertainty-driven investigation |
| 9 | NORMAL / MONITOR / ALERT decisions | ✅ Safety engine (Phase 10) | ✅ COVERED | Rule-based, configurable |
| 10 | Expected event output (JSON format) | ✅ Pydantic schemas | ✅ COVERED | Matches assignment JSON examples |
| 11 | Complete summary output (JSON) | ✅ Phase 7 + schemas | ✅ COVERED | All fields from assignment present |
| 12 | Difficult/failure cases (≥3) | ✅ Phase 12 | ✅ COVERED | 6 cases proposed (assignment lists 13 scenarios) |
| 13 | Evaluation (activity, events, duration) | ✅ Phase 11 | ✅ COVERED | Accuracy, precision, recall, F1, confusion matrix, duration error |
| 14 | Deliverables checklist | ✅ Phase 13 | ✅ COVERED | All 9 deliverables addressed |
| 15 | UNKNOWN for insufficient evidence | ✅ Explicit in state engine | ✅ COVERED | Architectural principle #8 |
| 16 | Explain alert rule logic | ⚠️ Implicit in Phase 10 | ⚠️ NEEDS DOCS | Assignment says "explain the logic behind alert rules" — need explicit documentation |
| 17 | Interview defensibility | ✅ Architectural principle #12 | ✅ COVERED | "explain and modify without coding agents" |

---

## 2. Missing Requirements

### 2.1 — `OUT_OF_BED` as a State vs. an Event

> [!IMPORTANT]
> The assignment lists `OUT_OF_BED` as one of the 7 core **states**, not just a bed event outcome. The proposed architecture treats bed exit/return as *events* but must also maintain `OUT_OF_BED` as a trackable activity state with its own duration. The temporal state engine must handle this: once a BED_EXIT event is confirmed, the person's activity state becomes `OUT_OF_BED` (or a more specific state like WALKING/SITTING_OUTSIDE_BED).

**Resolution**: `OUT_OF_BED` is a **meta-state** — it is active whenever the person is confirmed to have exited the bed and not yet returned. The specific sub-activity (WALKING, STANDING, SITTING_OUTSIDE_BED) continues to be tracked. Duration aggregation must report both the specific activities AND the total `time_out_of_bed`.

### 2.2 — `bed_return_count` in Summary Output

The assignment's example complete summary includes `bed_return_count: 2`. The proposed architecture mentions return count in Phase 7 but must ensure this is a first-class tracked metric.

### 2.3 — `longest_out_of_bed_period_sec` 

The assignment example includes this. Phase 7 mentions it, but this requires careful tracking across potentially multiple out-of-bed segments.

### 2.4 — Alert Logic Explanation

Assignment §6 states: *"The candidate should explain the logic behind their alert rules."* This is a documentation requirement, not just implementation. Must produce a clear, interview-ready explanation of each rule.

### 2.5 — `final_state` in Summary

The assignment example summary includes `final_state: "lying_in_bed"`. Must be tracked.

---

## 3. Unnecessary or Over-Engineered Components

### 3.1 — ByteTrack Tracking

> [!NOTE]
> ByteTrack is proposed for person tracking. For a **single elderly person** in a fixed-camera indoor video, full multi-object tracking may be over-engineering. However, it provides:
> - Persistent track IDs (useful when a caregiver enters)
> - Robustness during brief occlusions
> - Clean separation of the primary subject from visitors

**Verdict**: KEEP, but keep it lightweight. The caregiver-enters scenario explicitly requires distinguishing multiple people. ByteTrack is justified.

### 3.2 — FastAPI

Proposed as "optional, only if it materially improves the submission." 

**Verdict**: REMOVE from consideration. A CLI is explicitly sufficient. FastAPI adds no evaluation value and consumes implementation time.

### 3.3 — CSV Output

Phase 7 proposes both JSON and CSV outputs.

**Verdict**: JSON is sufficient for the assignment. CSV is low-cost but optional. Keep if trivial, don't prioritize.

---

## 4. Architectural Risks

### Risk 1: VLM Availability and Cost

| Risk | VLM API calls may be slow, expensive, or rate-limited |
|------|------------------------------------------------------|
| Impact | HIGH — the agentic analysis loop depends on VLM |
| Mitigation | Cache VLM results. Limit invocations to genuinely ambiguous segments. Implement a mock VLM for development/testing. Design the system so it produces useful output even when VLM is unavailable (degraded mode). |

### Risk 2: Bed Region Definition

| Risk | The system needs to know where the bed is to distinguish SITTING_ON_BED from SITTING_OUTSIDE_BED |
|------|------------------------------------------------------|
| Impact | HIGH — spatial reasoning is foundational |
| Mitigation | Three strategies (in priority order): (1) Manual bed-region configuration via config file (most reliable). (2) Auto-detection using YOLO object detection for "bed" class. (3) Heuristic: use the person's location during LYING_IN_BED as the bed region. Strategy 1 should be the default. |

### Risk 3: Pose Estimation Accuracy Under Occlusion

| Risk | Blankets, poor lighting, unusual angles degrade pose estimation |
|------|------------------------------------------------------|
| Impact | MEDIUM — affects body-position features |
| Mitigation | Use visibility/confidence scores per keypoint. When too many keypoints are occluded, classify body position as low-confidence and trigger UNKNOWN or VLM investigation. |

### Risk 4: No Ground-Truth Dataset

| Risk | Assignment does not provide an official dataset or ground-truth labels |
|------|------------------------------------------------------|
| Impact | HIGH — cannot run evaluation without annotations |
| Mitigation | (1) Design a simple annotation format. (2) Use publicly available bedroom monitoring videos for development. (3) Manually annotate a short segment for evaluation. (4) Clearly label as development data. |

### Risk 5: Temporal Smoothing Parameters

| Risk | Thresholds for state confirmation, hysteresis windows, and minimum durations are hard to tune without data |
|------|------------------------------------------------------|
| Impact | MEDIUM — affects false positive/negative rates |
| Mitigation | Make ALL temporal parameters configurable via YAML. Document default values and their rationale. |

### Risk 6: Interview Defensibility with Coding Agents

| Risk | Assignment states: "Candidates should be prepared to explain and modify their implementation during the interview **without coding agents**" |
|------|------------------------------------------------------|
| Impact | HIGH — the implementation must be thoroughly understood |
| Mitigation | Keep codebase small and clear. Avoid excessive abstraction. Document design decisions. Ensure every component can be explained in 2-3 sentences. |

---

## 5. Evaluation Gaps

### Gap 1: Temporal Alignment

The assignment's evaluation compares predicted vs. ground-truth durations, but does not specify how to align predicted timeline segments with ground truth. Need to implement:
- **Segment-level IoU** for timeline evaluation
- **Frame-level accuracy** for state classification
- **Event-level precision/recall** with a temporal tolerance window (e.g., ±5 seconds for bed exit/return timing)

### Gap 2: Confusion Matrix Scope

The assignment says "confusion between similar states." Must specifically analyze:
- LYING_IN_BED ↔ SITTING_ON_BED (the most likely confusion)
- SITTING_ON_BED ↔ SITTING_OUTSIDE_BED (spatial distinction)
- STANDING ↔ WALKING (motion threshold)
- Any state ↔ UNKNOWN (when does the system correctly/incorrectly abstain?)

### Gap 3: False Bed-Exit Analysis

The assignment asks for "false bed-exit detections." Must track:
- Sitting up without leaving (the most important false positive scenario)
- Brief standing → sitting back down
- These must be explicitly documented as failure case studies

---

## 6. Assumptions

1. **Single primary subject**: The video contains one elderly person as the primary subject. Others (caregivers) may appear but are not the analysis target.
2. **Fixed camera**: The camera is stationary. No camera motion compensation needed.
3. **Indoor environment**: Well-defined room with a bed visible in most frames.
4. **Continuous video**: Single continuous recording, not multiple clips.
5. **Video length**: The assignment example uses 20 minutes. The system should handle videos of 10–60 minutes.
6. **Frame rate**: Standard video frame rates (24–30 fps). Processing every frame is not required.
7. **Bed visibility**: The bed is at least partially visible in most frames.
8. **VLM access**: An API key for Claude or equivalent VLM will be available at runtime.

---

## 7. Possible Improvements Over Proposed Architecture

### 7.1 — Segment-Based Processing

Instead of frame-by-frame processing, the video should be divided into **temporal segments** (e.g., 2–5 second windows). Each segment produces one observation. This:
- Reduces computational cost
- Provides natural temporal context
- Aligns with the agent's tool design (get_previous_segment, etc.)

### 7.2 — Degraded Mode Without VLM

The system should produce a complete (if less accurate) output even without VLM access. The deterministic pipeline (YOLO + pose + spatial + temporal) should be the backbone. VLM is an enhancement, not a dependency.

### 7.3 — Configurable Safety Rules as YAML

Rather than hardcoding alert rules, define them in a YAML configuration:

```yaml
safety_rules:
  monitor:
    - condition: "sitting_on_bed_edge"
      duration_threshold_sec: 120
      description: "Prolonged bed-edge sitting"
    - condition: "low_confidence"
      confidence_threshold: 0.5
      description: "Activity cannot be confidently determined"
  alert:
    - condition: "out_of_bed"
      duration_threshold_sec: 600
      description: "Prolonged absence from bed"
    - condition: "unknown_extended"
      duration_threshold_sec: 300
      description: "Extended unclassifiable activity"
```

### 7.4 — Annotated Debug Video Output

Produce an annotated video showing:
- Bounding boxes
- Pose skeleton
- Current state label
- Bed region overlay
- Confidence score
- Event markers

This is invaluable for debugging and impressive for interview demonstrations.

---

## 8. Final Architecture Validation

### ✅ APPROVED Components

| Component | Justification |
|-----------|--------------|
| OpenCV video ingestion | Standard, reliable |
| Frame sampling | Required — can't process every frame |
| YOLO person detection | Pretrained, fast, accurate |
| ByteTrack tracking | Handles caregiver entry, occlusion recovery |
| Pose estimation (YOLO-Pose) | Pretrained, provides body position features |
| Bed-region spatial analysis | Essential for ON_BED vs. OUTSIDE_BED distinction |
| Temporal state engine | Core requirement — transitions, not per-frame classification |
| Bed event engine | Core requirement — multi-step event sequences |
| LangChain + VLM | Provides semantic analysis for ambiguous cases |
| LangGraph agent | Provides temporal investigation loop |
| Safety decision engine | Core requirement — NORMAL/MONITOR/ALERT |
| Pydantic schemas | Structured, validated outputs |
| Evaluation pipeline | Core requirement |

### ❌ REMOVED Components

| Component | Reason |
|-----------|--------|
| FastAPI | CLI is sufficient, no evaluation value |
| Frontend | Explicitly not required |
| Multi-agent system | Over-engineering |
| New model training | Explicitly not required |

### ⚠️ MODIFIED Components

| Component | Modification |
|-----------|-------------|
| VLM integration | Must support degraded mode (no VLM available) |
| Safety rules | Must be YAML-configurable with documented rationale |
| Evaluation | Must include temporal tolerance for event timing |
| Debug output | Add annotated video as optional output |

---

## 9. Architecture Diagram

See [architecture_diagram.md](file:///d:/Personal%20Project_certificate%20work%20Space(Projects%20)/Sleeping%20posture%20detection%20system/docs/architecture_diagram.md) for the Mermaid-based system architecture diagram.

See [architecture_diagram.png](file:///d:/Personal%20Project_certificate%20work%20Space(Projects%20)/Sleeping%20posture%20detection%20system/docs/architecture_diagram.png) for the rendered diagram.

---

## 10. Phase 1 Task List

After architecture approval, Phase 1 will create the project foundation:

### Files to Create

```
sleep_monitor/
├── __init__.py
├── __main__.py              # CLI entry point
├── config/
│   ├── __init__.py
│   ├── settings.py          # Pydantic Settings
│   └── default_config.yaml  # Default configuration
├── schemas/
│   ├── __init__.py
│   ├── video.py             # Video/frame data models
│   ├── detection.py         # Detection/tracking models
│   ├── observation.py       # Observation/pose models
│   ├── state.py             # Activity state models
│   ├── event.py             # Bed event models
│   ├── timeline.py          # Timeline/duration models
│   ├── safety.py            # Safety decision models
│   └── report.py            # Final report model
├── utils/
│   ├── __init__.py
│   └── logging.py           # Logging setup
├── cli.py                   # Click/argparse CLI
tests/
├── __init__.py
├── conftest.py
├── test_schemas.py
├── test_config.py
.env.example
requirements.txt
pyproject.toml
README.md
docs/
├── architecture.md
├── architecture_diagram.md
├── architecture_diagram.png
```

### Dependencies

```
# Core
opencv-python>=4.8
numpy>=1.24
pydantic>=2.0
pydantic-settings>=2.0
pyyaml>=6.0
click>=8.0
python-dotenv>=1.0

# Computer Vision
ultralytics>=8.0     # YOLO detection + pose
                      # (includes ByteTrack)

# AI/Agent
langchain>=0.2
langchain-anthropic>=0.1
langgraph>=0.1

# Data
pandas>=2.0

# Evaluation
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12

# Testing
pytest>=7.0
pytest-cov>=4.0

# Dev
ruff>=0.1            # Linting
```

### Phase 1 Deliverables

1. ✅ Working `pip install -e .` installation
2. ✅ `python -m sleep_monitor --help` CLI
3. ✅ Configuration loading from YAML + env
4. ✅ All Pydantic schemas defined (empty implementations OK)
5. ✅ Logging configured
6. ✅ Test infrastructure passing
7. ✅ README skeleton
8. ✅ .env.example with documented variables

---

## 11. Risks for Phase 1

| Risk | Mitigation |
|------|-----------|
| Dependency version conflicts | Pin major versions, test installation |
| Ultralytics heavy download | First YOLO model download is ~50MB, plan for it |
| Python version | Require 3.11+, document clearly |

---

## 12. Summary

The proposed architecture is **well-aligned** with the assignment requirements. All 17 tracked requirements are covered. The key modifications needed are:

1. **Ensure `OUT_OF_BED` works as both a state and meta-state**
2. **Add degraded mode (no VLM)**
3. **Make safety rules YAML-configurable with documented rationale**
4. **Add temporal tolerance to evaluation metrics**
5. **Remove FastAPI from consideration**
6. **Keep codebase small for interview defensibility**

The architecture is **approved for implementation** pending user confirmation.
