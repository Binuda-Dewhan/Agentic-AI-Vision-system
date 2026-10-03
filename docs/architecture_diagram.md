# Architecture Diagrams

All diagrams correspond to components defined in [architecture.md](architecture.md).

---

## 1. High-Level System Data Flow

```mermaid
graph TD
    A["📹 Video File"] --> B["Video Ingestion<br/>(OpenCV: decode, metadata)"]
    B --> C["Frame Sampling<br/>(configurable rate)"]
    C --> D["Temporal Segments<br/>(configurable window)"]

    subgraph PERCEPTION ["Deterministic Perception Pipeline"]
        D --> E["Person Detection<br/>(YOLO)"]
        E --> F["Person Tracking<br/>(ByteTrack)"]
        F --> G["Target Person Selection"]
        G --> H["Pose Estimation<br/>(YOLO-Pose)"]
        H --> I["Spatial Analysis<br/>(bed region, if configured)"]
        I --> J["Observation Builder"]
    end

    J --> K["Observation<br/>(structured per-segment data)"]

    subgraph STATE ["Temporal State Engine"]
        K --> L["Evidence Accumulation<br/>(sliding window)"]
        L --> M["State Scoring<br/>(transition-weighted)"]
        M --> N{"Confidence<br/>Check"}
        N -->|"≥ threshold"| O["State Update"]
        N -->|"< threshold"| P["LangGraph<br/>Investigation Agent"]
        P --> O
    end

    subgraph EVENTS ["Bed Event Engine"]
        O --> Q["Transition Sequence<br/>Monitoring"]
        Q --> R{"BED_EXIT /<br/>RETURN_TO_BED<br/>sequence complete?"}
        R -->|"Yes"| S["Confirm Event"]
        R -->|"No"| T["Continue Monitoring"]
    end

    subgraph TIMELINE ["Timeline & Duration Engine"]
        S --> U["Timeline Generation"]
        T --> U
        O --> U
        U --> V["Duration Aggregation"]
        V --> W["Bed Summary<br/>(in-bed / out-of-bed time,<br/>exit count, return count,<br/>longest out-of-bed, final_state)"]
    end

    subgraph SAFETY ["Safety Decision Engine"]
        W --> X["Rule Evaluation<br/>(YAML-configurable)"]
        X --> Y["NORMAL"]
        X --> Z["MONITOR"]
        X --> AA["ALERT"]
    end

    W --> AB["📄 Structured JSON Report"]
    X --> AB
```

---

## 2. Temporal State Engine — Dual-Dimension State Model

The system tracks two independent dimensions: **Activity State** (what the person is doing) and **Bed Context** (relationship to bed). This avoids the ambiguity of a single state field holding both `WALKING` and `OUT_OF_BED`.

### Activity State Transitions

```mermaid
stateDiagram-v2
    [*] --> LYING_IN_BED

    LYING_IN_BED --> SITTING_ON_BED : Sits up
    SITTING_ON_BED --> LYING_IN_BED : Lies back down

    SITTING_ON_BED --> STANDING : Stands up
    STANDING --> SITTING_ON_BED : Sits back on bed

    STANDING --> WALKING : Starts moving
    WALKING --> STANDING : Stops moving

    STANDING --> SITTING_OUTSIDE_BED : Sits away from bed
    WALKING --> SITTING_OUTSIDE_BED : Sits away from bed
    SITTING_OUTSIDE_BED --> STANDING : Stands from seat

    state "Any State" as ANY
    ANY --> UNKNOWN : Insufficient evidence
    UNKNOWN --> ANY : Evidence recovered

    note right of LYING_IN_BED
        Transition rules bias
        toward plausible paths.
        LYING → WALKING requires
        intermediate states.
    end note
```

### Bed Context Transitions

```mermaid
stateDiagram-v2
    [*] --> IN_BED

    IN_BED --> OUT_OF_BED : BED_EXIT event confirmed
    OUT_OF_BED --> IN_BED : RETURN_TO_BED event confirmed

    note right of IN_BED
        Active when activity is
        LYING_IN_BED or SITTING_ON_BED
    end note

    note right of OUT_OF_BED
        Active from BED_EXIT
        until RETURN_TO_BED.
        Coexists with activity state
        (WALKING, STANDING, etc.)
    end note
```

---

## 3. BED_EXIT and RETURN_TO_BED Event Logic

These are **events** confirmed by observing state transition sequences — not states themselves.

```mermaid
graph TD
    subgraph BED_EXIT_SEQUENCE ["BED_EXIT Event Detection"]
        BE1["IN_BED state<br/>(LYING_IN_BED or SITTING_ON_BED)"]
        BE2["STANDING detected<br/>(upright pose, near bed)"]
        BE3["Spatial movement away<br/>(bed overlap decreasing)"]
        BE4{"Sustained outside bed<br/>for ≥ hysteresis window?"}
        BE5["✅ BED_EXIT confirmed"]
        BE6["❌ Not a bed exit<br/>(returned to bed region)"]

        BE1 --> BE2
        BE2 --> BE3
        BE3 --> BE4
        BE4 -->|"Yes"| BE5
        BE4 -->|"No — person returns"| BE6
    end

    subgraph FALSE_EXIT ["False Exit Prevention"]
        F1["Turning in bed<br/>→ no upright transition"]
        F2["Sitting up<br/>→ no STANDING, stays in bed region"]
        F3["Edge sitting<br/>→ no STANDING, partial bed overlap"]
        F4["Brief standing<br/>→ returns within hysteresis window"]
        F5["Stand + sit back<br/>→ returns within hysteresis window"]
    end

    subgraph RETURN_SEQUENCE ["RETURN_TO_BED Event Detection"]
        RT1["OUT_OF_BED<br/>(prior BED_EXIT exists)"]
        RT2["Approaches bed region<br/>(distance decreasing)"]
        RT3["Enters bed region<br/>(overlap above threshold)"]
        RT4["In-bed state confirmed<br/>(SITTING_ON_BED or LYING_IN_BED)"]
        RT5{"Sustained in-bed<br/>for ≥ hysteresis window?"}
        RT6["✅ RETURN_TO_BED confirmed"]

        RT1 --> RT2
        RT2 --> RT3
        RT3 --> RT4
        RT4 --> RT5
        RT5 -->|"Yes"| RT6
    end
```

---

## 4. LangGraph Investigation Agent Workflow

The agent is invoked ONLY when the Temporal State Engine's confidence check falls below threshold. This is a single-pass workflow — it does NOT loop.

```mermaid
graph TD
    START(("Invoked by<br/>State Engine")) --> ASSESS["ASSESS_UNCERTAINTY<br/>What is ambiguous?<br/>(state classification? bed event?<br/>occlusion? multiple persons?)"]

    ASSESS --> GATHER["GATHER_CONTEXT<br/>Tool: get_temporal_context<br/>(prev + current + next segments)<br/>Tool: get_state_history<br/>(recent confirmed transitions)"]

    GATHER --> SUFFICIENT{"Can resolve<br/>from context<br/>alone?"}

    SUFFICIENT -->|"Yes"| DECIDE
    SUFFICIENT -->|"No"| VLM["VLM_ANALYSIS<br/>Tool: analyze_with_vlm<br/>(send representative frames<br/>with focused question)"]

    VLM --> DECIDE["DECIDE<br/>Produce AgentDecision:<br/>- resolved_state<br/>- confidence<br/>- decision_type<br/>- evidence<br/>- uncertainty"]

    DECIDE --> RETURN(("Return to<br/>State Engine"))

    style START fill:#6366f1,color:white
    style RETURN fill:#6366f1,color:white
    style VLM fill:#f59e0b,color:black
```

---

## 5. Safety Decision Flow

All thresholds are configurable via YAML. These are assignment-level monitoring rules, NOT medical recommendations.

```mermaid
graph TD
    INPUT["Current State +<br/>State History +<br/>Duration Tracking +<br/>Confidence"] --> R1

    R1{"Confidence<br/>< monitor_threshold?"}
    R1 -->|"Yes"| MONITOR["🟡 MONITOR<br/>Activity cannot be<br/>confidently determined"]
    R1 -->|"No"| R2

    R2{"SITTING_ON_BED duration<br/>> bed_edge_threshold?"}
    R2 -->|"Yes"| MONITOR
    R2 -->|"No"| R3

    R3{"OUT_OF_BED duration<br/>> prolonged_absence_threshold?"}
    R3 -->|"Yes"| ALERT["🔴 ALERT<br/>Prolonged absence<br/>from bed"]
    R3 -->|"No"| R4

    R4{"UNKNOWN duration<br/>> unknown_threshold?"}
    R4 -->|"Yes"| ALERT
    R4 -->|"No"| NORMAL["🟢 NORMAL<br/>Recognized activity,<br/>no concerning pattern"]

    style MONITOR fill:#f59e0b,stroke:#d97706,color:black
    style ALERT fill:#ef4444,stroke:#dc2626,color:white
    style NORMAL fill:#10b981,stroke:#059669,color:white
```

---

## 6. Evaluation Pipeline

The evaluation pipeline is separate from the runtime pipeline. It compares predictions against manually created ground-truth annotations.

```mermaid
graph TD
    subgraph RUNTIME ["Runtime Pipeline (produces predictions)"]
        V["Video"] --> PIPE["Processing Pipeline"]
        PIPE --> PRED["Predictions:<br/>- State timeline<br/>- Bed events<br/>- Durations<br/>- Safety decisions"]
    end

    subgraph GROUND_TRUTH ["Ground Truth (manual)"]
        GT["Annotations:<br/>- State segments with timestamps<br/>- Bed exit/return event times"]
    end

    subgraph EVAL ["Evaluation Pipeline"]
        PRED --> ALIGN["Temporal Alignment"]
        GT --> ALIGN

        ALIGN --> ACT_EVAL["Activity Evaluation:<br/>- Accuracy<br/>- Per-state P/R/F1<br/>- Confusion matrix"]

        ALIGN --> EVT_EVAL["Bed Event Evaluation:<br/>- BED_EXIT precision/recall<br/>- False positives/negatives<br/>- Temporal tolerance matching"]

        ALIGN --> DUR_EVAL["Duration Evaluation:<br/>- Predicted vs GT duration<br/>- Absolute error per activity"]

        ACT_EVAL --> REPORT["Evaluation Report<br/>(metrics + plots + failure cases)"]
        EVT_EVAL --> REPORT
        DUR_EVAL --> REPORT
    end

    style RUNTIME fill:#1e3a5f,color:#e2e8f0
    style GROUND_TRUTH fill:#3b1f5e,color:#e2e8f0
    style EVAL fill:#1e5128,color:#e2e8f0
```
