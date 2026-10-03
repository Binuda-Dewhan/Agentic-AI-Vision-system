# 🛏️ Elderly Sleep Monitoring System

An **Agentic AI + Vision system** built to monitor the safety and sleep quality of elderly individuals. It analyzes continuous indoor video footage to precisely determine activity states, track bed exits and returns, and automatically trigger safety alerts when necessary.

This project was built as a solution for the **Associate AI/ML Engineer Assignment**, demonstrating a production-ready, highly robust computer vision and agentic reasoning pipeline.

---

## 🏗️ Architecture & Development Strategies

This system does **not** rely on a black-box LLM processing every single frame. Instead, it employs a highly efficient **Hybrid Architecture** that combines deterministic Computer Vision with Agentic Vision-Language Models (VLMs) as a semantic fallback.

### 1. The Core Vision Pipeline (Deterministic & Fast)
- **Detection & Tracking:** Uses `YOLOv8n` + `ByteTrack` to track individuals across the frame. If tracking ID is briefly lost (due to occlusion), the pipeline intelligently falls back to spatial correlation to prevent silent frame drops.
- **Pose Estimation:** Uses `YOLOv8n-Pose` to extract 17 keypoints. 
- **Robust Orientation Classifier:** Solving the "side-camera" problem. A person lying down filmed from the foot of the bed appears as a tall, vertical shape in 2D space, which confuses standard angle algorithms. This system uses a **multi-signal approach**: verifying the full-body span angle, the bounding-box aspect ratio, and the lateral displacement of hips-to-shoulders to achieve near-perfect lying vs standing detection regardless of camera placement.
- **Spatial Analyzer:** Calculates bounding box overlaps and proximity to a configured `bed_region_polygon` to establish contextual awareness (e.g., `INSIDE`, `ON_EDGE`, `OUTSIDE`).

### 2. Temporal State Engine (Hysteresis)
Analyzing video frame-by-frame leads to flickering (e.g., sitting up for 1 second shouldn't count as a bed exit).
- The `StateEngine` accumulates probabilistic evidence over a sliding temporal window.
- **Hysteresis:** A state transition only occurs if a new state sustains high confidence for a configured duration (e.g., 10 seconds for a bed exit). 

### 3. Agentic VLM Fallback (LangGraph + Gemini)
When the deterministic pipeline fails (e.g., the person is completely tangled in blankets, lighting is poor, or YOLO loses confidence), the state drops to `UNKNOWN`.
- A **LangGraph-based Investigation Agent** is automatically invoked.
- It pulls the surrounding frames and temporal context and passes them to a Vision-Language Model (`gemini-3.8-flash`).
- **Fail-Fast Quota Handling:** If the Gemini API hits a rate limit (`429 RESOURCE_EXHAUSTED`), the SDK is configured to fail instantly rather than hanging. The pipeline gracefully degrades, logging `UNKNOWN`, ensuring the video processing never freezes.

### 4. Safety Decision Engine
Automatically outputs a safety status based on configurable context:
- `NORMAL`: Typical sleeping or moving behavior.
- `MONITOR`: Prolonged sitting on the edge of the bed or an unconfirmed state.
- `ALERT`: Fall detection (e.g., lying horizontally while spatially out of bed), multiple bed exits in one night, or a prolonged absence (e.g., out of bed for >15 minutes).

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11+
- Google API Key (for the Gemini VLM investigation agent)

### Installation

1. Clone the repository and navigate into it.
2. Install the package and its dependencies:

```bash
pip install -e .
```

3. Create a `.env` file in the root directory to authorize the VLM agent:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 💻 Running the System

The system provides a unified CLI via the `sleep_monitor` command.

### 1. Analyze a Video
To analyze a video and generate a full timeline and JSON report:

```bash
python -m sleep_monitor analyze path/to/video.mp4
```

**Options:**
- `-c, --config`: Path to a custom YAML configuration file.
- `-o, --output`: Path to save the output JSON report (defaults to `output/{video_name}_report.json`).
- `-v, --verbose`: Enable debug logging.

Example:
```bash
python -m sleep_monitor analyze 01.mp4 -o output/01_report.json -v
```

### 2. Evaluate Against Ground Truth
If you have a manually annotated YAML file, you can evaluate the system's accuracy:

```bash
python -m sleep_monitor evaluate output/01_report.json annotations.yaml
```
This generates precision/recall metrics, absolute duration error stats, and a confusion matrix heatmap.

---

## 📊 Example Output

The system produces a highly structured JSON report matching the assignment requirements exactly:

```json
{
  "observation_duration_sec": 1200.0,
  "target_person": {
    "final_state": "lying_in_bed"
  },
  "activity_durations": {
    "lying_in_bed": 702.0,
    "sitting_on_bed": 128.0,
    "walking": 167.0,
    "unknown": 45.0
  },
  "bed_summary": {
    "time_in_bed": 830.0,
    "time_out_of_bed": 370.0,
    "exit_count": 2,
    "return_count": 2,
    "longest_out_of_bed": 241.0
  },
  "safety_decision": "MONITOR",
  "bed_events": [
    {
      "event_type": "BED_EXIT",
      "timestamp_sec": 308.0,
      "confirmed_at_sec": 320.0,
      "previous_state": "sitting_on_bed",
      "current_state": "walking",
      "confidence": 0.9,
      "decision": "MONITOR"
    }
  ],
  "timeline": [
    {
      "start_time_sec": 0.0,
      "end_time_sec": 308.0,
      "state": "lying_in_bed"
    },
    {
      "start_time_sec": 308.0,
      "end_time_sec": 555.0,
      "state": "walking"
    }
  ]
}
```

---

## 📚 Further Reading (Docs)

For deeper details into the system design, consult the `docs/` directory:
- `architecture.md` / `architecture_diagram.md`: Full architectural specification and Mermaid diagrams.
- `alert_rules.md`: Rationale behind the safety rules (`NORMAL`, `MONITOR`, `ALERT`).
- `failure_cases.md`: Analysis of how the system handles difficult edge cases (occlusions, false exits, multiple people).
