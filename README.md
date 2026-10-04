# Agentic AI Vision System: Elderly Sleep & Safety Monitor

## 📖 Overview
This project implements a hybrid Agentic AI and Computer Vision system designed to continuously monitor indoor video of an elderly person. Its primary purpose is to recognize their physical activity, detect critical **Bed Exit** and **Return to Bed** events, and generate an automated safety timeline.

The system was built with robustness and privacy in mind, utilizing a combination of high-speed deterministic algorithms (YOLOv8 + spatial polygons) and a secondary **Agentic Vision-Language Model (VLM) fallback** to resolve highly ambiguous situations.

---

## ✨ Core Features
- **Temporal Activity State Tracking:** Accurately classifies ongoing states including `LYING_IN_BED`, `SITTING_ON_BED`, `STANDING`, `WALKING`, and `UNKNOWN`.
- **Robust Bed Event Detection:** Uses configurable temporal hysteresis (e.g., 10 seconds) to completely eliminate false positives. Someone merely sitting up or rolling over will *not* trigger an exit alarm.
- **Disappearance Inference (Deterministic):** If a person completely leaves the camera's field of view, the system utilizes their last known state to intelligently infer they are `WALKING` and `OUT_OF_BED` without crashing or relying on AI.
- **Agentic VLM Fallback (Optional):** When the deterministic pipeline encounters heavy occlusion or highly ambiguous poses (`UNKNOWN` states), it leverages LangGraph and Google's Gemini 3.8 Flash to interpret the scene using temporal context. *(Note: If the API is rate-limited or disabled, the pipeline gracefully bypasses the agent and relies on local deterministic logic).*
- **Safety Engine:** Evaluates the timeline history against configurable safety rules to output a final decision: `NORMAL`, `MONITOR`, or `ALERT`.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[Video Input] --> B[Video Ingester]
    B -->|Frames (2 FPS)| C[Perception Layer]
    
    subgraph Perception Layer
        C1[YOLOv8n Detector]
        C2[ByteTrack Target Selector]
        C3[YOLOv8n-Pose Estimator]
        C4[Spatial Analyzer]
        C1 --> C2 --> C3 --> C4
    end
    
    C -->|Frame Observations| D[Observation Builder]
    D -->|Segment Observation| E[State Engine]
    
    E -->|Current State| F{Confidence / Ambiguity Check}
    F -->|Low Confidence / UNKNOWN| G[LangGraph VLM Agent]
    G -.->|Gemini 3.8 Flash| H[(VLM Inference)]
    H -.-> G
    G -->|Confirmed State| I[Bed Event Engine]
    
    F -->|High Confidence| I
    
    I -->|Context & Events| J[Timeline Engine]
    J -->|Timeline| K[Safety Engine]
    K -->|NORMAL / MONITOR / ALERT| L[Final JSON Report]
```

### System Components Explained
1. **VideoIngester:** Extracts frames at a low frame rate (e.g., 2 FPS) to save compute, chunking them into temporal segments (e.g., 2 seconds).
2. **Perception Layer:** 
   - **YOLOv8n:** Detects persons in the frame.
   - **TargetSelector (ByteTrack):** Locks onto a single primary target to ignore caregivers/bystanders.
   - **YOLOv8n-pose:** Extracts skeletal keypoints.
   - **SpatialAnalyzer:** Evaluates the bounding boxes against a user-defined bed polygon.
3. **StateEngine:** Uses a sliding window and hysteresis to determine the `ActivityState`. Implements deterministic logic to track targets that leave the camera's FOV.
4. **InvestigationAgent (VLM):** An agentic fallback using LangGraph. If the `StateEngine` outputs `UNKNOWN`, it sends frames and context to Gemini to resolve the ambiguity.
5. **BedEventEngine:** Tracks transitions into and out of the bed, triggering `BED_EXIT` and `RETURN_TO_BED` events only after a hysteresis threshold is met.
6. **Timeline & Safety Engines:** Aggregates states, formats activity durations, and outputs a final JSON report with safety alerts based on predefined rules.

---

## 🚀 Installation & Setup

1. **Clone the repository and enter the directory:**
   ```bash
   git clone <your-repo-url>
   cd "Sleeping posture detection system"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys:**
   The VLM fallback requires a Google Gemini API key. Create a `.env` file in the root directory and add your key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

---

## ⚙️ Configuration & Usage

### 1. Define the Bed Region (Important for new videos)
Before analyzing a new video with a different camera angle, you must configure the spatial boundaries of the bed. Run the helper script:
```bash
python get_bed_polygon.py <path_to_video>
```
*Click the 4 corners of the bed on the image popup. Press `q` to quit. Copy the resulting coordinates into `config.yaml`.*

### 2. Run the Analysis Pipeline
To analyze a video and generate a full timeline report, use the CLI:

```bash
python -m sleep_monitor.cli analyze "video_1.mp4" --config "config.yaml"
```

**Want to see the system working live?** Add the `--visualize` flag to open a debug window showing bounding boxes, pose keypoints, and the bed polygon in real-time:
```bash
python -m sleep_monitor.cli analyze "video_1.mp4" --config "config.yaml" --visualize
```

### 3. Review the Output
Once the pipeline finishes, it will generate a comprehensive JSON report located at `output/video_1_report.json`. This report includes:
- Total durations for all recognized activities (Lying, Sitting, Walking).
- Exact timestamps for `BED_EXIT` and `RETURN_TO_BED` events.
- An aggregated block-by-block timeline of the person's state.
- The final `safety_decision` (e.g., `NORMAL` or `ALERT`).
