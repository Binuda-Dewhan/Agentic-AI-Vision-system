# Agentic AI Vision System - Sleep Monitor

This project implements an Agentic AI + Vision system that analyzes continuous indoor video of an elderly person to determine their activity, detect bed exits/returns, and output an activity timeline with a safety decision.

## Overview

The system uses a combination of deterministic spatial/temporal rules and an Agentic Vision-Language Model (VLM) fallback to analyze human posture and spatial relationships to a configured "bed region."

Key capabilities:
- **Activity State Recognition:** Classifies states such as `LYING_IN_BED`, `SITTING_ON_BED`, `STANDING`, `WALKING`, and `UNKNOWN`.
- **Bed Exit and Return:** Uses hysteresis to filter out false exits/returns and only triggers when a person physically moves away from or returns to the bed.
- **Disappearance Inference (Deterministic Fallback):** If the target person completely disappears from the camera frame (0 detections), the system infers they are `WALKING` and `OUT_OF_BED` (if they were previously upright/near the edge). This handles camera FOV exits robustly without AI.
- **Agentic Fallback (Optional):** Uses LangGraph and Gemini 3.8 Flash to investigate ambiguous segments (e.g. `UNKNOWN` states) by looking at temporal context. **Note:** The VLM is strictly an optional ambiguity-investigation path. If the API is unavailable, rate-limited, or disabled, the pipeline gracefully bypasses the agent and relies on the deterministic state tracking.

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure API Keys:**
   The VLM requires a Gemini API key. Set it in your environment:
   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```
3. **Configure Bed Region:**
   To configure the bed polygon for a new video, run the helper script:
   ```bash
   python get_bed_polygon.py <path_to_video>
   ```
   Click on the image to select the 4 corners of the bed. Right-click to undo, and press `q` to quit and copy the polygon to `config.yaml`.

## Running the Pipeline

To analyze a video and generate a full timeline report:

```bash
python -m sleep_monitor.cli analyze "video_1.mp4" --config "config.yaml"
```

To see visual debugging (shows bounding boxes, poses, and polygons on the frames as they are processed):
```bash
python -m sleep_monitor.cli analyze "video_1.mp4" --config "config.yaml" --visualize
```

The system will output a final report to `output/video_1_report.json` containing the durations, timeline, bed events, and safety decisions.

## Architecture Diagram

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

## System Components

1. **VideoIngester:** Extracts frames at a configured sample rate and groups them into temporal segments.
2. **Perception Layer:** 
   - **YOLOv8n:** Detects persons in the frame.
   - **TargetSelector (ByteTrack):** Locks onto a single primary target to ignore caregivers/bystanders.
   - **YOLOv8n-pose:** Extracts skeletal keypoints.
   - **SpatialAnalyzer:** Evaluates bounding boxes against the configured bed polygon.
3. **StateEngine:** Uses a sliding window and hysteresis to determine the deterministic `ActivityState`. Implements logic to track targets that leave the camera's FOV.
4. **InvestigationAgent (VLM):** An agentic fallback using LangGraph. If the `StateEngine` outputs `UNKNOWN`, it sends frames and temporal context to Gemini to resolve ambiguity.
5. **BedEventEngine:** Tracks transitions into and out of the bed, triggering `BED_EXIT` and `RETURN_TO_BED` events only after a configurable hysteresis threshold.
6. **Safety & Timeline Engines:** Generates the final timeline, formats activity durations, and outputs safety alerts based on predefined rules.
