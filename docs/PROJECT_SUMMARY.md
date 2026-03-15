# Basketball Computer Vision — Project Summary

## What It Does

An end-to-end basketball video analysis pipeline that ingests broadcast footage and outputs an annotated video with player tracking, team classification, ball possession detection, and a tactical bird's-eye court view.

---

## Pipeline: Detect → Track → Classify → Project → Visualize

```
Input Video
  → Player Tracking (YOLO + ByteTrack)
  → Ball Tracking (YOLO + outlier filter + interpolation)
  → Court Keypoint Detection (YOLO-Pose, 18 keypoints)
  → Team Assignment (CLIP vision-language model)
  → Ball Possession Detection (proximity + containment heuristics)
  → Tactical View Projection (homography → 2D court)
  → Visualization (multiple specialized drawer classes)
  → Output Annotated Video
```

---

## File & Folder Descriptions

### Root
- **main.py** — Minimal entry point; wires inputs/outputs and calls the pipeline with defaults.
- **pyproject.toml** — Python 3.12+ project config; key deps: `ultralytics`, `transformers`, `supervision`, `opencv`, `torch`, `pandas`.

---

### src/ — Core application code

| File | Purpose |
|------|---------|
| `src/pipeline.py` | **Main orchestrator** (~600 lines). `BasketballAnalysisPipeline` sequences all steps, handles caching, and drives frame-by-frame processing. |
| `src/team_assigner.py` | Uses a **CLIP vision-language model** (fashion-clip) to classify each player as team 1 or 2 based on jersey appearance. Caches per-player assignments; resets every 50 frames to handle substitutions. |
| `src/ball_acquisition_detector.py` | Determines ball possession using **containment ratio** (ball bbox overlap with player bbox) and **proximity distance**. Requires 11 consecutive confirmatory frames before assigning possession. |
| `src/tactical_view_converter.py` | Maps video-frame player positions to a **standardized NBA 2D court** (94×50 ft) using homography. Includes a 4-step keypoint validation pipeline (mirror-remap, proportion check, ordering check, jump filter). Reuses last good homography for up to 10 frames when detection is poor. |
| `src/homography.py` | Thin wrapper around `cv2.findHomography` + `cv2.perspectiveTransform`. |
| `src/video_utils.py` | Video I/O (`read_video`, `save_video`), OpenCV drawing helpers, and optional stats info panel overlay. |
| `src/bbox_utils.py` | Bounding box math: center, foot position, width/height, Euclidean distance. |
| `src/utils.py` | Pickle-based caching helpers (`read_stub`, `save_stub`). |

---

### src/trackers/

| File | Purpose |
|------|---------|
| `player_tracker.py` | YOLOv8 detection + **ByteTrack** multi-object tracking. Processes in batches of 20 frames. Returns persistent `{track_id: {"bbox": ...}}` dicts per frame. |
| `ball_tracker.py` | YOLOv8 detection (highest-confidence ball per frame). Removes outlier detections (>25px/frame jumps). Fills gaps with **pandas linear interpolation**. |

---

### src/drawers/ — One class per visual concern, all implementing `draw(frames, ...)`

| File | Purpose |
|------|---------|
| `player_tracks_drawer.py` | Team-colored ellipses at player feet; red triangle above ball carrier. |
| `ball_tracks_drawer.py` | Green triangle pointer above ball position. |
| `court_keypoints_drawer.py` | Red circles + numeric labels at each detected court keypoint (uses supervision library). |
| `tactical_view_drawer.py` | Semi-transparent (60%) bird's-eye court mini-map overlaid on the frame, with team-colored player dots. |
| `frame_number_drawer.py` | Frame counter in top-left corner. |
| `drawers/utils.py` | Low-level drawing primitives (ellipse, triangle). |

---

### models/

| File | Purpose |
|------|---------|
| `Basketball-Players-17.pt` | YOLOv8 detector trained on 7 classes: Ball, Clock, Hoop, Overlay, Player, Ref, Scoreboard. |
| `court-keypoints.pt` | YOLOv8-Pose model detecting 18 court keypoints (baselines, lane boundaries, free-throw elbows, etc.). |
| CLIP model | Auto-downloaded from HuggingFace on first run (~1 GB). |

---

### Other Directories

| Directory | Purpose |
|-----------|---------|
| `data/` | Training datasets in YOLO format (Roboflow-sourced), with train/valid/test splits. |
| `runs/` | Output artifacts: annotated videos, pickle caches for player tracks, ball tracks, and team assignments. |
| `images/` | `basketball_court.png` — NBA court diagram used as the tactical view background. |
| `docs/` | Pipeline walkthrough, quickstart guide, team assignment explanation, and restructuring notes. |
| `notebooks/` | Jupyter notebooks for experimentation. |

---

## Key Design Decisions

1. **Modular separation** — Trackers, drawers, and analysis components are fully independent classes composed by the pipeline. Easy to swap or extend any piece.
2. **Pickle caching** — Expensive operations (tracking, team assignment) are cached so you can iterate on visualization without re-running inference.
3. **Defensive keypoint validation** — 4-step filtering on court keypoints guards against homography blowing up from noisy detections.
4. **CLIP for team assignment** — No per-game retraining required; jersey descriptions are plain text (e.g., `"white jersey"`, `"dark jersey"`).
5. **Possession hysteresis** — 11-frame confirmation window prevents noisy ball-possession flickering.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Object detection | YOLOv8 (Ultralytics) |
| Multi-object tracking | ByteTrack (via supervision library) |
| Team classification | CLIP / fashion-clip (HuggingFace Transformers) |
| Court projection | OpenCV homography |
| Ball gap-filling | Pandas linear interpolation |
| Video I/O | OpenCV |
| Deep learning backend | PyTorch |
