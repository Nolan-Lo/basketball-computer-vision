# Basketball Video Analytics

Automated basketball video analysis using computer vision. Processes standard broadcast basketball footage to detect and track players, the basketball, team affiliations, court keypoints, and ball possession — then projects all entities onto a 2D tactical court view.

**Group 14** — Nolan Lo, Matthew Zidell, Mehul Kalsi
DSC 288 Capstone, UC San Diego

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup & Installation](#setup--installation)
4. [Download Models & Input Video](#download-models--input-video)
5. [Running the Pipeline](#running-the-pipeline)
6. [Configuring main.py](#configuring-mainpy)
7. [Using the CLI Directly](#using-the-cli-directly)
8. [Caching & Reprocessing](#caching--reprocessing)
9. [Output](#output)
10. [Project Structure](#project-structure)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline runs six stages on each video:

1. **Player & Ball Detection** — Fine-tuned YOLOv5l6u detects players, referees, ball, and overlays.
2. **Multi-Object Tracking** — ByteTrack assigns persistent track IDs across frames.
3. **Court Keypoint Detection** — Fine-tuned YOLOv8x-Pose localizes 18 court reference points.
4. **Team Assignment** — CLIP (fashion-clip) classifies each player's jersey via text-image similarity.
5. **Ball Possession Detection** — Algorithmic proximity analysis determines which player has the ball.
6. **Visualization & Tactical View** — Draws annotated bounding boxes, court keypoints, and a 2D minimap via homography projection.

---

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager (used instead of pip/venv)
- **Git**
- A CUDA-compatible GPU is recommended but not required (CPU works, just slower)

### Install uv

If you don't have `uv` installed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

---

## Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Nolan-Lo/basketball-computer-vision.git
   cd basketball-computer-vision
   ```

2. **Install all dependencies:**

   ```bash
   uv sync
   ```

   This creates a `.venv/` virtual environment and installs everything from `pyproject.toml` (PyTorch, Ultralytics, OpenCV, Transformers, Supervision, etc.). No manual `pip install` needed.

---

## Download Models & Input Video

The trained model weights and a sample input video are hosted on Google Drive since they are too large for the repository.

**Google Drive link:** https://drive.google.com/drive/u/1/folders/1oBlI9mgDUCCAup_HQz0ozvzkKqaC20HR


Download the following files and place them in the correct locations:

| File | Place in | Description |
|------|----------|-------------|
| `Basketball-Players-17.pt` | `models/Basketball-Players-17.pt` | Fine-tuned YOLOv5l6u for player/ball detection |
| `court-keypoints.pt` | `models/court-keypoints.pt` | Fine-tuned YOLOv8x-Pose for court keypoint detection |
| `video_1.mp4` (or other input videos) | `input_videos/video_1.mp4` | Sample broadcast basketball video |

After downloading, your directory should look like:

```
basketball-video-analytics/
├── models/
│   ├── Basketball-Players-17.pt   ← downloaded
│   └── court-keypoints.pt         ← downloaded
├── input_videos/
│   └── video_1.mp4                ← downloaded
├── images/
│   └── basketball_court.png       (already in repo)
└── ...
```

---

## Running the Pipeline

The simplest way to run the full pipeline:

```bash
uv run python main.py
```

This uses the default settings defined in `main.py`. The output video will be written to the path specified in that file (by default, `runs/pipeline_output/`).

---

## Configuring main.py

Open `main.py` and edit the variables near the top of the `main()` function to control the pipeline:

```python
# --- Change these to match your setup ---
input_video = "input_videos/video_1.mp4"          # Path to your input video
output_video = "runs/pipeline_output/output.mp4"   # Where to save the annotated video
```

To change **team jersey descriptions** (controls how CLIP classifies players), find the `BasketballAnalysisPipeline(...)` call and edit:

```python
pipeline = BasketballAnalysisPipeline(
    player_model_path=player_model,
    court_model_path=court_model,
    team_1_description="white jersey",    # ← describe Team 1's jersey
    team_2_description="dark jersey",     # ← describe Team 2's jersey
    court_image_path=court_image
)
```

---

## Using the CLI Directly

We suggest using main.py for guaranteed reproducability, but if desired, for more control, run the pipeline module directly with command-line

```bash
uv run python -m src.pipeline \
    --input inp
    --output runs/pipeline_output/output.mp4 \
    --team1 "white jersey" \
    --team2 "dark jersey"
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | *(required)* | Path to input video file |
| `--output`, `-o` | *(required)* | Path for output video file |
| `--player-model` | `models/Basketball-Players-17.pt` | Path to player/ball detection model |
| `--court-model` | `models/court-keypoints.pt` | Path to court keypoint model |
| `--team1` | `"white jersey"` | Text description of Team 1's jersey |
| `--team2` | `"dark jersey"` | Text description of Team 2's jersey |
| `--cache-dir` | `runs/cache/` | Directory for caching intermediate results |
| `--court-image` | `images/basketball_court.png` | Court background image for tactical minimap |
| `--no-info` | *(off)* | Hide the info panel overlay on the output video |

---

## Caching & Reprocess

The pipeline caches intermediate results (player tracks, ball tracks, team assignments) as `.pkl` files in `runs/cache/`. 

Cache files are named by video f

runs/cache/
├── video_1
├── video_1_ball_tracks.pkl
└── video_1_teams.pkl

### When to clear the cache

Delete the cache files if you want to **fully reprocess** a video from scratch — for example, af

```bash
# Delete all cache files for a specific video
rm runs/cache/video_1_*.pkl

# Or delete the entire cache directory
rm -rf runs/cache/
```

The pipeline will regenerate the cache on the next run.

**Note:** If you only changed the team descriptions (e.g., `--team1` / `--team2`), you only need to delete the `*_teams.pkl` file for that video. Player and ball track caches can be kept.

---

## Output

The pipeline produces an annotated video with:

- ⬜ **White bounding boxes** — Team 1 players
- 🟧 **Orange bounding boxes** — Team 2 players
- 🟨 **Yellow bounding box** — Basketball
- 🟪 **Magenta bounding box + "[BALL]" label** — Current ball carrier
- 🟡 **Yellow circles** — Detected court keypoints
- 🗺️ **Tactical minimap** — 2D court projection of player positions (bottom-left)
- 📊 **Info panel** — Frame number, possession status, and stats (top-right)

Output videos are saved to the path specified by `--output` or the `output_vid

---

## Project Structure

```
basketball-video-analytics/
├── main.py    
├── pyproject.toml                     # Dependenci
├── src/                               # Core source code
│   ├── __init__.py     

│   ├── trackers/
│
│   │   └── ball_tracker.py            # Ball detection, outlier removal, interpolation
│   ├── team_assigner.py               # CLIP-based team classification
│   ├── ball_acquisition_detector.py   # Algorithmic ball possession detection
│   ├── homography.py                  # Keypoint validation and homography estimation
│   ├── tactical_view_converter.py     # 2D court projection
│   ├── drawers/                       # Visualization modules
│   │   ├── player_tracks_drawer.py
│   │   ├── ball_tracks_drawer.py
│   │   ├── court_keypoints_drawer.py
│   │   ├── tactical_view_drawer.py
│   │   ├── team_ball_control_drawer.py
│   │   └── frame_number_drawer.py

│   ├── bbox_utils.py      
│   └── utils.py                       # Pickle caching utilities
├── models/                            # Trained model weights (download separately)
│   ├── Basketball-Players-17.pt
│   ├── court-keypoints.pt
│   └── pretrained/                    # Base 
├── input_videos/                      # Place input videos here
├── images/
│   └── basketball_court.png      
├── runs/                              # Outputs
│   ├── pipeline_output/               # Output videos
│   └── cache/                         # Cached tracks and team assignments (.pkl)
├── notebooks/                         # Jupyter notebooks (exploration & training)
│   ├── 01-data-exploration.ipynb
│   ├── 02-player-ball-detection.ipynb
│   ├── 03-court-keypoint-detection.ipynb
│   └── pipeline_runner.ipynb
├── data/                              # Training datasets
│   ├── Basketball-Playe
│   └── court-keypoints/              # Court keypoint dataset (YOLOv8-Pose format)
└── docs/                              # Documentation and report
    ├── final_report.md
    ├── PIPELINE.md
    ├── QUICKSTART.md
    └── TEAM_ASSIGNMENT.md
```

---

## Troubleshooting

**"Input video not found" error**
- Ensure your video is placed in `input_videos/` and the filename in `main.py` matches exactly.

**"Model not found" error**
- Download the model weights from the Google Drive link above and place them in `models/`.

**Pipeline is very slow**
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- CPU-only processing is 5-10x slower than GPU.
- Caching is enabled by default — the second run on the same video 

**Poor team classification**
- Use more descriptive jersey text (color, trim, numbers).
- Delete

**CLIP model downloading on fi


**Out of memory**
- Close 
- Process a shorter video clip for testing.

---

## Acknowledgments

- [Ultralytics](https://github.com/ult
- [Roboflow Supervision](https://github.com/roboflow/supervision) — ByteTrack implementation
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — CLIP model
- [OpenCV

