================================================================================
  Basketball Video Analytics — Reproduction Guide
  Group 14: Nolan Lo, Matthew Zidell, Mehul Kalsi
  DSC 288 Capstone, UC San Diego
================================================================================

This document explains how to set up and run our basketball video analysis
pipeline from scratch.

--------------------------------------------------------------------------------
1. PREREQUISITES
--------------------------------------------------------------------------------

  - Python 3.12 or later
  - Git
  - uv (fast Python package manager): https://docs.astral.sh/uv/
  - A CUDA-compatible GPU is recommended but not required

  To install uv:

    curl -LsSf https://astral.sh/uv/install.sh | sh

  Or:

    pip install uv

--------------------------------------------------------------------------------
2. CLONE THE REPOSITORY
--------------------------------------------------------------------------------

    git clone https://github.com/Nolan-Lo/basketball-computer-vision.git
    cd basketball-computer-vision

--------------------------------------------------------------------------------
3. INSTALL DEPENDENCIES
--------------------------------------------------------------------------------

    uv sync

  This creates a .venv/ virtual environment and installs all dependencies from
  pyproject.toml (PyTorch, Ultralytics, OpenCV, Transformers, Supervision, etc.).
  No manual pip install is needed.

--------------------------------------------------------------------------------
4. DOWNLOAD MODELS AND INPUT VIDEO
--------------------------------------------------------------------------------

  The trained model weights and sample input videos are too large for the
  repository and are hosted on Google Drive:

    https://drive.google.com/drive/u/1/folders/1oBlI9mgDUCCAup_HQz0ozvzkKqaC20HR

  (Replace the link above with the actual shared folder URL.)

  Download these files and place them as follows:

    Basketball-Players-17.pt  -->  models/Basketball-Players-17.pt
    court-keypoints.pt        -->  models/court-keypoints.pt
    video_1.mp4 (or others)   -->  input_videos/video_1.mp4

  The court background image (images/basketball_court.png) is already included
  in the repository.

--------------------------------------------------------------------------------
5. RUN THE PIPELINE
--------------------------------------------------------------------------------

  The simplest way:

    uv run python main.py

  This uses the default settings in main.py. The annotated output video will be
  saved to the path specified in that file (runs/pipeline_output/ by default).

--------------------------------------------------------------------------------
6. CONFIGURE main.py
--------------------------------------------------------------------------------

  Open main.py in a text editor to change the following settings:

  INPUT / OUTPUT PATHS:
  Near the top of the main() function, edit these variables:

    input_video = "input_videos/video_1.mp4"
    output_video = "runs/pipeline_output/output.mp4"

  TEAM JERSEY DESCRIPTIONS:
  In the BasketballAnalysisPipeline(...) constructor call, edit:

    team_1_description="white jersey"
    team_2_description="dark jersey"

  More specific descriptions improve accuracy. Examples:
    "white jersey with gold numbers"
    "navy blue jersey with white trim"

--------------------------------------------------------------------------------
7. USE THE CLI FOR MORE CONTROL
--------------------------------------------------------------------------------

  Instead of editing main.py, you can pass all options as command-line arguments 
  (though we suggest using main.py for guaranteed reproducability):

    uv run python -m src.pipeline \
        --input input_videos/video_1.mp4 \
        --output runs/pipeline_output/output.mp4 \
        --team1 "white jersey" \
        --team2 "dark jersey"

  All available flags:

    --input, -i       (required)  Path to input video file
    --output, -o      (required)  Path for output video file
    --player-model    (default: models/Basketball-Players-17.pt)
    --court-model     (default: models/court-keypoints.pt)
    --team1           (default: "white jersey")   Team 1 jersey description
    --team2           (default: "dark jersey")    Team 2 jersey description
    --cache-dir       (default: runs/cache)       Cache directory
    --court-image     (default: images/basketball_court.png)
    --no-info         Hide the info panel on the output video

--------------------------------------------------------------------------------
8. CACHING AND REPROCESSING
--------------------------------------------------------------------------------

  The pipeline caches player tracks, ball tracks, and team assignments as .pkl
  files in runs/cache/. Subsequent runs on the same video load from cache,
  which is much faster.

  Cache files are named by video:

    runs/cache/video_1_player_tracks.pkl
    runs/cache/video_1_ball_tracks.pkl
    runs/cache/video_1_teams.pkl

  TO FULLY REPROCESS A VIDEO (e.g., after retraining a model):

    rm runs/cache/video_1_*.pkl

  Or delete the entire cache:

    rm -rf runs/cache/

  If you only changed team descriptions, you only need to delete the teams
  cache file:

    rm runs/cache/video_1_teams.pkl

  Player and ball track caches can be kept.

--------------------------------------------------------------------------------
9. OUTPUT
--------------------------------------------------------------------------------

  The pipeline produces an annotated video with:

    - White bounding boxes       Team 1 players
    - Orange bounding boxes      Team 2 players
    - Yellow bounding box        Basketball
    - Magenta box + "[BALL]"     Current ball carrier
    - Yellow circles             Detected court keypoints
    - Tactical minimap           2D court projection (bottom-left corner)
    - Info panel (optional)      Frame stats and possession info (top-right)

  Output is saved to the path specified by --output or the output_video
  variable in main.py.

--------------------------------------------------------------------------------
10. PROJECT STRUCTURE
--------------------------------------------------------------------------------

  basketball-video-analytics/
  ├── main.py                            Entry point (edit this)
  ├── pyproject.toml                     Dependencies
  ├── src/                               Core source code
  │   ├── pipeline.py                    Pipeline orchestrator & CLI
  │   ├── trackers/
  │   │   ├── player_tracker.py          YOLO + ByteTrack player tracking
  │   │   └── ball_tracker.py            Ball detection & interpolation
  │   ├── team_assigner.py               CLIP team classification
  │   ├── ball_acquisition_detector.py   Ball possession detection
  │   ├── homography.py                  Court keypoint processing
  │   ├── tactical_view_converter.py     2D court projection
  │   ├── drawers/                       Visualization modules
  │   ├── video_utils.py                 Video I/O
  │   ├── bbox_utils.py                  Bounding box helpers
  │   └── utils.py                       Caching utilities
  ├── models/                            Model weights (download separately)
  ├── input_videos/                      Input videos (download separately)
  ├── images/                            Court image for minimap
  ├── runs/                              Output videos and cache
  ├── notebooks/                         Jupyter notebooks
  ├── data/                              Training datasets
  └── docs/                              Documentation and report

--------------------------------------------------------------------------------
11. TROUBLESHOOTING
--------------------------------------------------------------------------------

  "Input video not found"
    - Check that the filename in main.py matches your file in input_videos/.

  "Model not found"
    - Download model weights from the Google Drive link and place them in
      models/.

  Pipeline runs very slowly
    - Check GPU: python -c "import torch; print(torch.cuda.is_available())"
    - CPU processing is 5-10x slower.
    - The second run on the same video uses cache and is much faster.

  Poor team classification
    - Use more descriptive jersey text.
    - Delete runs/cache/<video>_teams.pkl after changing descriptions.

  CLIP model downloads on first run
    - The fashion-clip model (~1 GB) downloads automatically from HuggingFace.
      This is a one-time download.

  Out of memory
    - Close other GPU applications.
    - Try a shorter video clip.

================================================================================
