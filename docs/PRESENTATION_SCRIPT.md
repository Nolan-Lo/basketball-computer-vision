# Presentation Script — Basketball Video Analytics

> **Total time: ~10 minutes**
> Format: Code walkthrough (6 min) → Live demo (3 min) → Wrap-up (1 min)

---

## PRE-PRESENTATION CHECKLIST

Before you start, have these ready:

- [ ] VS Code open with the project root (`capstone/`)
- [ ] Terminal open, `cd`'d into `capstone/`
- [ ] File explorer sidebar visible (collapsed to top-level folders)
- [ ] `main.py` open in the editor
- [ ] Verify input video exists: `ls input_videos/video_1.mp4`
- [ ] Verify cache files exist: `ls runs/cache/` (you should see `.pkl` files)
- [ ] Optionally have a finished output video ready to show at the end in case the demo is slow: `runs/pipeline_output/video1_v7_40pct.mp4`

---

## PART 1 — INTRODUCTION (1 minute)

> **What to say:**

"This project is an automated basketball video analysis system. The problem it solves is simple: traditional basketball stats like box scores only tell you points, rebounds, assists — they miss everything spatial.  Where are players standing? Who's moving off-ball? What does the court look like at any given moment?

Manually annotating video frame-by-frame isn't realistic, so this system uses computer vision to automatically process broadcast basketball video. It detects and tracks players, finds the ball, figures out which team each player is on, determines who has possession, and maps everything onto a 2D court diagram — all from regular TV footage, no special cameras needed.

Let me walk you through how it's organized and how each piece works."

---

## PART 2 — PROJECT STRUCTURE (1.5 minutes)

> **Action:** Show the VS Code file explorer sidebar. Expand top-level folders one by one as you talk.

> **What to say:**

"The project is cleanly organized into a few key areas.

**`data/`** — This holds our training datasets. We have two: one for player and ball detection (`Basketball-Players-17`), and one for court keypoints (`court-keypoints`). Both are labeled image datasets in YOLO format — images plus text files with bounding box coordinates.

**`models/`** — These are our trained model weights. `Basketball-Players-17.pt` is our fine-tuned YOLO model that detects players and the ball. `court-keypoints.pt` is a YOLO Pose model that detects fixed points on the court like corners and free-throw lines. Under `pretrained/` we have the base models we started from before fine-tuning.

**`input_videos/`** — Raw basketball broadcast clips we process.

**`runs/`** — All output goes here: training run logs, output videos, and importantly a `cache/` folder with pickle files that save intermediate results so we don't have to re-run expensive detection every time.

**`notebooks/`** — Jupyter notebooks for data exploration, model training experiments, and a pipeline runner.

**`src/`** — This is the core source code. Let me dive into this."

> **Action:** Expand the `src/` folder in the sidebar.

"Inside `src/` we have a clean modular layout:
- **`pipeline.py`** — The main orchestrator that ties everything together
- **`trackers/`** — Player tracker and ball tracker modules
- **`team_assigner.py`** — Figures out which team each player is on
- **`ball_acquisition_detector.py`** — Determines who has the ball
- **`tactical_view_converter.py` and `homography.py`** — Maps video positions to a flat 2D court
- **`drawers/`** — Six separate drawer classes, each responsible for drawing one thing on the output video
- **`video_utils.py`, `bbox_utils.py`, `utils.py`** — Shared helpers

The whole `src/` package is about 2,800 lines of Python total. Each file has one clear responsibility."

---

## PART 3 — PIPELINE WALKTHROUGH (3.5 minutes)

### 3A — Entry Point: `main.py` (~30 sec)

> **Action:** Open `main.py` (should already be open). Scroll through it.

> **What to say:**

"Everything starts in `main.py`. It's intentionally simple — about 80 lines. It defines file paths for the input video, the two models, and the court image. It checks that everything exists, then creates a `BasketballAnalysisPipeline` object and calls `process_video()`. That's it. All the real logic lives in the pipeline class."

---

### 3B — The Pipeline: `pipeline.py` (~1.5 min)

> **Action:** Open `src/pipeline.py`. Show the class docstring and `__init__`.

> **What to say:**

"The pipeline class is the brain of the system. In `__init__`, it loads all the components: a player tracker, a ball tracker, the court keypoint model, a team assigner, the ball possession detector, and optionally a tactical view converter. It also creates all six drawer objects for visualization.

The key method is `process_video()`."

> **Action:** Scroll to `process_video()` (around line 430). Walk through the steps.

"Here's where the seven-stage pipeline runs in order:

1. **Read the video** — loads all frames into memory using OpenCV
2. **Track players** — runs YOLO detection on every frame in batches of 20, then feeds results through ByteTrack to assign persistent IDs so player #5 stays player #5 across frames
3. **Track the ball** — same YOLO model, but picks only the highest-confidence ball detection per frame, removes outlier detections that jump impossibly far between frames, then interpolates gaps where the ball was hidden
4. **Detect court keypoints** — a separate YOLO Pose model finds 18 fixed points on the court like corners, free-throw lines, and sideline marks
5. **Assign teams** — this is the clever one: it crops each player's bounding box from the frame and feeds it to a CLIP model — that's a vision-language model — along with text descriptions like 'white jersey' and 'dark jersey', and CLIP tells us which description matches better
6. **Detect ball possession** — purely algorithmic, no ML here. It measures how much the ball's bounding box overlaps with each player's box, plus the distance from the ball to key points on the player. A player has to 'hold' the ball for several consecutive frames before we confirm possession
7. **Compute tactical view** — uses homography, which is a math transformation, to map the detected court keypoints from the camera's perspective to a flat overhead court diagram. Then it projects player positions through that same transformation."

> **Action:** Briefly scroll to the `visualize_detections()` method.

"Finally, everything gets drawn onto the frames — keypoints, player ellipses with team colors, ball marker, a tactical mini-map overlay, frame counter, and an info panel. Each of these is handled by its own drawer class, which keeps the drawing code separate from the detection logic."

---

### 3C — Key Modules (quick highlights) (~1 min)

> **Action:** Quickly open each file, show just the class and key method (don't read code line by line).

**Player Tracker** (`src/trackers/player_tracker.py`)
"The player tracker wraps YOLO and ByteTrack from the `supervision` library. YOLO detects players frame-by-frame, ByteTrack links those detections across frames to maintain consistent track IDs. It also supports caching — if a pickle cache file exists, it loads that instead of re-running detection."

**Team Assigner** (`src/team_assigner.py`)
"Uses `fashion-clip`, a CLIP model fine-tuned for clothing. We crop each player, send the image plus our team descriptions to CLIP, and whichever description scores higher becomes that player's team. Once assigned, it's cached per player ID so we don't re-classify the same player."

**Ball Acquisition Detector** (`src/ball_acquisition_detector.py`)
"This is algorithmic — no model. It checks two things per frame: how much of the ball box is inside a player's box (containment ratio), and the minimum pixel distance from the ball center to multiple points around the player's body. It requires a player to pass these checks for several consecutive frames before confirming possession. This avoids flickering."

**Tactical View Converter** (`src/tactical_view_converter.py`)
"This maps video coordinates to a 2D court. It defines 18 reference points that correspond to known positions on an NBA court — corners, free-throw lines, sideline marks. Using OpenCV's homography function, it computes a perspective transformation matrix from the detected keypoints to these reference points, then projects every player's foot position through that matrix to get their position on the mini-court."

---

### 3D — Caching System (~30 sec)

> **Action:** Open `src/utils.py`. Show `read_stub` and `save_stub`.

> **What to say:**

"One important design choice: caching. The detection steps — player tracking, ball tracking, and team assignment — are the most expensive because they run neural networks on every frame. So after running them once, results are saved as pickle files in `runs/cache/`. On the next run, the pipeline checks if a cache file exists and loads it instantly instead of re-running detection. This is critical for development iteration — you only pay the cost once per video."

---

## PART 4 — LIVE DEMO (3 minutes)

> **Action:** Switch to the terminal.

> **What to say:**

"Let me run the pipeline live so you can see what happens. I'll run `main.py` on a short basketball clip."

> **Action:** Run the command:
```bash
uv run python main.py
```

> **What to say as output appears:**

**When you see the header banner:**
"It prints a nice header, then checks that all required files exist — the input video, both models, and the court image."

**When you see 'Initializing Basketball Analysis Pipeline...':**
"Now it's loading the models. The player/ball detection model, court keypoint model, the CLIP model for team assignment, and setting up the tactical view converter."

**When you see 'Reading video...' and the frame count:**
"It loaded all frames from the video into memory. You can see the resolution and frame rate."

**When you see 'Step 1/5: Tracking players...':**
"Now — and this is the caching part I mentioned — watch what happens. It's going to check if `runs/cache/video_1_player_tracks.pkl` exists."

- **If cache EXISTS (likely scenario):** "See that? It loaded the cached player tracks instantly instead of running YOLO on every frame. Same for ball tracks and team assignments. This saves minutes of processing."
- **If cache does NOT exist:** "It's running YOLO detection on every frame in batches. This is the slow part. On a GPU it's fast, on CPU it takes a bit. The results will be cached for next time."

**When you see 'Step 3/5: Detecting court keypoints...':**
"Court keypoints are NOT cached because they're fast enough to re-run — it's just one pose estimation per frame. You can see it processing every 30th frame in the progress output."

**When you see 'Step 5/5: Detecting ball possession...':**
"This is the algorithmic step — no neural net, just geometry. It runs almost instantly."

**When you see 'Computing tactical-view positions...':**
"Now it's computing homography matrices and projecting player positions to the flat court for each frame."

**When you see 'Visualizing detections...':**
"Now all six drawers are running: keypoints, player ellipses, ball marker, tactical mini-map, frame counter, and info panel — all layered onto each frame."

**When you see 'Processing Complete!':**
"Done. Look at the summary stats — average players detected per frame, ball detection rate, possession detection rate. The output video is saved and ready to view."

> **Action:** If time permits, open the output video briefly:
```bash
xdg-open runs/pipeline_output/video1_v7_40pct.mp4
```

> **What to say:**
"Here's the output. You can see team-colored ellipses under each player with their track IDs, the ball highlighted, the current ball carrier marked in red, court keypoints drawn as dots, and in the top-left corner, the tactical mini-map showing the bird's-eye view of player positions. The info panel in the top-right shows live stats per frame."

---

## PART 5 — WRAP-UP (30 seconds)

> **What to say:**

"To summarize: this is a software-only system that takes regular broadcast basketball video and automatically extracts spatial information that traditional box scores miss. It uses YOLO for detection, ByteTrack for tracking, CLIP for team classification, custom algorithms for possession detection, and homography for court mapping. Everything is modular — each component has one job, results are cached for fast iteration, and the pipeline orchestrates them all in sequence. This gives us a foundation for building more advanced basketball analytics on top — things like player movement patterns, defensive positioning analysis, or shot quality metrics."

---

## TIMING GUIDE

| Section | Duration | Cumulative |
|---------|----------|------------|
| Introduction | 1:00 | 1:00 |
| Project Structure | 1:30 | 2:30 |
| Pipeline Walkthrough | 3:30 | 6:00 |
| Live Demo | 3:00 | 9:00 |
| Wrap-up | 0:30 | 9:30 |
| **Buffer** | **0:30** | **10:00** |

---

## TIPS

- **Don't read code line by line.** Point at the screen and describe what a method does in plain English. The audience needs to understand the *flow*, not every line.
- **Use the terminal printouts as your guide during the demo.** Each "Step X/5" message is a natural talking point.
- **If the demo runs slow** (no GPU / no cache), narrate what *would* happen and show a pre-rendered output video from `runs/pipeline_output/`.
- **If someone asks about model training,** point to `notebooks/` and the `data/` folder — "We fine-tuned the base YOLO model on labeled basketball data, which is in the notebooks."
- **If someone asks about accuracy,** mention mAP from training runs and point to `runs/pose/train/` for training logs.
- **Practice once to get timing.** The pipeline walkthrough is the densest part — keep it moving.
