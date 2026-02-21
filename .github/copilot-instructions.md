# Basketball Video Analytics - Project Context

## Project Overview

This project develops a computer vision system for automated basketball video analysis that captures spatial and off-ball dynamics not available in traditional box scores and play-by-play data. The system processes broadcast basketball video to identify and track key game entities in real time, enabling context-aware performance evaluation and advanced spatial analytics.

## Problem Statement

Traditional basketball analytics rely on box scores and play-by-play data that fail to capture:
- Spatial positioning of players
- Off-ball movement patterns
- Player interactions and defensive positioning
- Real-time court dynamics

Manual frame-level annotation is infeasible at scale, necessitating automated computer vision approaches.

## Project Objectives

### Primary Goals
1. **Entity Detection**: Identify and track players, teams, basketball, and ball carrier in video frames
2. **Court Mapping**: Detect fixed court anchor points (keypoints) for spatial reference
3. **2D Transformation**: Map detections to a standardized 2D court representation
4. **Real-Time Processing**: Achieve near real-time inference speeds for practical deployment

### Secondary Goals (Time Permitting)
- Downstream basketball analytics (e.g., player gravity, shot quality metrics)
- Advanced spatial analysis features

## Technical Approach

### Model Architecture
- **Primary Framework**: YOLO (You Only Look Once) family of models
- **Rationale**: 
  - Real-time object detection capabilities
  - PyTorch-based implementation
  - Strong community support and established performance
  - Proven effectiveness for similar detection tasks

### Detection Targets
1. Individual players
2. Team identification
3. Basketball location
4. Ball carrier identification
5. Court keypoints (for spatial mapping)

### Alternative Architectures (Evaluation)
- YOLO variants (v5, v8, etc.)
- Transformer-based detectors (DETR, RF-DETR)
- Comparison based on detection performance and inference speed

### Approach Philosophy
- Apply and adapt established models rather than novel architecture development
- Software-only solution using standard broadcast footage (no specialized hardware)
- Focus on practical implementation of proven techniques

## Dataset

### Input Data
- Basketball broadcast video clips
- Individual frames extracted from video as model inputs
- Labeled NBA broadcast datasets with annotated:
  - Player bounding boxes and positions
  - Ball locations
  - Court keypoints

### Data Characteristics
- Publicly available labeled datasets for supervised training
- Additional unlabeled video for testing and evaluation
- Large volume of broadcast footage for model training

### Feature Extraction
- Handled implicitly by deep learning models
- No hand-engineered features required

## Success Criteria

### Quantitative Metrics
1. **Object Detection Performance**:
   - Precision: Ratio of correct detections to all detections
   - Recall: Ratio of correct detections to all ground truth objects
   - Mean Average Precision (mAP): Overall detection quality across classes

2. **Court Mapping Accuracy**:
   - Accuracy of keypoint detection
   - Stability of 2D court transformation across frames

3. **Performance**:
   - Inference speed (target: real-time or near real-time)
   - Processing latency per frame

### Qualitative Criteria
- Reliable localization of all major on-court entities
- Sufficient detection quality to support downstream spatial analytics
- Robust performance across different game conditions (lighting, camera angles, etc.)

## Project Context

### Motivation
Advances in computer vision enable automated analysis of broadcast basketball video, allowing:
- Real-time modeling of player positioning and movement
- Capture of interactions missed by traditional statistics
- Context-aware performance evaluation
- Spatial analytics for strategic insights

### Why Machine Learning?
- Scale: Large volumes of video data require automated processing
- Visual Complexity: High variability in player appearance, occlusions, and camera angles
- Temporal Nature: Need to track entities across frames and game situations
- Data Availability: Large datasets of broadcast footage and labeled examples

### Prior Work
- Multi-camera systems and wearable sensors (hardware-based)
- Commercial optical tracking systems (specialized hardware)
- Academic research using deep neural networks on broadcast video
- Recent work with real-time object detection and transformer-based models

### Project Differentiation
- Software-only approach using standard broadcast footage
- Focus on practical implementation with YOLO models
- Emphasis on real-time spatial tracking and court mapping
- Foundation for original downstream analytics contributions

## Project Structure

### Key Directories
- `data/`: Training and validation datasets
  - `Basketball-Players-17/`: Player and ball detection dataset
  - `court-keypoints/`: Court keypoint detection dataset
- `input_videos/`: Raw video files for processing
- `models/`: Trained model weights
  - `pretrained/`: Base models for fine-tuning
- `notebooks/`: Jupyter notebooks for exploration and development
- `runs/`: Model training runs, output videos, and cache files
- `src/`: Core source code package
  - `src/trackers/`: Player and ball tracking modules (YOLO + ByteTrack)
  - `src/ball_acquisition_detector.py`: Algorithmic ball possession detection
  - `src/team_assigner.py`: CLIP-based team classification
  - `src/pipeline.py`: Main video processing pipeline orchestrating all stages
  - `src/video_utils.py`: Video I/O, drawing, and visualization utilities
  - `src/bbox_utils.py`: Bounding box geometry helpers (distance, center, etc.)
  - `src/utils.py`: Pickle-based caching (stub read/save)

### Key Files
- `main.py`: Primary execution script
- `pyproject.toml`: Project dependencies and configuration (includes `supervision` for ByteTrack)
- `data.yaml`: Dataset configuration files

## Pipeline Architecture

The video analysis pipeline runs in six sequential stages:

1. **Player Tracking** (`PlayerTracker` in `src/trackers/player_tracker.py`)
   - The fine-tuned YOLO model (`Basketball-Players-17.pt`) detects players in batches of 20 frames.
   - Raw detections are fed through ByteTrack (`supervision.ByteTrack`) to assign persistent track IDs across frames.
   - Output: per-frame `{track_id: {"bbox": [x1,y1,x2,y2]}}` dictionaries.

2. **Ball Tracking** (`BallTracker` in `src/trackers/ball_tracker.py`)
   - The same YOLO model detects the basketball, picking the highest-confidence "Ball" detection per frame.
   - Outlier removal filters false positives that jump unreasonable distances between frames.
   - Pandas linear interpolation fills gaps where the ball was occluded.
   - Output: per-frame `{1: {"bbox": [x1,y1,x2,y2]}}` dictionaries.

3. **Court Keypoint Detection** (YOLO Pose model `court-keypoints.pt`)
   - A separate fine-tuned YOLO Pose model detects court boundary/anchor keypoints.
   - Output: per-frame list of `(x, y, confidence)` tuples.

4. **Team Assignment** (`TeamAssigner` in `src/team_assigner.py`)
   - Crops each tracked player's bounding box from the frame.
   - Uses CLIP (`patrickjohncyh/fashion-clip`) to match the cropped image against text descriptions of each team's jersey.
   - Assigns each track ID to Team 1 or Team 2; caches results for reprocessing.

5. **Ball Possession Detection** (`BallAcquisitionDetector` in `src/ball_acquisition_detector.py`)
   - For each frame, computes containment ratio (ball bbox overlap with player bbox) and minimum distance from ball center to multiple key points on each player's bbox.
   - Requires a player to hold the ball for a minimum number of consecutive frames before confirming possession.
   - Output: per-frame player_id with possession, or -1 if unknown.

6. **Visualization** (`video_utils.py`)
   - Draws team-colored bounding boxes, ball box, ball carrier highlight (magenta + "[BALL]" label), court keypoints, and an info panel with frame stats and possession status.

## Development Workflow

1. **Data Exploration**: Analyze datasets and annotation formats
2. **Model Selection**: Evaluate YOLO variants and alternatives
3. **Training**: Fine-tune models on basketball-specific datasets
4. **Evaluation**: Measure detection performance using standard metrics
5. **Court Mapping**: Implement keypoint detection and 2D transformation
6. **Integration**: Combine detection and mapping for full pipeline
7. **Optimization**: Improve inference speed for real-time processing
8. **Analytics**: Develop downstream spatial analysis features (if time permits)

## Expected Outputs

### Per Frame
- Bounding boxes with persistent track IDs for all detected players
- Ball position (filtered and interpolated)
- Ball carrier / possession identification (algorithmic, not model-based)
- Team assignments for each player
- Court keypoint locations
- Transformed 2D court coordinates for all entities

### Per Video
- Annotated video with overlaid detections
- Spatial tracking data (CSV or JSON format)
- Performance metrics and statistics

## Notes

- This is an implementation project, not a novel research contribution in computer vision
- Focus is on applying existing techniques effectively to basketball video
- Original contributions expected in downstream analytics and insights
- Real-time performance is critical for practical deployment
