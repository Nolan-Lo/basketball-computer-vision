# Basketball Video Analytics

Automated basketball video analysis system that captures spatial and off-ball dynamics using computer vision.

## Overview

This system processes broadcast basketball videos to identify and track key game entities in real-time:
- **Players** - Detect and track all players on court
- **Ball** - Locate the basketball
- **Teams** - Classify players into teams based on jersey appearance
- **Court** - Detect court keypoints for spatial mapping

## Quick Start

### Run the Complete Pipeline

```bash
# Process a video with all detections
uv run python -m src.pipeline \
    --input input_videos/video_1.mp4 \
    --output runs/output.mp4 \
    --team1 "white jersey" \
    --team2 "dark jersey"
```

### Or use the simplified main script

```bash
uv run python main.py
```

## Pipeline Architecture

```
Input Video →  Player/Ball Detection  →  Court Keypoints  →  Team Assignment  →  Annotated Output
              (YOLOv8)                  (YOLOv8-Pose)       (CLIP)               (Single Video)
```

### Pipeline Stages

1. **Player & Ball Detection** - Detect all players, ball, and ball carrier using trained YOLO model
2. **Court Keypoint Detection** - Identify court boundary points and key locations using YOLO Pose  
3. **Team Assignment** - Classify players into teams using CLIP vision-language model
4. **Visualization** - Draw all detections on video with color-coding and info panel

## Features

✅ **All-in-One Pipeline**: Single script processes video through all stages  
✅ **Smart Caching**: Team assignments cached for faster reprocessing  
✅ **GPU Acceleration**: Automatic GPU usage if CUDA available  
✅ **Customizable**: Adjust team descriptions, colors, and visualization  
✅ **Info Panel**: Real-time stats overlay on output video

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd capstone

# Install dependencies with uv
uv sync
```

## Project Structure

```
capstone/
├── main.py                          # Quick start entry point
├── src/                            # Source code ⭐
│   ├── pipeline.py                 # Main pipeline script
│   ├── team_assigner.py            # Team classification module
│   ├── video_utils.py              # Video processing utilities
│   └── utils.py                    # Caching utilities
├── models/
│   ├── Basketball-Players-17.pt    # Player/ball detector
│   └── court-keypoints.pt          # Court keypoint detector
├── input_videos/                   # Place input videos here
├── runs/                           # Output videos and cache
├── notebooks/                      # Jupyter notebooks
│   ├── 01-data-exploration.ipynb
│   ├── 02-player-ball-detection.ipynb
│   └── 03-court-keypoint-detection.ipynb
├── data/                           # Training datasets
│   ├── Basketball-Players-17/
│   └── court-keypoints/
├── docs/                           # Documentation
│   ├── QUICKSTART.md              # Quick start guide
│   ├── PIPELINE.md                # Detailed pipeline docs
│   └── TEAM_ASSIGNMENT.md         # Team assignment details
└── examples/                       # Example scripts
    └── team_assignment_example.py
```

## Usage Examples

### Basic Usage

```bash
uv run python -m src.pipeline \
    --input input_videos/video_1.mp4 \
    --output runs/annotated_video.mp4
```

### Custom Team Descriptions (Better Accuracy)

```bash
uv run python -m src.pipeline \
    --input input_videos/game.mp4 \
    --output runs/game_annotated.mp4 \
    --team1 "white jersey with gold numbers" \
    --team2 "navy blue jersey with white trim"
```

### Hide Info Panel

```bash
uv run python -m src.pipeline \
    --input input_videos/video_1.mp4 \
    --output runs/clean_output.mp4 \
    --no-info
```

### Use as Python Module

```python
from src import BasketballAnalysisPipeline

# Initialize pipeline
pipeline = BasketballAnalysisPipeline(
    player_model_path='models/Basketball-Players-17.pt',
    court_model_path='models/court-keypoints.pt',
    team_1_description='white jersey',
    team_2_description='blue jersey'
)

# Process video
stats = pipeline.process_video(
    video_path='input_videos/video_1.mp4',
    output_path='runs/output.mp4',
    cache_dir='runs/cache',
    show_info=True
)
```

## Output Video Visualization

The pipeline produces annotated videos with:

- ⬜ **White boxes** - Team 1 players
- 🟧 **Orange boxes** - Team 2 players
- 🟨 **Yellow boxes** - Basketball
- 🟪 **Magenta boxes** - Ball carrier
- 🟡 **Yellow circles** - Court keypoints
- 📊 **Info panel** (optional) - Frame stats in top-right

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Get started quickly
- [Pipeline Documentation](docs/PIPELINE.md) - Detailed usage guide
- [Team Assignment](docs/TEAM_ASSIGNMENT.md) - How team classification works
- [Project Instructions](.github/copilot-instructions.md) - Project context

## Models

### Player & Ball Detection (`Basketball-Players-17.pt`)
- **Architecture**: YOLOv8
- **Classes**: player, ball, ball_carrier
- **Training**: Fine-tuned on basketball-specific dataset
- **Performance**: Real-time inference (~30 FPS on GPU)

### Court Keypoint Detection (`court-keypoints.pt`)
- **Architecture**: YOLOv8-Pose
- **Output**: Court boundary points and key locations
- **Use Case**: Spatial mapping and 2D court transformation

### Team Classification
- **Model**: CLIP (fashion-clip variant from HuggingFace)
- **Method**: Text-image matching for jersey appearance
- **Downloads**: Automatically on first use (~1GB)
- **Caching**: Results cached for performance

## Performance

| Metric | First Run | Cached Run |
|--------|-----------|------------|
| **Processing Speed** | 2-5 sec/frame | 0.5-1 sec/frame |
| **GPU Memory** | ~2-4 GB | ~1-2 GB |
| **Model Loading** | ~30 sec | ~5 sec |

*Tested on NVIDIA GPU. CPU processing is 5-10x slower.*

## Requirements

- Python 3.12+
- CUDA-compatible GPU (recommended)
- ~5GB disk space for models and dependencies

### Key Dependencies

- `ultralytics` - YOLO models
- `transformers` - CLIP model for team assignment
- `opencv-python` - Video processing
- `torch` - PyTorch backend
- `pillow` - Image processing

## Development

### Training Models

See Jupyter notebooks in `notebooks/`:
- [Player Detection Training](notebooks/02-player-ball-detection.ipynb)
- [Court Keypoint Training](notebooks/03-court-keypoint-detection.ipynb)

### Running Examples

```bash
# Run team assignment example
uv run python examples/team_assignment_example.py

# Run main pipeline with defaults
uv run python main.py
```

## Troubleshooting

**Pipeline runs slowly?**
- Ensure GPU is available: `torch.cuda.is_available()`
- Enable caching with `--cache-dir` (enabled by default)
- Process shorter video clips for testing

**Poor team classification?**
- Use more specific jersey descriptions
- Ensure jersey colors are visually distinct
- Check that lighting is consistent

**Out of memory?**
- Reduce video resolution
- Process fewer frames at a time
- Close other GPU applications

## Future Work

- [ ] 2D court coordinate transformation
- [ ] Player tracking with consistent IDs across frames
- [ ] Spatial analytics (player spacing, heat maps)
- [ ] Shot quality and player gravity metrics
- [ ] Real-time streaming support

## Project Context

This project develops a computer vision system for automated basketball video analysis that captures spatial and off-ball dynamics not available in traditional box scores. See [project instructions](.github/copilot-instructions.md) for full context.

## License

[Add your license here]

## Acknowledgments

- YOLO models from Ultralytics
- CLIP model from HuggingFace
- Basketball datasets from Roboflow
