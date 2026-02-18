# Basketball Video Analysis Pipeline

## Complete Video Processing System

This pipeline processes basketball videos through multiple detection and analysis stages, outputting a single annotated video with all detections visible.

## Pipeline Architecture

```
Input Video
    ↓
┌───────────────────────────────────┐
│  Step 1: Player & Ball Detection  │  (YOLOv8)
│  - Detect all players             │
│  - Detect basketball              │
│  - Detect ball carrier            │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│  Step 2: Court Keypoint Detection │  (YOLOv8-Pose)
│  - Detect court boundary points   │
│  - Detect key court features      │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│  Step 3: Team Assignment          │  (CLIP)
│  - Classify players by jersey     │
│  - Assign to Team 1 or Team 2     │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│  Step 4: Visualization            │
│  - Draw bounding boxes            │
│  - Color-code by team             │
│  - Show court keypoints           │
│  - Add info panel                 │
└───────────────────────────────────┘
    ↓
Output Video (Fully Annotated)
```

## Quick Start

### Basic Usage

```bash
# Process a video with default settings
python -m src.pipeline --input input_videos/video_1.mp4 --output runs/output.mp4
```

This will:
1. Load your trained models from `models/`
2. Process the video through all pipeline stages
3. Output an annotated video to `runs/output.mp4`
4. Cache team assignments for faster reprocessing

### Custom Team Descriptions

For better accuracy, customize the jersey descriptions to match your specific game:

```bash
python -m src.pipeline \
    --input input_videos/video_1.mp4 \
    --output runs/lakers_celtics.mp4 \
    --team1 "yellow jersey with purple trim" \
    --team2 "green jersey with white numbers"
```

## Command Line Options

```
Required Arguments:
  --input, -i          Path to input video file
  --output, -o         Path for output video file

Optional Arguments:
  --player-model       Path to player/ball detection model
                       (default: models/Basketball-Players-17.pt)
  
  --court-model        Path to court keypoint model
                       (default: models/court-keypoints.pt)
  
  --team1              Description of Team 1's jersey
                       (default: "white jersey")
  
  --team2              Description of Team 2's jersey
                       (default: "dark jersey")
  
  --cache-dir          Directory for cache files
                       (default: runs/cache)
  
  --no-info            Hide info panel on output video
```

## Using as a Python Module

You can also import and use the pipeline programmatically:

```python
from pipeline import BasketballAnalysisPipeline

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

print(f"Processed {stats['total_frames']} frames")
print(f"Avg players per frame: {stats['avg_players_per_frame']:.1f}")
```

## Output Visualization

The pipeline outputs a video with the following annotations:

### Color Coding
- **White boxes**: Team 1 players
- **Orange boxes**: Team 2 players
- **Yellow boxes**: Basketball
- **Magenta boxes**: Ball carrier
- **Yellow circles**: Court keypoints

### Info Panel (Top-Right)
- Current frame number
- Team 1 player count
- Team 2 player count
- Ball detection status
- Court keypoint count

## Performance Features

### Caching System

The pipeline automatically caches team assignments to speed up reprocessing:

```python
# First run: ~2-5 seconds per frame (model inference + team classification)
python pipeline.py --input video.mp4 --output output1.mp4

# Subsequent runs: ~0.5-1 second per frame (cached team assignments)
python pipeline.py --input video.mp4 --output output2.mp4
```

Cache files are stored in `runs/cache/` by default.

### GPU Acceleration

The pipeline automatically uses GPU if available:

```python
import torch
print(f"Using GPU: {torch.cuda.is_available()}")
```

For GPU support, ensure you have CUDA-enabled PyTorch installed.

## Examples

### Example 1: Basic Processing

```bash
python -m src.pipeline \
    --input input_videos/video_1.mp4 \
    --output runs/video1_annotated.mp4
```

### Example 2: Custom Teams

```bash
python -m src.pipeline \
    --input input_videos/game2.mp4 \
    --output runs/game2_annotated.mp4 \
    --team1 "light gray jersey" \
    --team2 "dark red jersey"
```

### Example 3: No Info Panel

```bash
python -m src.pipeline \
    --input input_videos/video_1.mp4 \
    --output runs/clean_output.mp4 \
    --no-info
```

### Example 4: Custom Models

```bash
python -m src.pipeline \
    --input input_videos/video_1.mp4 \
    --output runs/output.mp4 \
    --player-model models/custom_player_model.pt \
    --court-model models/custom_court_model.pt
```

## Pipeline Components

### 1. Player & Ball Detection (`Basketball-Players-17.pt`)
- Trained on basketball-specific dataset
- Detects: players, basketball, ball carrier
- Output: Bounding boxes with confidence scores

### 2. Court Keypoint Detection (`court-keypoints.pt`)
- YOLO pose model for court features
- Detects: court boundary points, key locations
- Output: (x, y, confidence) keypoints

### 3. Team Assignment (`TeamAssigner`)
- Uses CLIP vision-language model
- Classifies players by jersey appearance
- Maintains consistent IDs across frames
- Auto-caches results for performance

### 4. Visualization (`video_utils`)
- Draws all detections on frames
- Color-codes by team and object type
- Adds informative overlays
- Exports to MP4 video

## File Structure

```
capstone/
├── main.py                         # Simple entry point
├── src/                            # Source code
│   ├── pipeline.py                 # Main pipeline script
│   ├── team_assigner.py            # Team classification
│   ├── video_utils.py              # Video I/O and drawing
│   └── utils.py                    # Caching utilities
├── models/
│   ├── Basketball-Players-17.pt # Player/ball detector
│   └── court-keypoints.pt       # Court keypoint detector
├── input_videos/                # Put input videos here
└── runs/                        # Output videos and cache
```

## Troubleshooting

### Issue: "Model not found"
**Solution**: Ensure models are in the `models/` directory:
```bash
ls models/Basketball-Players-17.pt
ls models/court-keypoints.pt
```

### Issue: Out of memory
**Solution**: Process fewer frames or reduce video resolution:
```python
# In pipeline.py, modify read_video to limit frames:
frames = frames[:300]  # Process first 300 frames only
```

### Issue: Poor team classification
**Solution**: Provide more specific jersey descriptions:
```bash
--team1 "bright white jersey with navy blue numbers and stripes"
--team2 "dark navy blue jersey with white trim"
```

### Issue: Slow processing
**Solution**: 
1. Enable caching (automatic by default)
2. Use GPU acceleration
3. Reduce video resolution before processing

## Next Steps

1. **Test the pipeline**: Run on your test video
2. **Tune team descriptions**: Adjust for your specific game
3. **Add analytics**: Use the detection data for spatial analysis
4. **Integrate 2D mapping**: Transform coordinates to standard court view

## Advanced Usage

See [examples/team_assignment_example.py](examples/team_assignment_example.py) for more advanced integration patterns and customization options.
