# Team Assignment Module - Implementation Guide

## Overview

The `TeamAssigner` class uses a pre-trained CLIP (Contrastive Language-Image Pre-training) vision-language model to automatically classify basketball players into teams based on their jersey colors and appearance.

## How It Works

### Technical Approach

1. **Visual Feature Extraction**: The system crops player bounding boxes from video frames
2. **Text-Image Matching**: Uses CLIP to compare the cropped image against text descriptions (e.g., "white jersey" vs "dark jersey")
3. **Classification**: Assigns players to teams based on which description best matches their appearance
4. **Caching**: Maintains player-team mappings across frames to reduce computation

### Model Details

- **Model**: `patrickjohncyh/fashion-clip` - A CLIP variant fine-tuned for fashion/clothing classification
- **Framework**: HuggingFace Transformers
- **Auto-Download**: The model downloads automatically on first use (~1GB)
- **No Manual Setup Required**: Just call `load_model()` and it handles everything

## Architecture Integration

### File Structure

```
capstone/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── team_assigner.py      # TeamAssigner class
│   └── utils.py              # Stub caching utilities
├── examples/
│   └── team_assignment_example.py  # Usage examples
├── models/
│   └── Basketball-Players-17.pt    # Player detection model
└── main.py                   # Main execution script
```

### Integration Points

1. **After Player Detection**: Run team assignment after YOLO detects player bounding boxes
2. **Before Analytics**: Team assignments enable team-specific analytics
3. **Caching Layer**: Use stub files to avoid reprocessing on repeated runs

## Usage Guide

### Basic Usage

```python
from src.team_assigner import TeamAssigner

# Initialize with jersey descriptions specific to your video
team_assigner = TeamAssigner(
    team_1_class_name="white jersey with gold trim",
    team_2_class_name="navy blue jersey"
)

# Load the model (automatic download on first use)
team_assigner.load_model()

# Classify a single player
team_id = team_assigner.get_player_team(frame, player_bbox, player_id)
print(f"Player assigned to Team {team_id}")
```

### Full Pipeline Integration

```python
from ultralytics import YOLO
from src.team_assigner import TeamAssigner
import cv2

# 1. Detect players with YOLO
player_model = YOLO("models/Basketball-Players-17.pt")
video = cv2.VideoCapture("input_videos/video_1.mp4")

video_frames = []
player_tracks = []

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    video_frames.append(frame)
    
    # Get player detections
    results = player_model(frame)
    frame_detections = {}
    
    for i, box in enumerate(results[0].boxes):
        player_id = i  # Use proper tracker in production
        bbox = box.xyxy[0].cpu().numpy()
        frame_detections[player_id] = {'bbox': bbox}
    
    player_tracks.append(frame_detections)

video.release()

# 2. Assign teams to all players across all frames
team_assigner = TeamAssigner(
    team_1_class_name="light jersey",
    team_2_class_name="dark jersey"
)

team_assignments = team_assigner.get_player_teams_across_frames(
    video_frames=video_frames,
    player_tracks=player_tracks,
    read_from_stub=True,  # Cache results
    stub_path="runs/team_assignments.pkl"
)

# 3. Use team assignments for analytics
for frame_num, assignments in enumerate(team_assignments):
    for player_id, team_id in assignments.items():
        # Now you know which team each player belongs to
        print(f"Frame {frame_num}, Player {player_id}: Team {team_id}")
```

## Key Features

### 1. Automatic Caching

The system automatically caches team assignments to disk, avoiding reprocessing:

```python
team_assignments = team_assigner.get_player_teams_across_frames(
    video_frames=video_frames,
    player_tracks=player_tracks,
    read_from_stub=True,  # Enable caching
    stub_path="runs/my_cache.pkl"  # Cache file location
)
```

### 2. Dynamic Player Tracking

The system resets player-team mappings every 50 frames to handle:
- Player substitutions
- Lighting changes
- Camera angle changes

### 3. Flexible Team Descriptions

Customize team descriptions to match the specific jerseys in your video:

```python
# Detailed descriptions work better
team_assigner = TeamAssigner(
    team_1_class_name="white jersey with blue stripes",
    team_2_class_name="red jersey with white numbers"
)

# Generic descriptions also work
team_assigner = TeamAssigner(
    team_1_class_name="light colored shirt",
    team_2_class_name="dark colored shirt"
)
```

## Dependencies

All required packages are in `pyproject.toml`:

- `transformers` - HuggingFace library for CLIP model
- `torch` - PyTorch backend for model inference
- `pillow` - Image processing
- `opencv-python` - Video frame handling

**No manual model downloads required** - the model downloads automatically from HuggingFace on first use.

## Performance Considerations

### Speed

- **First run**: Slower due to model download (~1GB) and initial classification
- **Cached runs**: Nearly instant - just loads from pickle file
- **Per-frame processing**: ~0.1-0.5 seconds per player depending on hardware

### GPU Acceleration

The CLIP model automatically uses GPU if available through PyTorch:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

For GPU usage, ensure you have:
- CUDA-compatible GPU
- `torch` with CUDA support installed

### Memory Usage

- Model: ~500MB GPU/RAM during inference
- Video frames: Depends on video resolution and length
- Consider processing video in batches for very long videos

## Expected Output Format

```python
# team_assignments is a list of dictionaries
# Format: team_assignments[frame_num][player_id] = team_id

team_assignments = [
    {1: 1, 2: 2, 3: 1, 4: 2},  # Frame 0: Player 1 & 3 are Team 1, Player 2 & 4 are Team 2
    {1: 1, 2: 2, 3: 1, 4: 2},  # Frame 1: Same assignments
    # ... more frames
]
```

## Next Steps

1. **Run Player Detection**: First detect players with your YOLO model
2. **Configure Team Descriptions**: Customize jersey descriptions for your specific game
3. **Process Video**: Run team assignment across all frames
4. **Build Analytics**: Use team assignments for advanced spatial analytics like:
   - Team possession tracking
   - Offensive/defensive formations
   - Player spacing analysis
   - Team-specific performance metrics

## Common Issues

### Issue: Model download fails
**Solution**: Check internet connection - model downloads from HuggingFace on first use

### Issue: Incorrect team assignments
**Solution**: Improve jersey descriptions to be more specific and distinctive

### Issue: Slow processing
**Solution**: Enable caching with `read_from_stub=True` and use GPU acceleration

### Issue: Memory errors
**Solution**: Process video in smaller batches instead of loading all frames at once

## See Also

- [team_assignment_example.py](../examples/team_assignment_example.py) - Complete usage examples
- [HuggingFace CLIP Documentation](https://huggingface.co/docs/transformers/model_doc/clip)
- Project notebooks for integration with court keypoint detection
