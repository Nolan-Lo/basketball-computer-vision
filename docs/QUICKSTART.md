# Basketball Video Analysis Pipeline - Quick Start Guide

## Setup Complete! ✓

Your pipeline is ready to process basketball videos through complete analysis:

### What the Pipeline Does

1. **Tracks Players** using YOLO + ByteTrack for persistent player IDs across frames
2. **Tracks Ball** using YOLO with outlier filtering and temporal interpolation
3. **Detects Court Keypoints** using your trained `court-keypoints.pt` model  
4. **Assigns Teams** to players using AI-powered jersey classification (CLIP)
5. **Detects Ball Possession** algorithmically from player/ball proximity and containment
6. **Outputs Annotated Video** with all detections, team colors, and ball carrier highlighted

## How to Run

### Option 1: Simple Run (Default Settings)

```bash
python main.py
```

This processes `input_videos/video_1.mp4` with default team descriptions.

### Option 2: Full Control

```bash
python -m src.pipeline \
    --input input_videos/video_1.mp4 \
    --output runs/my_output.mp4 \
    --team1 "white jersey" \
    --team2 "dark blue jersey"
```

## What You'll See in the Output

The output video will show:

- ⬜ **White boxes** = Team 1 players
- 🟧 **Orange boxes** = Team 2 players  
- 🟨 **Yellow boxes** = Basketball
- 🟪 **Magenta boxes** = Ball carrier (algorithmically detected)
- 🟡 **Yellow circles** = Court keypoints
- 📊 **Info panel** (top-right) with frame stats, ball carrier ID, and possession status

## Key Features

### 🚀 Performance
- **First run**: ~2-5 seconds per frame (includes model inference)
- **Subsequent runs**: ~0.5-1 second per frame (uses cached team assignments)
- **GPU acceleration**: Automatic if CUDA is available

### 💾 Smart Caching
Team assignments are cached automatically - rerunning the same video is much faster!

### 🎨 Customizable
Adjust team jersey descriptions for better accuracy:

```bash
python -m src.pipeline \
    --input input_videos/game.mp4 \
    --output runs/output.mp4 \
    --team1 "light gray jersey with blue stripes" \
    --team2 "dark red jersey with white numbers"
```

## Command Reference

```bash
# Basic usage
python -m src.pipeline --input VIDEO.mp4 --output OUTPUT.mp4

# Custom team descriptions
python -m src.pipeline -i VIDEO.mp4 -o OUTPUT.mp4 --team1 "white" --team2 "blue"

# Hide info panel
python -m src.pipeline -i VIDEO.mp4 -o OUTPUT.mp4 --no-info

# Custom cache location
python -m src.pipeline -i VIDEO.mp4 -o OUTPUT.mp4 --cache-dir my_cache/

# Use different models
python -m src.pipeline -i VIDEO.mp4 -o OUTPUT.mp4 \
    --player-model models/my_model.pt \
    --court-model models/my_court.pt
```

## Troubleshooting

**No video file?**
```bash
# Place your video in input_videos/ directory
cp /path/to/your/video.mp4 input_videos/video_1.mp4
```

**Models not found?**
```bash
# Check that your trained models exist:
ls models/Basketball-Players-17.pt
ls models/court-keypoints.pt
```

**Out of memory?**
- Close other applications
- Try processing fewer frames
- Reduce video resolution before processing

**Poor team classification?**
- Use more specific jersey descriptions
- Check that jersey colors are distinct in the video

## Example Workflow

```bash
# 1. Ensure you have a video
ls input_videos/video_1.mp4

# 2. Run the pipeline
python -m src.pipeline \
    --input input_videos/video_1.mp4 \
    --output runs/game_analysis.mp4 \
    --team1 "white jersey with gold trim" \
    --team2 "blue jersey with white numbers"

# 3. Check the output
# Output video saved to: runs/game_analysis.mp4

# 4. Re-run with different visualization (much faster with cache!)
python -m src.pipeline \
    --input input_videos/video_1.mp4 \
    --output runs/game_analysis_no_info.mp4 \
    --team1 "white jersey with gold trim" \
    --team2 "blue jersey with white numbers" \
    --no-info
```

## Next Steps

1. ✅ **Test it**: Run on your video
2. 📊 **Analyze**: Review the annotated output
3. 🎯 **Tune**: Adjust team descriptions for accuracy
4. 🚀 **Scale**: Process multiple videos
5. 📈 **Build analytics**: Use detection data for insights

## Documentation

- [Full Pipeline Documentation](PIPELINE.md) - Detailed usage guide
- [Team Assignment Details](TEAM_ASSIGNMENT.md) - How team classification works
- [Examples](../examples/) - Code examples for custom integration

## File Structure

```
capstone/
├── main.py                  # Quick start script
├── src/                    # Source code ⭐
│   ├── pipeline.py        # Main pipeline script
│   ├── trackers/          # Detection + tracking
│   │   ├── player_tracker.py  # YOLO + ByteTrack
│   │   └── ball_tracker.py    # YOLO + interpolation
│   ├── ball_acquisition_detector.py  # Ball possession
│   ├── team_assigner.py  # Team classification (CLIP)
│   ├── video_utils.py    # Video processing
│   ├── bbox_utils.py     # Bounding box geometry
│   └── utils.py          # Caching
├── models/
│   ├── Basketball-Players-17.pt    # Player/ball detector
│   └── court-keypoints.pt          # Court detector
├── input_videos/           # Put videos here
└── runs/                   # Outputs go here
```

## Need Help?

Check the full documentation:
```bash
python -m src.pipeline --help
```

Or see [PIPELINE.md](PIPELINE.md) for detailed usage examples.
