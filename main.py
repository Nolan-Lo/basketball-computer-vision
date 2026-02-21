"""Basketball Video Analysis - Main Entry Point

Quick start:
  uv run python main.py

For full pipeline with options:
  uv run python -m src.pipeline --input input_videos/video_1.mp4 --output runs/output.mp4
"""

from pathlib import Path
from src import BasketballAnalysisPipeline


def main():
    """Run the basketball video analysis pipeline with default settings."""
    
    print("\n" + "="*70)
    print(" Basketball Video Analysis Pipeline")
    print("="*70 + "\n")
    
    # Default configuration
    input_video = "input_videos/video_1.mp4"
    output_video = "runs/pipeline_output/video1_annotated.mp4"
    player_model = "models/Basketball-Players-17.pt"
    court_model = "models/court-keypoints.pt"
    court_image = "images/basketball_court.png"
    
    # Check if input video exists
    if not Path(input_video).exists():
        print(f"❌ Input video not found: {input_video}")
        print(f"\nPlease place a video file at: {input_video}")
        print(f"Or use the full pipeline script with custom path:")
        print(f"  python pipeline.py --input YOUR_VIDEO.mp4 --output OUTPUT.mp4\n")
        return
    
    # Check if models exist
    if not Path(player_model).exists():
        print(f"❌ Player detection model not found: {player_model}")
        print(f"Please ensure the model is trained and saved to: {player_model}\n")
        return
    
    if not Path(court_model).exists():
        print(f"❌ Court keypoint model not found: {court_model}")
        print(f"Please ensure the model is trained and saved to: {court_model}\n")
        return
    
    if not Path(court_image).exists():
        print(f"⚠ Court image not found: {court_image} (tactical view will be disabled)")
        court_image = None
    
    print("✓ All required files found")
    print(f"  Input: {input_video}")
    print(f"  Player Model: {player_model}")
    print(f"  Court Model: {court_model}")
    print(f"  Court Image: {court_image or 'N/A (tactical view disabled)'}")
    print(f"  Output: {output_video}\n")
    
    # Initialize and run pipeline
    try:
        pipeline = BasketballAnalysisPipeline(
            player_model_path=player_model,
            court_model_path=court_model,
            team_1_description="white jersey",
            team_2_description="dark jersey",
            court_image_path=court_image
        )
        
        pipeline.process_video(
            video_path=input_video,
            output_path=output_video,
            cache_dir="runs/cache",
            show_info=True
        )
        
        print("\n✓ Pipeline completed successfully!")
        print(f"\nTo customize settings, use:")
        print(f"  uv run python -m src.pipeline --help\n")
        
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        print(f"\nFor more options, try:")
        print(f"  uv run python -m src.pipeline --help\n")
        raise


if __name__ == "__main__":
    main()
