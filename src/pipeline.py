"""Basketball Video Analysis Pipeline

This script processes basketball videos through a complete analysis pipeline:
1. Player detection & tracking (YOLO + ByteTrack)
2. Ball detection, filtering & interpolation (YOLO + temporal smoothing)
3. Court keypoint detection (YOLOv8-Pose)
4. Team assignment (CLIP-based classification)
5. Ball possession detection (proximity + containment heuristics)
6. Visualization of all detections on output video
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse

from .trackers import PlayerTracker, BallTracker
from .ball_acquisition_detector import BallAcquisitionDetector
from .team_assigner import TeamAssigner
from .video_utils import (
    read_video, save_video, draw_bounding_box, draw_keypoints_skeleton,
    add_info_panel, get_team_color, get_class_color
)


class BasketballAnalysisPipeline:
    """
    Complete pipeline for basketball video analysis.

    Pipeline stages:
        1. Player detection & tracking (YOLO + ByteTrack)
        2. Ball detection, outlier removal & interpolation
        3. Court keypoint detection (YOLO Pose)
        4. Team assignment (CLIP)
        5. Ball possession detection
        6. Visualization
    """
    
    def __init__(self, 
                 player_model_path,
                 court_model_path,
                 team_1_description="white jersey",
                 team_2_description="dark jersey"):
        """
        Initialize the pipeline with model paths and team descriptions.
        
        Args:
            player_model_path (str): Path to the player/ball detection model.
            court_model_path (str): Path to the court keypoint detection model.
            team_1_description (str): Description of Team 1's jersey.
            team_2_description (str): Description of Team 2's jersey.
        """
        print("Initializing Basketball Analysis Pipeline...")
        
        # Player tracker (YOLO + ByteTrack)
        print(f"Loading player tracker: {player_model_path}")
        self.player_tracker = PlayerTracker(player_model_path)

        # Ball tracker (YOLO + temporal filtering)
        print(f"Loading ball tracker: {player_model_path}")
        self.ball_tracker = BallTracker(player_model_path)

        # Court keypoint model
        print(f"Loading court keypoint model: {court_model_path}")
        self.court_model = YOLO(court_model_path)

        # Ball possession detector
        self.ball_acquisition_detector = BallAcquisitionDetector()
        
        # Initialize team assigner
        print("Initializing team assigner...")
        self.team_assigner = TeamAssigner(
            team_1_class_name=team_1_description,
            team_2_class_name=team_2_description
        )
        
        print("✓ Pipeline initialized successfully\n")
    
    def track_players(self, frames, cache_path=None, verbose=True):
        """
        Detect and track players across all frames using ByteTrack.
        
        Args:
            frames (list): List of video frames.
            cache_path (str): Optional path to cache file.
            verbose (bool): Whether to print progress.
        
        Returns:
            list: List of dicts per frame mapping track_id to {"bbox": [...]}.
        """
        if verbose:
            print("Step 1/5: Tracking players (YOLO + ByteTrack)...")
        
        player_tracks = self.player_tracker.get_object_tracks(
            frames,
            read_from_stub=(cache_path is not None),
            stub_path=str(cache_path) if cache_path else None
        )
        
        if verbose:
            total_detections = sum(len(f) for f in player_tracks)
            print(f"✓ Tracked players across {len(frames)} frames "
                  f"({total_detections} total detections)\n")
        
        return player_tracks
    
    def track_ball(self, frames, cache_path=None, verbose=True):
        """
        Detect, filter and interpolate ball positions across all frames.
        
        Args:
            frames (list): List of video frames.
            cache_path (str): Optional path to cache file.
            verbose (bool): Whether to print progress.
        
        Returns:
            list: List of dicts per frame with key 1 → {"bbox": [...]}.
        """
        if verbose:
            print("Step 2/5: Tracking ball (YOLO + filtering + interpolation)...")
        
        ball_tracks = self.ball_tracker.get_object_tracks(
            frames,
            read_from_stub=(cache_path is not None),
            stub_path=str(cache_path) if cache_path else None
        )
        
        raw_detection_count = sum(1 for f in ball_tracks if f.get(1))
        
        # Remove outlier detections (impossible jumps)
        ball_tracks = self.ball_tracker.remove_wrong_detections(ball_tracks)
        
        # Interpolate gaps for smooth tracking
        ball_tracks = self.ball_tracker.interpolate_ball_positions(ball_tracks)
        
        if verbose:
            print(f"✓ Ball tracked in {len(frames)} frames "
                  f"(raw detections: {raw_detection_count}, "
                  f"after interpolation: {len(ball_tracks)})\n")
        
        return ball_tracks
    
    def detect_court_keypoints(self, frames, verbose=True):
        """
        Detect court keypoints in all frames.
        
        Args:
            frames (list): List of video frames.
            verbose (bool): Whether to print progress.
        
        Returns:
            list: List of keypoint detections per frame.
        """
        if verbose:
            print("Step 3/5: Detecting court keypoints...")
        
        all_keypoints = []
        
        for frame_num, frame in enumerate(frames):
            if verbose and frame_num % 30 == 0:
                print(f"  Processing frame {frame_num}/{len(frames)}")
            
            results = self.court_model(frame, verbose=False)
            
            frame_keypoints = []
            
            # YOLO pose results contain keypoints
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                keypoints_data = results[0].keypoints.xy.cpu().numpy()
                
                if len(keypoints_data) > 0:
                    # Get the first detection's keypoints
                    kpts = keypoints_data[0]  # Shape: (num_keypoints, 2)
                    
                    # Add confidence if available
                    if hasattr(results[0].keypoints, 'conf'):
                        conf = results[0].keypoints.conf.cpu().numpy()[0]
                        for i, (x, y) in enumerate(kpts):
                            frame_keypoints.append((x, y, conf[i]))
                    else:
                        for x, y in kpts:
                            frame_keypoints.append((x, y, 1.0))
            
            all_keypoints.append(frame_keypoints)
        
        if verbose:
            print(f"✓ Detected court keypoints in {len(frames)} frames\n")
        
        return all_keypoints
    
    def assign_teams(self, frames, player_tracks, cache_path=None, verbose=True):
        """
        Assign teams to tracked players.
        
        Args:
            frames (list): List of video frames.
            player_tracks (list): Per-frame dicts mapping track_id to {"bbox": [...]}.
            cache_path (str): Path to cache file for team assignments.
            verbose (bool): Whether to print progress.
        
        Returns:
            list: List of dicts mapping track_id to team_id per frame.
        """
        if verbose:
            print("Step 4/5: Assigning teams to players...")
        
        # player_tracks already in the format TeamAssigner expects
        team_assignments = self.team_assigner.get_player_teams_across_frames(
            video_frames=frames,
            player_tracks=player_tracks,
            read_from_stub=(cache_path is not None),
            stub_path=str(cache_path) if cache_path else None
        )
        
        if verbose:
            print(f"✓ Assigned teams for {len(frames)} frames\n")
        
        return team_assignments

    def detect_ball_possession(self, player_tracks, ball_tracks, verbose=True):
        """
        Determine which player has the ball on each frame.
        
        Args:
            player_tracks (list): Per-frame player track dicts.
            ball_tracks (list): Per-frame ball track dicts.
            verbose (bool): Whether to print progress.
        
        Returns:
            list[int]: Per-frame player_id with possession, or -1.
        """
        if verbose:
            print("Step 5/5: Detecting ball possession...")
        
        possession = self.ball_acquisition_detector.detect_ball_possession(
            player_tracks, ball_tracks
        )
        
        if verbose:
            frames_with_possession = sum(1 for p in possession if p != -1)
            print(f"✓ Ball possession detected in {frames_with_possession}/{len(possession)} frames\n")
        
        return possession
    
    def visualize_detections(self, frames, player_tracks, ball_tracks, keypoints,
                           team_assignments, possession, show_info=True, verbose=True):
        """
        Draw all detections on video frames.
        
        Args:
            frames (list): List of video frames.
            player_tracks (list): Per-frame player track dicts.
            ball_tracks (list): Per-frame ball track dicts.
            keypoints (list): List of court keypoints per frame.
            team_assignments (list): Per-frame team assignment dicts.
            possession (list[int]): Per-frame possessing player_id (-1 = none).
            show_info (bool): Whether to show info panel.
            verbose (bool): Whether to print progress.
        
        Returns:
            list: List of annotated frames.
        """
        if verbose:
            print("Visualizing detections on frames...")
        
        annotated_frames = []
        
        # Court keypoint connections (adjust if your model defines specific topology)
        court_connections = []
        
        for frame_num, frame in enumerate(frames):
            if verbose and frame_num % 30 == 0:
                print(f"  Annotating frame {frame_num}/{len(frames)}")
            
            annotated_frame = frame.copy()
            frame_players = player_tracks[frame_num] if frame_num < len(player_tracks) else {}
            frame_ball = ball_tracks[frame_num] if frame_num < len(ball_tracks) else {}
            frame_keypoints = keypoints[frame_num] if frame_num < len(keypoints) else []
            frame_teams = team_assignments[frame_num] if frame_num < len(team_assignments) else {}
            frame_possession = possession[frame_num] if frame_num < len(possession) else -1
            
            # Draw court keypoints first (background layer)
            if frame_keypoints:
                annotated_frame = draw_keypoints_skeleton(
                    annotated_frame, 
                    frame_keypoints,
                    connections=court_connections,
                    color=(0, 255, 255)  # Yellow
                )
            
            # Draw players with team colors
            team_1_count = 0
            team_2_count = 0
            
            for track_id, player_info in frame_players.items():
                team_id = frame_teams.get(track_id, 0)
                
                if team_id == 1:
                    team_1_count += 1
                elif team_id == 2:
                    team_2_count += 1
                
                # Highlight the ball carrier
                is_carrier = (track_id == frame_possession)
                
                if is_carrier:
                    color = get_class_color('ball_carrier')
                    label = f"Team {team_id} [BALL]" if team_id > 0 else "Player [BALL]"
                    thickness = 3
                else:
                    color = get_team_color(team_id) if team_id > 0 else (0, 255, 0)
                    label = f"Team {team_id}" if team_id > 0 else "Player"
                    thickness = 2
                
                annotated_frame = draw_bounding_box(
                    annotated_frame,
                    player_info['bbox'],
                    color=color,
                    thickness=thickness,
                    label=label
                )
            
            # Draw ball
            ball_info = frame_ball.get(1, {})
            if ball_info:
                annotated_frame = draw_bounding_box(
                    annotated_frame,
                    ball_info['bbox'],
                    color=get_class_color('ball'),
                    thickness=3,
                    label="Ball"
                )
            
            # Add info panel
            if show_info:
                info = {
                    'Frame': f"{frame_num + 1}/{len(frames)}",
                    'Team 1': team_1_count,
                    'Team 2': team_2_count,
                    'Ball': 'Yes' if ball_info else 'No',
                    'Carrier': frame_possession if frame_possession != -1 else 'N/A',
                    'Keypoints': len(frame_keypoints)
                }
                annotated_frame = add_info_panel(annotated_frame, info, position='top-right')
            
            annotated_frames.append(annotated_frame)
        
        if verbose:
            print(f"✓ Annotated {len(frames)} frames\n")
        
        return annotated_frames
    
    def process_video(self, video_path, output_path, cache_dir=None, show_info=True):
        """
        Process a complete video through the pipeline.
        
        Args:
            video_path (str): Path to input video.
            output_path (str): Path for output video.
            cache_dir (str): Directory for cache files.
            show_info (bool): Whether to show info panel on video.
        
        Returns:
            dict: Processing statistics.
        """
        print(f"\n{'='*60}")
        print(f"Processing Video: {video_path}")
        print(f"{'='*60}\n")
        
        # Read video
        print("Reading video...")
        frames, fps, width, height = read_video(video_path)
        print(f"✓ Loaded {len(frames)} frames ({width}x{height} @ {fps} FPS)\n")
        
        # Setup cache paths
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            player_cache_path = cache_dir / f"{Path(video_path).stem}_player_tracks.pkl"
            ball_cache_path = cache_dir / f"{Path(video_path).stem}_ball_tracks.pkl"
            team_cache_path = cache_dir / f"{Path(video_path).stem}_teams.pkl"
        else:
            player_cache_path = None
            ball_cache_path = None
            team_cache_path = None
        
        # Run pipeline steps
        player_tracks = self.track_players(frames, cache_path=player_cache_path)
        ball_tracks = self.track_ball(frames, cache_path=ball_cache_path)
        keypoints = self.detect_court_keypoints(frames)
        team_assignments = self.assign_teams(frames, player_tracks, cache_path=team_cache_path)
        possession = self.detect_ball_possession(player_tracks, ball_tracks)
        
        # Visualize results
        annotated_frames = self.visualize_detections(
            frames, player_tracks, ball_tracks, keypoints,
            team_assignments, possession, show_info=show_info
        )
        
        # Save output video
        print("Saving output video...")
        save_video(annotated_frames, output_path, fps=fps)
        
        # Calculate statistics
        total_players = sum(len(f) for f in player_tracks)
        total_ball_detections = sum(1 for f in ball_tracks if f.get(1))
        total_keypoints = sum(len(k) for k in keypoints)
        frames_with_possession = sum(1 for p in possession if p != -1)
        
        stats = {
            'total_frames': len(frames),
            'fps': fps,
            'resolution': f"{width}x{height}",
            'total_players': total_players,
            'avg_players_per_frame': total_players / len(frames),
            'ball_detection_rate': total_ball_detections / len(frames) * 100,
            'avg_keypoints_per_frame': total_keypoints / len(frames) if frames else 0,
            'possession_detection_rate': frames_with_possession / len(frames) * 100
        }
        
        print(f"\n{'='*60}")
        print("Processing Complete!")
        print(f"{'='*60}")
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Avg Players/Frame: {stats['avg_players_per_frame']:.1f}")
        print(f"Ball Detection Rate: {stats['ball_detection_rate']:.1f}%")
        print(f"Avg Court Keypoints/Frame: {stats['avg_keypoints_per_frame']:.1f}")
        print(f"Ball Possession Rate: {stats['possession_detection_rate']:.1f}%")
        print(f"Output saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return stats


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Basketball Video Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video with default settings
  python pipeline.py --input input_videos/video_1.mp4 --output runs/output.mp4
  
  # Customize team jersey descriptions
  python pipeline.py --input input_videos/video_1.mp4 --output runs/output.mp4 \\
      --team1 "white jersey with blue trim" --team2 "red jersey with black numbers"
  
  # Use caching for faster reprocessing
  python pipeline.py --input input_videos/video_1.mp4 --output runs/output.mp4 \\
      --cache-dir runs/cache
        """
    )
    
    parser.add_argument('--input', '-i', required=True, 
                       help='Path to input video file')
    parser.add_argument('--output', '-o', required=True,
                       help='Path for output video file')
    parser.add_argument('--player-model', default='models/Basketball-Players-17.pt',
                       help='Path to player/ball detection model (default: models/Basketball-Players-17.pt)')
    parser.add_argument('--court-model', default='models/court-keypoints.pt',
                       help='Path to court keypoint model (default: models/court-keypoints.pt)')
    parser.add_argument('--team1', default='white jersey',
                       help='Description of Team 1 jersey (default: "white jersey")')
    parser.add_argument('--team2', default='dark jersey',
                       help='Description of Team 2 jersey (default: "dark jersey")')
    parser.add_argument('--cache-dir', default='runs/cache',
                       help='Directory for cache files (default: runs/cache)')
    parser.add_argument('--no-info', action='store_true',
                       help='Hide info panel on output video')
    
    args = parser.parse_args()
    
    # Validate input paths
    if not Path(args.input).exists():
        print(f"Error: Input video not found: {args.input}")
        return
    
    if not Path(args.player_model).exists():
        print(f"Error: Player model not found: {args.player_model}")
        return
    
    if not Path(args.court_model).exists():
        print(f"Error: Court model not found: {args.court_model}")
        return
    
    # Initialize pipeline
    pipeline = BasketballAnalysisPipeline(
        player_model_path=args.player_model,
        court_model_path=args.court_model,
        team_1_description=args.team1,
        team_2_description=args.team2
    )
    
    # Process video
    pipeline.process_video(
        video_path=args.input,
        output_path=args.output,
        cache_dir=args.cache_dir,
        show_info=not args.no_info
    )


if __name__ == "__main__":
    main()
