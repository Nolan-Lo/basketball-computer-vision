"""Basketball Video Analysis Pipeline

This script processes basketball videos through a complete analysis pipeline:
1. Player and ball detection (YOLO)
2. Court keypoint detection (YOLO Pose)
3. Team assignment (CLIP-based classification)
4. Visualization of all detections on output video
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse

from .team_assigner import TeamAssigner
from .video_utils import (
    read_video, save_video, draw_bounding_box, draw_keypoints_skeleton,
    add_info_panel, get_team_color, get_class_color
)


class BasketballAnalysisPipeline:
    """
    Complete pipeline for basketball video analysis.
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
        
        # Load YOLO models
        print(f"Loading player detection model: {player_model_path}")
        self.player_model = YOLO(player_model_path)
        
        print(f"Loading court keypoint model: {court_model_path}")
        self.court_model = YOLO(court_model_path)
        
        # Initialize team assigner
        print("Initializing team assigner...")
        self.team_assigner = TeamAssigner(
            team_1_class_name=team_1_description,
            team_2_class_name=team_2_description
        )
        
        print("✓ Pipeline initialized successfully\n")
    
    def detect_players_and_ball(self, frames, verbose=True):
        """
        Detect players and ball in all frames.
        
        Args:
            frames (list): List of video frames.
            verbose (bool): Whether to print progress.
        
        Returns:
            list: List of detections per frame.
        """
        if verbose:
            print("Step 1/3: Detecting players and ball...")
        
        all_detections = []
        
        for frame_num, frame in enumerate(frames):
            if verbose and frame_num % 30 == 0:
                print(f"  Processing frame {frame_num}/{len(frames)}")
            
            results = self.player_model(frame, verbose=False)
            
            frame_detections = {
                'players': [],
                'ball': None,
                'ball_carrier': None
            }
            
            for box in results[0].boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.player_model.names[class_id]
                
                detection = {
                    'bbox': bbox,
                    'confidence': conf,
                    'class_name': class_name,
                    'class_id': class_id
                }
                
                if 'player' in class_name.lower():
                    frame_detections['players'].append(detection)
                elif 'ball' in class_name.lower() and 'carrier' not in class_name.lower():
                    if frame_detections['ball'] is None or conf > frame_detections['ball']['confidence']:
                        frame_detections['ball'] = detection
                elif 'carrier' in class_name.lower():
                    if frame_detections['ball_carrier'] is None or conf > frame_detections['ball_carrier']['confidence']:
                        frame_detections['ball_carrier'] = detection
            
            all_detections.append(frame_detections)
        
        if verbose:
            print(f"✓ Detected players and ball in {len(frames)} frames\n")
        
        return all_detections
    
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
            print("Step 2/3: Detecting court keypoints...")
        
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
    
    def assign_teams(self, frames, detections, cache_path=None, verbose=True):
        """
        Assign teams to detected players.
        
        Args:
            frames (list): List of video frames.
            detections (list): List of player/ball detections per frame.
            cache_path (str): Path to cache file for team assignments.
            verbose (bool): Whether to print progress.
        
        Returns:
            list: List of team assignments per frame.
        """
        if verbose:
            print("Step 3/3: Assigning teams to players...")
        
        # Prepare player tracks for team assigner
        player_tracks = []
        for frame_detections in detections:
            frame_tracks = {}
            for player_idx, player in enumerate(frame_detections['players']):
                frame_tracks[player_idx] = {
                    'bbox': player['bbox']
                }
            player_tracks.append(frame_tracks)
        
        # Assign teams
        team_assignments = self.team_assigner.get_player_teams_across_frames(
            video_frames=frames,
            player_tracks=player_tracks,
            read_from_stub=(cache_path is not None),
            stub_path=cache_path
        )
        
        if verbose:
            print(f"✓ Assigned teams for {len(frames)} frames\n")
        
        return team_assignments
    
    def visualize_detections(self, frames, detections, keypoints, team_assignments, 
                           show_info=True, verbose=True):
        """
        Draw all detections on video frames.
        
        Args:
            frames (list): List of video frames.
            detections (list): List of player/ball detections per frame.
            keypoints (list): List of court keypoints per frame.
            team_assignments (list): List of team assignments per frame.
            show_info (bool): Whether to show info panel.
            verbose (bool): Whether to print progress.
        
        Returns:
            list: List of annotated frames.
        """
        if verbose:
            print("Visualizing detections on frames...")
        
        annotated_frames = []
        
        # Define court keypoint connections (if your model has specific court structure)
        # This is a simple example - adjust based on your court keypoint model
        court_connections = []  # Add connections if needed, e.g., [(0, 1), (1, 2), ...]
        
        for frame_num, frame in enumerate(frames):
            if verbose and frame_num % 30 == 0:
                print(f"  Annotating frame {frame_num}/{len(frames)}")
            
            annotated_frame = frame.copy()
            frame_detections = detections[frame_num]
            frame_keypoints = keypoints[frame_num]
            frame_teams = team_assignments[frame_num] if frame_num < len(team_assignments) else {}
            
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
            
            for player_idx, player in enumerate(frame_detections['players']):
                team_id = frame_teams.get(player_idx, 0)
                
                if team_id == 1:
                    team_1_count += 1
                elif team_id == 2:
                    team_2_count += 1
                
                color = get_team_color(team_id) if team_id > 0 else (0, 255, 0)
                label = f"Team {team_id}" if team_id > 0 else "Player"
                
                annotated_frame = draw_bounding_box(
                    annotated_frame,
                    player['bbox'],
                    color=color,
                    thickness=2,
                    label=label
                )
            
            # Draw ball
            if frame_detections['ball']:
                annotated_frame = draw_bounding_box(
                    annotated_frame,
                    frame_detections['ball']['bbox'],
                    color=get_class_color('ball'),
                    thickness=3,
                    label="Ball"
                )
            
            # Draw ball carrier
            if frame_detections['ball_carrier']:
                annotated_frame = draw_bounding_box(
                    annotated_frame,
                    frame_detections['ball_carrier']['bbox'],
                    color=get_class_color('ball_carrier'),
                    thickness=3,
                    label="Ball Carrier"
                )
            
            # Add info panel
            if show_info:
                info = {
                    'Frame': f"{frame_num + 1}/{len(frames)}",
                    'Team 1': team_1_count,
                    'Team 2': team_2_count,
                    'Ball': 'Yes' if frame_detections['ball'] else 'No',
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
            team_cache_path = cache_dir / f"{Path(video_path).stem}_teams.pkl"
        else:
            team_cache_path = None
        
        # Run pipeline steps
        detections = self.detect_players_and_ball(frames)
        keypoints = self.detect_court_keypoints(frames)
        team_assignments = self.assign_teams(frames, detections, cache_path=team_cache_path)
        
        # Visualize results
        annotated_frames = self.visualize_detections(
            frames, detections, keypoints, team_assignments, show_info=show_info
        )
        
        # Save output video
        print("Saving output video...")
        save_video(annotated_frames, output_path, fps=fps)
        
        # Calculate statistics
        total_players = sum(len(d['players']) for d in detections)
        total_ball_detections = sum(1 for d in detections if d['ball'] is not None)
        total_keypoints = sum(len(k) for k in keypoints)
        
        stats = {
            'total_frames': len(frames),
            'fps': fps,
            'resolution': f"{width}x{height}",
            'total_players': total_players,
            'avg_players_per_frame': total_players / len(frames),
            'ball_detection_rate': total_ball_detections / len(frames) * 100,
            'avg_keypoints_per_frame': total_keypoints / len(frames) if frames else 0
        }
        
        print(f"\n{'='*60}")
        print("Processing Complete!")
        print(f"{'='*60}")
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Avg Players/Frame: {stats['avg_players_per_frame']:.1f}")
        print(f"Ball Detection Rate: {stats['ball_detection_rate']:.1f}%")
        print(f"Avg Court Keypoints/Frame: {stats['avg_keypoints_per_frame']:.1f}")
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
