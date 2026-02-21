"""Basketball Video Analytics - Source Code Package

This package contains core modules for basketball video analysis:
- pipeline: Main video processing pipeline
- trackers: Player and ball tracking (YOLO + ByteTrack)
- ball_acquisition_detector: Determines which player has ball possession
- team_assigner: Assigns players to teams based on jersey colors
- video_utils: Video I/O and visualization utilities
- utils: Utility functions for caching and data handling
- bbox_utils: Bounding box geometry helpers
"""

__version__ = "0.1.0"

from .pipeline import BasketballAnalysisPipeline
from .trackers import PlayerTracker, BallTracker
from .ball_acquisition_detector import BallAcquisitionDetector
from .team_assigner import TeamAssigner
from .video_utils import read_video, save_video, draw_bounding_box, draw_keypoints_skeleton
from .utils import read_stub, save_stub
from .bbox_utils import measure_distance, get_center_of_bbox

__all__ = [
    'BasketballAnalysisPipeline',
    'PlayerTracker',
    'BallTracker',
    'BallAcquisitionDetector',
    'TeamAssigner',
    'read_video',
    'save_video',
    'draw_bounding_box',
    'draw_keypoints_skeleton',
    'read_stub',
    'save_stub',
    'measure_distance',
    'get_center_of_bbox',
]
