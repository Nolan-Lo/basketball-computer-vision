"""Basketball Video Analytics - Source Code Package

This package contains core modules for basketball video analysis:
- pipeline: Main video processing pipeline
- team_assigner: Assigns players to teams based on jersey colors
- video_utils: Video I/O and visualization utilities
- utils: Utility functions for caching and data handling
"""

__version__ = "0.1.0"

from .pipeline import BasketballAnalysisPipeline
from .team_assigner import TeamAssigner
from .video_utils import read_video, save_video, draw_bounding_box, draw_keypoints_skeleton
from .utils import read_stub, save_stub

__all__ = [
    'BasketballAnalysisPipeline',
    'TeamAssigner',
    'read_video',
    'save_video',
    'draw_bounding_box',
    'draw_keypoints_skeleton',
    'read_stub',
    'save_stub',
]
