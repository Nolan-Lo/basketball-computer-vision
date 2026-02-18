"""Video processing utilities for basketball video analysis."""

import cv2
import numpy as np
from pathlib import Path


def read_video(video_path):
    """
    Read all frames from a video file.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        tuple: (frames, fps, width, height)
            - frames: List of video frames as numpy arrays
            - fps: Frames per second
            - width: Frame width
            - height: Frame height
    """
    video = cv2.VideoCapture(str(video_path))
    frames = []
    
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    
    video.release()
    return frames, fps, width, height


def save_video(frames, output_path, fps=30):
    """
    Save frames to a video file.
    
    Args:
        frames (list): List of video frames as numpy arrays.
        output_path (str): Path where the output video should be saved.
        fps (int): Frames per second for the output video.
    """
    if len(frames) == 0:
        print("No frames to save!")
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"✓ Video saved to {output_path}")


def draw_bounding_box(frame, bbox, color=(0, 255, 0), thickness=2, label=None):
    """
    Draw a bounding box on a frame.
    
    Args:
        frame (numpy.ndarray): The frame to draw on.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        color (tuple): BGR color for the box.
        thickness (int): Line thickness.
        label (str): Optional label to display above the box.
    
    Returns:
        numpy.ndarray: Frame with bounding box drawn.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame


def draw_keypoint(frame, point, color=(0, 0, 255), radius=5):
    """
    Draw a keypoint (circle) on a frame.
    
    Args:
        frame (numpy.ndarray): The frame to draw on.
        point (tuple): (x, y) coordinates of the keypoint.
        color (tuple): BGR color for the keypoint.
        radius (int): Radius of the circle.
    
    Returns:
        numpy.ndarray: Frame with keypoint drawn.
    """
    x, y = map(int, point)
    cv2.circle(frame, (x, y), radius, color, -1)
    # Draw white outline for better visibility
    cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)
    return frame


def draw_keypoints_skeleton(frame, keypoints, connections=None, color=(0, 255, 255)):
    """
    Draw keypoints and their connections (skeleton) on a frame.
    
    Args:
        frame (numpy.ndarray): The frame to draw on.
        keypoints (list): List of (x, y, confidence) tuples for each keypoint.
        connections (list): List of (idx1, idx2) tuples indicating which keypoints to connect.
        color (tuple): BGR color for the skeleton.
    
    Returns:
        numpy.ndarray: Frame with skeleton drawn.
    """
    # Draw connections first (so they appear behind the points)
    if connections:
        for idx1, idx2 in connections:
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
                x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
                conf1 = keypoints[idx1][2] if len(keypoints[idx1]) > 2 else 1.0
                conf2 = keypoints[idx2][2] if len(keypoints[idx2]) > 2 else 1.0
                
                # Only draw if both keypoints are confident
                if conf1 > 0.5 and conf2 > 0.5:
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw keypoints
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        conf = kp[2] if len(kp) > 2 else 1.0
        
        if conf > 0.5:  # Only draw confident keypoints
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
    
    return frame


def add_info_panel(frame, info_dict, position='top-left'):
    """
    Add an information panel to a frame.
    
    Args:
        frame (numpy.ndarray): The frame to draw on.
        info_dict (dict): Dictionary of label: value pairs to display.
        position (str): Position of the panel ('top-left', 'top-right', 'bottom-left', 'bottom-right').
    
    Returns:
        numpy.ndarray: Frame with info panel drawn.
    """
    height, width = frame.shape[:2]
    
    # Panel settings
    line_height = 25
    padding = 10
    max_width = 0
    
    # Calculate panel dimensions
    for label, value in info_dict.items():
        text = f"{label}: {value}"
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        max_width = max(max_width, text_width)
    
    panel_width = max_width + 2 * padding
    panel_height = len(info_dict) * line_height + 2 * padding
    
    # Determine panel position
    if position == 'top-left':
        x, y = 10, 10
    elif position == 'top-right':
        x, y = width - panel_width - 10, 10
    elif position == 'bottom-left':
        x, y = 10, height - panel_height - 10
    else:  # bottom-right
        x, y = width - panel_width - 10, height - panel_height - 10
    
    # Draw semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (255, 255, 255), 2)
    
    # Draw text
    text_y = y + padding + 15
    for label, value in info_dict.items():
        text = f"{label}: {value}"
        cv2.putText(frame, text, (x + padding, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        text_y += line_height
    
    return frame


def get_team_color(team_id):
    """
    Get a color for a team based on team ID.
    
    Args:
        team_id (int): Team ID (1 or 2).
    
    Returns:
        tuple: BGR color tuple.
    """
    colors = {
        1: (255, 255, 255),  # White for Team 1
        2: (0, 100, 255),    # Orange for Team 2
    }
    return colors.get(team_id, (0, 255, 0))  # Default green


def get_class_color(class_name):
    """
    Get a color for a detection class.
    
    Args:
        class_name (str): Name of the class.
    
    Returns:
        tuple: BGR color tuple.
    """
    colors = {
        'player': (0, 255, 0),      # Green
        'ball': (0, 255, 255),      # Yellow
        'ball_carrier': (255, 0, 255),  # Magenta
    }
    return colors.get(class_name.lower(), (0, 255, 0))
