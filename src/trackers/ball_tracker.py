"""Ball tracker module using YOLO detection with temporal filtering.

Provides ball detection and tracking across video frames, including
outlier removal and interpolation to fill gaps where the ball is
momentarily occluded or missed by the detector.
"""

from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd

from ..utils import read_stub, save_stub


class BallTracker:
    """
    Handles basketball detection and tracking using YOLO.

    Provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.

    Attributes:
        model (YOLO): The YOLO detection model.
    """

    def __init__(self, model_path):
        """
        Initialize the BallTracker with a YOLO model.

        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)

    def detect_frames(self, frames, batch_size=20, conf=0.5):
        """
        Detect the ball in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.
            batch_size (int): Number of frames to process per batch.
            conf (float): Confidence threshold for detections.

        Returns:
            list: YOLO detection results for each frame.
        """
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i + batch_size], conf=conf, verbose=False
            )
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get ball tracking results for a sequence of frames with optional caching.

        Picks the highest-confidence "Ball" detection in each frame.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries per frame. Each dictionary maps
                  key ``1`` to ``{"bbox": [x1, y1, x2, y2]}`` when the ball
                  is detected, or is empty ``{}`` when it is not.
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)

        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            chosen_bbox = None
            max_confidence = 0

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]

                if cls_id == cls_names_inv.get('Ball', cls_names_inv.get('ball', -1)):
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox": chosen_bbox}

        save_stub(stub_path, tracks)
        return tracks

    def remove_wrong_detections(self, ball_positions):
        """
        Filter out incorrect ball detections based on maximum allowed movement.

        If the ball jumps an unreasonably large distance between frames the
        detection is treated as a false positive and removed.

        Args:
            ball_positions (list): List of detected ball positions across frames.

        Returns:
            list: Filtered ball positions with outlier detections removed.
        """
        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            if np.linalg.norm(
                np.array(last_good_box[:2]) - np.array(current_box[:2])
            ) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing ball positions to create smooth tracking.

        Uses pandas linear interpolation and back-fill to fill gaps where
        the ball was not detected.

        Args:
            ball_positions (list): List of ball positions with potential gaps.

        Returns:
            list: Ball positions with interpolated values filling the gaps.
        """
        ball_positions_flat = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions_flat, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()
        ]
        return ball_positions
