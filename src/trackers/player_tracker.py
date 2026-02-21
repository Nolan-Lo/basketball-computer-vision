"""Player tracker module using YOLO detection with ByteTrack tracking.

Provides persistent player identity tracking across video frames by combining
YOLO object detection with the ByteTrack multi-object tracking algorithm from
the supervision library.
"""

from ultralytics import YOLO
import supervision as sv

from ..utils import read_stub, save_stub


class PlayerTracker:
    """
    Handles player detection and tracking using YOLO and ByteTrack.

    Combines YOLO object detection with ByteTrack tracking to maintain consistent
    player identities across frames while processing detections in batches.

    Attributes:
        model (YOLO): The YOLO detection model.
        tracker (sv.ByteTrack): The ByteTrack multi-object tracker.
    """

    def __init__(self, model_path):
        """
        Initialize the PlayerTracker with a YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames, batch_size=20, conf=0.5):
        """
        Detect players in a sequence of frames using batch processing.

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
        Get player tracking results for a sequence of frames with optional caching.

        Runs YOLO detection on all frames, then applies ByteTrack to assign
        consistent track IDs to each detected player across frames.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries per frame, where each dictionary maps
                  track_id (int) to ``{"bbox": [x1, y1, x2, y2]}``.
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

            # Track objects with ByteTrack
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks.append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv.get('Player', cls_names_inv.get('player', -1)):
                    tracks[frame_num][track_id] = {"bbox": bbox}

        save_stub(stub_path, tracks)
        return tracks
