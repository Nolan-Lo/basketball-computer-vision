"""Player tracker module using YOLO detection with ByteTrack tracking.

Provides persistent player identity tracking across video frames by combining
YOLO object detection with the ByteTrack multi-object tracking algorithm from
the supervision library.
"""

from ultralytics import YOLO
import supervision as sv
import pandas as pd

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

    def __init__(self, model_path, lost_track_buffer=60,
                 minimum_matching_threshold=0.5,
                 track_activation_threshold=0.25):
        """
        Initialize the PlayerTracker with a YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
            lost_track_buffer (int): Number of frames to keep a lost track
                alive before removing it. Higher values retain tracks
                through longer occlusions (default 60 ≈ 2 s at 30 fps).
            minimum_matching_threshold (float): IOU threshold for
                re-associating detections with existing tracks. Lower
                values make it easier to reconnect after overlap/occlusion
                (default 0.5).
            track_activation_threshold (float): Minimum detection confidence
                required to start a new track (default 0.25).
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            track_activation_threshold=track_activation_threshold,
        )

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

    def interpolate_player_tracks(self, player_tracks, max_gap=30):
        """
        Fill short gaps in player tracks with linearly-interpolated bboxes.

        When a player is temporarily lost (e.g. due to occlusion or a missed
        detection) for fewer than *max_gap* consecutive frames, this method
        fills the missing frames by linearly interpolating the bounding box
        coordinates from the last known position to the next known position.

        Args:
            player_tracks (list[dict]): Per-frame dicts mapping
                ``track_id -> {"bbox": [x1, y1, x2, y2]}``.
            max_gap (int): Maximum number of consecutive missing frames to
                interpolate. Gaps longer than this are left empty (the
                player likely left the frame).

        Returns:
            list[dict]: Player tracks with gaps filled.
        """
        # Collect all unique track IDs across the video.
        all_track_ids = set()
        for frame_tracks in player_tracks:
            all_track_ids.update(frame_tracks.keys())

        for track_id in all_track_ids:
            # Build a DataFrame of bbox values for this track.
            frames_data = []
            for frame_num, frame_tracks in enumerate(player_tracks):
                if track_id in frame_tracks:
                    bbox = frame_tracks[track_id]["bbox"]
                    frames_data.append(
                        {"frame": frame_num, "x1": bbox[0], "y1": bbox[1],
                         "x2": bbox[2], "y2": bbox[3]}
                    )

            if len(frames_data) < 2:
                continue

            df = pd.DataFrame(frames_data).set_index("frame")

            # Reindex to the full range this track spans.
            full_range = range(df.index.min(), df.index.max() + 1)
            df = df.reindex(full_range)

            # Identify gap lengths so we only interpolate short ones.
            is_missing = df["x1"].isna()
            gap_id = (~is_missing).cumsum()
            gap_lengths = is_missing.groupby(gap_id).transform("sum")

            # Interpolate all, then blank out gaps that are too long.
            df = df.interpolate(method="linear", limit_area="inside")
            df.loc[is_missing & (gap_lengths > max_gap)] = float("nan")

            # Write interpolated boxes back into player_tracks.
            for frame_num in df.index:
                row = df.loc[frame_num]
                if pd.notna(row["x1"]):
                    player_tracks[frame_num].setdefault(
                        track_id,
                        {"bbox": [row["x1"], row["y1"], row["x2"], row["y2"]]}
                    )

        return player_tracks
