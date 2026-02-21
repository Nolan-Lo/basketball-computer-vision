"""Tactical-view coordinate transformer.

Maps player positions from broadcast-camera (video-frame) coordinates to
a standardised bird's-eye 2D court via homography, using detected court
keypoints as correspondences.

The 18 reference keypoints encode the official FIBA court geometry
(28 × 15 m) scaled to a fixed pixel canvas (300 × 161 px).

Pipeline integration
--------------------
Runs **after** court-keypoint detection, player tracking, and team
assignment, but **before** the tactical-view drawer.  It produces
per-frame ``{player_id: (x, y)}`` dictionaries that the drawer
overlays onto the output video.
"""

import numpy as np
import cv2
from copy import deepcopy

from .homography import Homography
from .bbox_utils import get_foot_position, measure_distance


class TacticalViewConverter:
    """Convert video-frame player positions to a fixed-size 2D court.

    Args:
        court_image_path (str): Path to the court background image used
            by :class:`TacticalViewDrawer`.
    """

    def __init__(self, court_image_path: str) -> None:
        self.court_image_path = court_image_path
        self.width = 300
        self.height = 161

        # Real-world court dimensions (FIBA)
        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15

        # 18 reference keypoints in tactical-view pixel coordinates.
        # The ordering **must** match the keypoint indices produced by the
        # court-keypoint YOLO Pose model.
        self.key_points = [
            # --- left edge (indices 0-5) ---
            (0, 0),
            (0, int((0.91 / self.actual_height_in_meters) * self.height)),
            (0, int((5.18 / self.actual_height_in_meters) * self.height)),
            (0, int((10 / self.actual_height_in_meters) * self.height)),
            (0, int((14.1 / self.actual_height_in_meters) * self.height)),
            (0, int(self.height)),
            # --- middle line (indices 6-7) ---
            (int(self.width / 2), self.height),
            (int(self.width / 2), 0),
            # --- left free-throw line (indices 8-9) ---
            (
                int((5.79 / self.actual_width_in_meters) * self.width),
                int((5.18 / self.actual_height_in_meters) * self.height),
            ),
            (
                int((5.79 / self.actual_width_in_meters) * self.width),
                int((10 / self.actual_height_in_meters) * self.height),
            ),
            # --- right edge (indices 10-15) ---
            (self.width, int(self.height)),
            (self.width, int((14.1 / self.actual_height_in_meters) * self.height)),
            (self.width, int((10 / self.actual_height_in_meters) * self.height)),
            (self.width, int((5.18 / self.actual_height_in_meters) * self.height)),
            (self.width, int((0.91 / self.actual_height_in_meters) * self.height)),
            (self.width, 0),
            # --- right free-throw line (indices 16-17) ---
            (
                int(
                    (
                        (self.actual_width_in_meters - 5.79)
                        / self.actual_width_in_meters
                    )
                    * self.width
                ),
                int((5.18 / self.actual_height_in_meters) * self.height),
            ),
            (
                int(
                    (
                        (self.actual_width_in_meters - 5.79)
                        / self.actual_width_in_meters
                    )
                    * self.width
                ),
                int((10 / self.actual_height_in_meters) * self.height),
            ),
        ]

    # ------------------------------------------------------------------
    # Keypoint validation
    # ------------------------------------------------------------------

    def validate_keypoints(self, keypoints_list):
        """Filter out mis-detected keypoints by checking distance proportions.

        For every detected keypoint the ratio of its distance to two
        reference neighbours is compared against the expected ratio from
        the tactical-view key-point layout.  If the error exceeds 80 %,
        the keypoint is zeroed out so the homography ignores it.

        Args:
            keypoints_list (list[list[tuple]]): Per-frame keypoints as
                ``[(x, y, conf), ...]``.  A keypoint with ``x == 0`` and
                ``y == 0`` is considered undetected.

        Returns:
            list[list[tuple]]: Cleaned keypoints (same structure, invalid
            entries replaced with ``(0, 0, 0)``).
        """
        keypoints_list = deepcopy(keypoints_list)

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            # Indices where the model actually detected something
            detected_indices = [
                i
                for i, kp in enumerate(frame_keypoints)
                if kp[0] > 0 and kp[1] > 0
            ]

            if len(detected_indices) < 3:
                continue

            invalid_keypoints = []

            for i in detected_indices:
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue

                other_indices = [
                    idx
                    for idx in detected_indices
                    if idx != i and idx not in invalid_keypoints
                ]
                if len(other_indices) < 2:
                    continue

                j, k = other_indices[0], other_indices[1]

                d_ij = measure_distance(
                    frame_keypoints[i][:2], frame_keypoints[j][:2]
                )
                d_ik = measure_distance(
                    frame_keypoints[i][:2], frame_keypoints[k][:2]
                )

                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float("inf")
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float("inf")

                    error = abs(
                        (prop_detected - prop_tactical) / prop_tactical
                    )

                    if error > 0.8:
                        # Zero-out the unreliable keypoint
                        keypoints_list[frame_idx][i] = (0.0, 0.0, 0.0)
                        invalid_keypoints.append(i)

        return keypoints_list

    # ------------------------------------------------------------------
    # Player projection
    # ------------------------------------------------------------------

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        """Project player foot-positions to the tactical-view court.

        Args:
            keypoints_list (list[list[tuple]]): Per-frame court keypoints
                as ``[(x, y, conf), ...]``.
            player_tracks (list[dict]): Per-frame dicts mapping
                ``track_id → {"bbox": [x1, y1, x2, y2]}``.

        Returns:
            list[dict]: Per-frame dicts mapping
            ``track_id → [x, y]`` in tactical-view pixels.
        """
        tactical_player_positions = []

        for frame_keypoints, frame_tracks in zip(keypoints_list, player_tracks):
            tactical_positions = {}

            if not frame_keypoints:
                tactical_player_positions.append(tactical_positions)
                continue

            # Keep only detected keypoints (x > 0 and y > 0)
            valid_indices = [
                i
                for i, kp in enumerate(frame_keypoints)
                if kp[0] > 0 and kp[1] > 0
            ]

            # Need >= 4 points for a reliable homography
            if len(valid_indices) < 4:
                tactical_player_positions.append(tactical_positions)
                continue

            source_points = np.array(
                [frame_keypoints[i][:2] for i in valid_indices], dtype=np.float32
            )
            target_points = np.array(
                [self.key_points[i] for i in valid_indices], dtype=np.float32
            )

            try:
                homography = Homography(source_points, target_points)

                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    player_position = np.array([get_foot_position(bbox)])
                    tactical_position = homography.transform_points(
                        player_position
                    )

                    tx, ty = tactical_position[0]
                    # Discard positions projected outside the court bounds
                    if 0 <= tx <= self.width and 0 <= ty <= self.height:
                        tactical_positions[player_id] = tactical_position[
                            0
                        ].tolist()

            except (ValueError, cv2.error):
                pass  # homography failed — empty dict for this frame

            tactical_player_positions.append(tactical_positions)

        return tactical_player_positions
