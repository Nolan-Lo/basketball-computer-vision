"""Tactical-view coordinate transformer.

Maps player positions from broadcast-camera (video-frame) coordinates to
a standardised bird's-eye 2D court via homography, using detected court
keypoints as correspondences.

The 18 reference keypoints encode the official NBA court geometry
(94 ft × 50 ft / 28.65 × 15.24 m) scaled to a fixed pixel canvas (300 × 161 px).

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

        # Real-world court dimensions (NBA)
        self.actual_width_in_meters = 28.65   # 94 ft
        self.actual_height_in_meters = 15.24  # 50 ft

        # 18 reference keypoints in tactical-view pixel coordinates.
        # The ordering **must** match the keypoint indices produced by the
        # court-keypoint YOLO Pose model.
        #
        # NBA-specific measurements used:
        #   - Free-throw line: 5.79 m (19 ft) from baseline
        #   - Lane width: 4.88 m (16 ft), centered → edges at 5.18 m and 10.06 m
        #   - Sideline mark: 0.91 m (~3 ft); mirror at 14.33 m
        self.key_points = [
            # --- left edge (indices 0-5) ---
            (0, 0),
            (0, int((0.91 / self.actual_height_in_meters) * self.height)),
            (0, int((5.18 / self.actual_height_in_meters) * self.height)),
            (0, int((10.06 / self.actual_height_in_meters) * self.height)),
            (0, int((14.33 / self.actual_height_in_meters) * self.height)),
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
                int((10.06 / self.actual_height_in_meters) * self.height),
            ),
            # --- right edge (indices 10-15) ---
            (self.width, int(self.height)),
            (self.width, int((14.33 / self.actual_height_in_meters) * self.height)),
            (self.width, int((10.06 / self.actual_height_in_meters) * self.height)),
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
                int((10.06 / self.actual_height_in_meters) * self.height),
            ),
        ]

    # ------------------------------------------------------------------
    # Keypoint validation
    # ------------------------------------------------------------------

    def _remap_mirrored_keypoints(self, frame_keypoints):
        """Remap left/right symmetric keypoints when spatially inconsistent.

        The YOLO Pose model sometimes assigns a keypoint the index of
        its mirror counterpart (e.g. right-side baseline point labelled
        as a left-side index).  This method uses a centre-x reference
        derived from midcourt or the average of detected anchors to
        detect and correct every left ↔ right mislabelling.

        Symmetric pairs (left_idx, right_idx)::

            Baselines:   (0, 15)  (1, 14)  (2, 13)  (3, 12)  (4, 11)  (5, 10)
            Elbows:      (8, 16)  (9, 17)

        Within each pair the left index should have a *smaller* x in the
        video frame than the right index.

        Args:
            frame_keypoints (list[tuple]): Mutable list of
                ``(x, y, conf)`` for a single frame (length 18).

        Returns:
            list[tuple]: The (possibly modified) keypoints.
        """
        if len(frame_keypoints) < 18:
            return frame_keypoints

        # All left ↔ right symmetric pairs
        mirror_pairs = [
            (0, 15),   # top-left corner      ↔ top-right corner
            (1, 14),   # left sideline mark 1  ↔ right sideline mark 1
            (2, 13),   # left lane top         ↔ right lane top
            (3, 12),   # left lane bottom      ↔ right lane bottom
            (4, 11),   # left sideline mark 2  ↔ right sideline mark 2
            (5, 10),   # bottom-left corner    ↔ bottom-right corner
            (8, 16),   # left FT elbow top     ↔ right FT elbow top
            (9, 17),   # left FT elbow bottom  ↔ right FT elbow bottom
        ]

        # --- Determine a reference centre-x from visible anchors ---
        #
        # Priority:
        #   1. Midcourt line (keypoints 6, 7) — most reliable.
        #   2. Baseline ↔ elbow spatial relationship:
        #      • Left court: baseline x  <  elbow x  → centre is RIGHT
        #        of everything visible.
        #      • Right court: baseline x  >  elbow x  → centre is LEFT
        #        of everything visible.
        #   3. Give up — cannot determine side.
        mid_xs = [
            frame_keypoints[i][0]
            for i in (6, 7)
            if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0
        ]

        if mid_xs:
            center_x = sum(mid_xs) / len(mid_xs)
        else:
            # Collect x-coords for any detected baseline and elbow points
            # (regardless of which index the model assigned).
            left_baseline_indices = range(0, 6)
            right_baseline_indices = range(10, 16)
            left_elbow_indices = (8, 9)
            right_elbow_indices = (16, 17)

            baseline_xs = [
                frame_keypoints[i][0]
                for i in (*left_baseline_indices, *right_baseline_indices)
                if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0
            ]
            elbow_xs = [
                frame_keypoints[i][0]
                for i in (*left_elbow_indices, *right_elbow_indices)
                if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0
            ]

            if baseline_xs and elbow_xs:
                avg_baseline_x = sum(baseline_xs) / len(baseline_xs)
                avg_elbow_x = sum(elbow_xs) / len(elbow_xs)

                # Collect all detected x values to find the extent
                all_detected_xs = [
                    frame_keypoints[i][0]
                    for i in range(18)
                    if i not in (6, 7)
                    and frame_keypoints[i][0] > 0
                    and frame_keypoints[i][1] > 0
                ]

                if avg_baseline_x < avg_elbow_x:
                    # Left court visible: baselines left of elbows.
                    # Place centre to the right of everything visible so
                    # all visible points are treated as "left-side".
                    center_x = max(all_detected_xs) + 100
                elif avg_baseline_x > avg_elbow_x:
                    # Right court visible: baselines right of elbows.
                    # Place centre to the left of everything visible so
                    # all visible points are treated as "right-side".
                    center_x = min(all_detected_xs) - 100
                else:
                    return frame_keypoints
            else:
                # Cannot determine court side — skip remapping
                return frame_keypoints

        # --- Check each symmetric pair ---
        for left_idx, right_idx in mirror_pairs:
            left_valid = (
                frame_keypoints[left_idx][0] > 0
                and frame_keypoints[left_idx][1] > 0
            )
            right_valid = (
                frame_keypoints[right_idx][0] > 0
                and frame_keypoints[right_idx][1] > 0
            )

            if left_valid and not right_valid:
                # Only "left" detected — but is it actually on the right?
                if frame_keypoints[left_idx][0] > center_x:
                    frame_keypoints[right_idx] = frame_keypoints[left_idx]
                    frame_keypoints[left_idx] = (0.0, 0.0, 0.0)

            elif right_valid and not left_valid:
                # Only "right" detected — but is it actually on the left?
                if frame_keypoints[right_idx][0] < center_x:
                    frame_keypoints[left_idx] = frame_keypoints[right_idx]
                    frame_keypoints[right_idx] = (0.0, 0.0, 0.0)

            elif left_valid and right_valid:
                # Both detected — swap if left has larger x than right
                if frame_keypoints[left_idx][0] > frame_keypoints[right_idx][0]:
                    frame_keypoints[left_idx], frame_keypoints[right_idx] = (
                        frame_keypoints[right_idx],
                        frame_keypoints[left_idx],
                    )

        return frame_keypoints

    def validate_keypoints(self, keypoints_list):
        """Filter out mis-detected keypoints using multiple heuristics.

        Validation steps applied **per frame**:
        0. **Mirror remap** – for every left/right symmetric keypoint
           pair, if the "left" index sits right of centre (or vice
           versa), the indices are swapped so the homography uses
           the correct correspondences.
        1. **Proportion check** – for every detected keypoint the ratio
           of its distance to two reference neighbours is compared
           against the expected ratio from the tactical-view key-point
           layout.  If the error exceeds 60 % the keypoint is zeroed.
        2. **Pairwise ordering check** – left-edge keypoints should
           have smaller x than right-edge keypoints in the image.
           Violations indicate a mis-detection (e.g. index 16 landing on
           top of index 8).
        3. **Frame-to-frame jump filter** – if a keypoint teleports
           more than ``max_jump_px`` pixels between consecutive frames
           it is zeroed, since real camera motion is continuous.

        Args:
            keypoints_list (list[list[tuple]]): Per-frame keypoints as
                ``[(x, y, conf), ...]``.  A keypoint with ``x == 0`` and
                ``y == 0`` is considered undetected.

        Returns:
            list[list[tuple]]: Cleaned keypoints (same structure; invalid
            entries replaced with ``(0, 0, 0)``).
        """
        keypoints_list = deepcopy(keypoints_list)

        # Pairs of keypoint indices that should maintain a spatial
        # ordering in the video frame (left < right in x).
        # Left-side elbows (8, 9) should have smaller x than right-side
        # elbows (16, 17).  Left baseline (0-5) < right baseline (10-15).
        ordered_pairs_x = [
            (8, 16),   # left FT elbow top  <  right FT elbow top
            (9, 17),   # left FT elbow bot  <  right FT elbow bot
            (0, 15),   # top-left corner    <  top-right corner
            (5, 10),   # bottom-left corner <  bottom-right corner
        ]

        max_jump_px = 120  # pixels – max plausible frame-to-frame shift

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            if not frame_keypoints:
                continue

            # --- Step 0: remap mis-indexed symmetric keypoints ---
            keypoints_list[frame_idx] = self._remap_mirrored_keypoints(
                keypoints_list[frame_idx]
            )
            frame_keypoints = keypoints_list[frame_idx]

            # Indices where the model actually detected something
            detected_indices = [
                i
                for i, kp in enumerate(frame_keypoints)
                if kp[0] > 0 and kp[1] > 0
            ]

            if len(detected_indices) < 3:
                # Too few points – zero them all to avoid a bad homography
                for i in detected_indices:
                    keypoints_list[frame_idx][i] = (0.0, 0.0, 0.0)
                continue

            invalid_keypoints = set()

            # --- Step 1: proportion check ---
            for i in detected_indices:
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

                    if error > 0.6:
                        keypoints_list[frame_idx][i] = (0.0, 0.0, 0.0)
                        invalid_keypoints.add(i)

            # --- Step 2: pairwise ordering check ---
            for left_idx, right_idx in ordered_pairs_x:
                if left_idx >= len(frame_keypoints) or right_idx >= len(frame_keypoints):
                    continue
                lkp = keypoints_list[frame_idx][left_idx]
                rkp = keypoints_list[frame_idx][right_idx]
                if lkp[0] > 0 and rkp[0] > 0 and lkp[0] >= rkp[0]:
                    # Left keypoint is NOT left of right → one is wrong;
                    # zero both to be safe.
                    keypoints_list[frame_idx][left_idx] = (0.0, 0.0, 0.0)
                    keypoints_list[frame_idx][right_idx] = (0.0, 0.0, 0.0)
                    invalid_keypoints.update({left_idx, right_idx})

            # --- Step 3: frame-to-frame jump filter ---
            if frame_idx > 0:
                prev_keypoints = keypoints_list[frame_idx - 1]
                for i in range(len(frame_keypoints)):
                    if i in invalid_keypoints:
                        continue
                    cur = keypoints_list[frame_idx][i]
                    if i >= len(prev_keypoints):
                        continue
                    prev = prev_keypoints[i]
                    if cur[0] > 0 and cur[1] > 0 and prev[0] > 0 and prev[1] > 0:
                        jump = measure_distance(cur[:2], prev[:2])
                        if jump > max_jump_px:
                            keypoints_list[frame_idx][i] = (0.0, 0.0, 0.0)
                            invalid_keypoints.add(i)

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
