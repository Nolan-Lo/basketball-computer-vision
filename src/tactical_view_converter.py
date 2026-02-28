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

# ---------------------------------------------------------------------------
# Court keypoint constants
# ---------------------------------------------------------------------------

# Mirror map: each keypoint index → its symmetric left↔right counterpart.
MIRROR_MAP = {
    0: 15, 1: 14, 2: 13, 3: 12, 4: 11, 5: 10,
    10: 5, 11: 4, 12: 3, 13: 2, 14: 1, 15: 0,
    8: 16, 9: 17, 16: 8, 17: 9,
}

# Structural groupings (independent of left/right labelling).
BASELINE_INDICES = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15}
ELBOW_INDICES = {8, 9, 16, 17}
MIDCOURT_INDICES = {6, 7}

# Valid keypoint indices for each court half.
LEFT_VALID = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
RIGHT_VALID = {6, 7, 10, 11, 12, 13, 14, 15, 16, 17}


class TacticalViewConverter:
    """Convert video-frame player positions to a fixed-size 2D court.

    Args:
        court_image_path (str): Path to the court background image used
            by :class:`TacticalViewDrawer`.
    """

    def __init__(self, court_image_path: str) -> None:
        self.court_image_path = court_image_path
        self.width = 300
        self.height = int(self.width * 15.24 / 28.65)  # 160 — preserves NBA aspect ratio

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
    # Court-side detection & keypoint enforcement
    # ------------------------------------------------------------------

    def _determine_court_side(self, keypoints_list, sample_frames=None):
        """Determine which half of the court is visible.

        Compares the average video-frame x-position of *baseline*-type
        keypoints (indices 0-5 / 10-15) against *elbow*-type keypoints
        (indices 8-9 / 16-17) aggregated across several frames.

        On a **left** half-court view the baseline is at the left edge
        of the frame and the free-throw elbows are further right
        (toward midcourt).  The converse holds for a **right** view.

        The method groups keypoints by structural type regardless of
        which specific index the model assigned, so it is robust to
        left ↔ right mis-labelling by the detector.

        Args:
            keypoints_list (list[list[tuple]]): Per-frame keypoints
                ``[(x, y, conf), ...]``.
            sample_frames (int | None): Number of frames to inspect.
                *None* uses every frame.

        Returns:
            str: ``"left"``, ``"right"``, or ``"unknown"``.
        """
        if sample_frames is None:
            sample_frames = len(keypoints_list)

        baseline_xs: list[float] = []
        elbow_xs: list[float] = []

        for frame_kps in keypoints_list[:sample_frames]:
            if not frame_kps:
                continue
            for i, kp in enumerate(frame_kps):
                if kp[0] <= 0 or kp[1] <= 0:
                    continue
                if i in BASELINE_INDICES:
                    baseline_xs.append(kp[0])
                elif i in ELBOW_INDICES:
                    elbow_xs.append(kp[0])

        if not baseline_xs or not elbow_xs:
            return "unknown"

        avg_baseline_x = sum(baseline_xs) / len(baseline_xs)
        avg_elbow_x = sum(elbow_xs) / len(elbow_xs)

        # Require a meaningful pixel separation to be confident.
        if abs(avg_baseline_x - avg_elbow_x) < 50:
            return "unknown"

        return "left" if avg_baseline_x < avg_elbow_x else "right"

    def _enforce_court_side(self, frame_keypoints, court_side):
        """Remap or zero keypoints that belong to the wrong court half.

        Given the determined *court_side*:

        * **left** → only indices in ``LEFT_VALID`` are allowed.
        * **right** → only indices in ``RIGHT_VALID`` are allowed.

        A detected keypoint whose index falls outside the valid set is
        *remapped* to its mirror index when the mirror slot is empty,
        or discarded (zeroed) when the mirror slot is already occupied
        (the duplicate / conflicting case such as 8 **and** 16 both
        being predicted at the same physical location).

        Args:
            frame_keypoints (list[tuple]): Mutable per-frame keypoints.
            court_side (str): ``"left"``, ``"right"``, or ``"unknown"``.

        Returns:
            list[tuple]: The (possibly modified) keypoints.
        """
        if court_side == "unknown" or len(frame_keypoints) < 18:
            return frame_keypoints

        valid_set = LEFT_VALID if court_side == "left" else RIGHT_VALID

        for i in range(18):
            if frame_keypoints[i][0] <= 0 or frame_keypoints[i][1] <= 0:
                continue

            if i in valid_set:
                continue  # correctly indexed for this court side

            # Wrong-side keypoint — try to remap to its mirror
            mirror = MIRROR_MAP.get(i)
            if mirror is not None and mirror in valid_set:
                mirror_empty = (
                    frame_keypoints[mirror][0] <= 0
                    or frame_keypoints[mirror][1] <= 0
                )
                if mirror_empty:
                    frame_keypoints[mirror] = frame_keypoints[i]
                # else: mirror slot already occupied → discard duplicate

            # Zero the wrong-side index in all cases
            frame_keypoints[i] = (0.0, 0.0, 0.0)

        return frame_keypoints

    def validate_keypoints(self, keypoints_list):
        """Filter out mis-detected keypoints using court-side enforcement
        and geometric heuristics.

        Validation steps:

        0. **Court-side determination** (once) – analyse all frames to
           decide whether the camera shows the left or right half-court.
        1. **Court-side enforcement** – per frame, remap any keypoint
           whose index belongs to the opposite half to its mirror, or
           zero it when the mirror slot is already filled.
        2. **Proportion check** – for every detected keypoint the ratio
           of its distance to two reference neighbours is compared
           against the expected ratio from the tactical-view key-point
           layout.  If the error exceeds 60 % the keypoint is zeroed.
        3. **Pairwise ordering check** – left-edge keypoints should
           have smaller x than right-edge keypoints in the image.
        4. **Frame-to-frame jump filter** – if a keypoint teleports
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

        # --- Step 0: determine court side (once for whole video) ---
        court_side = self._determine_court_side(keypoints_list)
        print(f"  Court side detected: {court_side}")

        # Pairs of keypoint indices that should maintain a spatial
        # ordering in the video frame (left < right in x).
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

            # --- Step 1: enforce court side ---
            keypoints_list[frame_idx] = self._enforce_court_side(
                keypoints_list[frame_idx], court_side
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

            # --- Step 2: proportion check ---
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

            # --- Step 3: pairwise ordering check ---
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

            # --- Step 4: frame-to-frame jump filter ---
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

        When the current frame has ≥ 4 valid keypoints a fresh homography
        is computed.  Otherwise the **last successful homography** is
        reused for up to ``max_reuse_frames`` frames.  This keeps the
        tactical map populated through brief keypoint dropouts without
        carrying forward stale pixel positions (which break when the
        camera pans).

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
        last_homography = None
        frames_since_good = 0
        max_reuse_frames = 10  # reuse for up to ~0.3 s at 30 fps

        for frame_keypoints, frame_tracks in zip(keypoints_list, player_tracks):
            tactical_positions = {}
            homography = None

            if frame_keypoints:
                # Keep only detected keypoints (x > 0 and y > 0)
                valid_indices = [
                    i
                    for i, kp in enumerate(frame_keypoints)
                    if kp[0] > 0 and kp[1] > 0
                ]

                # Need >= 4 points for a reliable homography
                if len(valid_indices) >= 4:
                    source_points = np.array(
                        [frame_keypoints[i][:2] for i in valid_indices],
                        dtype=np.float32,
                    )
                    target_points = np.array(
                        [self.key_points[i] for i in valid_indices],
                        dtype=np.float32,
                    )

                    try:
                        homography = Homography(source_points, target_points)
                        last_homography = homography
                        frames_since_good = 0
                    except (ValueError, cv2.error):
                        homography = None

            # Fall back to last good homography if current frame failed
            if homography is None and last_homography is not None:
                frames_since_good += 1
                if frames_since_good <= max_reuse_frames:
                    homography = last_homography

            # Project players through whichever homography we have
            if homography is not None:
                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    player_position = np.array([get_foot_position(bbox)])
                    try:
                        tactical_position = homography.transform_points(
                            player_position
                        )
                        tx, ty = tactical_position[0]
                        # Discard positions projected outside the court
                        if 0 <= tx <= self.width and 0 <= ty <= self.height:
                            tactical_positions[player_id] = (
                                tactical_position[0].tolist()
                            )
                    except (ValueError, cv2.error):
                        pass

            tactical_player_positions.append(tactical_positions)

        return tactical_player_positions
