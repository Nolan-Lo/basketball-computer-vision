"""Ball acquisition (possession) detection for basketball video analysis.

Determines which player is most likely in possession of the ball on each frame
by analysing bounding-box proximity and containment between detected players
and the tracked ball position.
"""

from .bbox_utils import measure_distance, get_center_of_bbox


class BallAcquisitionDetector:
    """
    Detects ball acquisition by players in a basketball game.

    Combines distance measurements between the ball and key points of each
    player's bounding box with containment ratios of the ball within a
    player's bounding box to decide who has possession.

    Attributes:
        possession_threshold (int): Maximum pixel distance for possession.
        min_frames (int): Consecutive frames required to confirm possession.
        containment_threshold (float): Ball-in-player bbox ratio for auto-possession.
    """

    def __init__(self, possession_threshold=50, min_frames=11, containment_threshold=0.8):
        """
        Initialize the BallAcquisitionDetector.

        Args:
            possession_threshold (int): Maximum distance (px) at which a player
                can be considered to have the ball if containment is insufficient.
            min_frames (int): Minimum consecutive frames required for a player
                to be confirmed as having possession.
            containment_threshold (float): Containment ratio above which a player
                is considered to hold the ball without requiring distance checking.
        """
        self.possession_threshold = possession_threshold
        self.min_frames = min_frames
        self.containment_threshold = containment_threshold

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        """
        Compute key points around a player's bounding box for distance checks.

        Includes adaptive edge-projection points (where a horizontal/vertical
        line from the ball center would intersect the bbox edges) as well as
        fixed structural points (corners, midpoints, etc.).

        Args:
            player_bbox (tuple): Bounding box ``(x1, y1, x2, y2)``.
            ball_center (tuple): ``(x, y)`` of the ball center.

        Returns:
            list[tuple]: ``(x, y)`` key points around the bounding box.
        """
        ball_center_x, ball_center_y = ball_center
        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1

        output_points = []

        # Adaptive edge projections
        if y1 < ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))
        if x1 < ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))

        # Fixed structural points
        output_points += [
            (x1 + width // 2, y1),              # top centre
            (x2, y1),                            # top right
            (x1, y1),                            # top left
            (x2, y1 + height // 2),              # centre right
            (x1, y1 + height // 2),              # centre left
            (x1 + width // 2, y1 + height // 2), # centre point
            (x2, y2),                            # bottom right
            (x1, y2),                            # bottom left
            (x1 + width // 2, y2),              # bottom centre
            (x1 + width // 2, y1 + height // 3), # mid-top centre
        ]
        return output_points

    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):
        """
        Fraction of the ball bounding box that is inside the player bbox.

        Args:
            player_bbox (tuple): ``(x1, y1, x2, y2)`` for the player.
            ball_bbox (tuple): ``(x1, y1, x2, y2)`` for the ball.

        Returns:
            float: 0.0 – 1.0 containment ratio.
        """
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        ix1 = max(px1, bx1)
        iy1 = max(py1, by1)
        ix2 = min(px2, bx2)
        iy2 = min(py2, by2)

        if ix2 < ix1 or iy2 < iy1:
            return 0.0

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        ball_area = (bx2 - bx1) * (by2 - by1)

        if ball_area == 0:
            return 0.0

        return intersection_area / ball_area

    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        """
        Minimum distance from any key point on a player bbox to the ball centre.

        Args:
            ball_center (tuple): ``(x, y)`` of the ball centre.
            player_bbox (tuple): ``(x1, y1, x2, y2)`` for the player.

        Returns:
            float: Smallest distance from the ball to any key point.
        """
        key_points = self.get_key_basketball_player_assignment_points(player_bbox, ball_center)
        return min(measure_distance(ball_center, point) for point in key_points)

    # ------------------------------------------------------------------
    # Per-frame candidate selection
    # ------------------------------------------------------------------

    def find_best_candidate_for_possession(self, ball_center, player_tracks_frame, ball_bbox):
        """
        Determine which player in a single frame most likely has the ball.

        Priority:
        1. Players whose bbox contains most of the ball (high containment).
        2. The closest player within ``possession_threshold`` pixels.

        Args:
            ball_center (tuple): ``(x, y)`` of the ball centre.
            player_tracks_frame (dict): ``{player_id: {"bbox": [...]}, ...}``.
            ball_bbox (tuple): ``(x1, y1, x2, y2)`` for the ball.

        Returns:
            int: ``player_id`` of the best candidate, or ``-1`` if none found.
        """
        high_containment_players = []
        regular_distance_players = []

        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get('bbox', [])
            if not player_bbox:
                continue

            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            min_distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)

            if containment > self.containment_threshold:
                high_containment_players.append((player_id, min_distance))
            else:
                regular_distance_players.append((player_id, min_distance))

        # First priority: players with high containment
        if high_containment_players:
            best_candidate = max(high_containment_players, key=lambda x: x[1])
            return best_candidate[0]

        # Second priority: closest player within threshold
        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda x: x[1])
            if best_candidate[1] < self.possession_threshold:
                return best_candidate[0]

        return -1

    # ------------------------------------------------------------------
    # Full-video possession detection
    # ------------------------------------------------------------------

    def detect_ball_possession(self, player_tracks, ball_tracks):
        """
        Detect which player has the ball in every frame.

        A player must hold the ball for at least ``min_frames`` consecutive
        frames before possession is confirmed.

        Args:
            player_tracks (list[dict]): Per-frame ``{player_id: {"bbox": [...]}}``
                as returned by ``PlayerTracker.get_object_tracks()``.
            ball_tracks (list[dict]): Per-frame ``{1: {"bbox": [...]}}``
                as returned by ``BallTracker`` after interpolation.

        Returns:
            list[int]: Per-frame player_id with possession, or ``-1``.
        """
        num_frames = len(ball_tracks)
        possession_list = [-1] * num_frames
        consecutive_possession_count = {}

        for frame_num in range(num_frames):
            ball_info = ball_tracks[frame_num].get(1, {})
            if not ball_info:
                continue

            ball_bbox = ball_info.get('bbox', [])
            if not ball_bbox:
                continue

            ball_center = get_center_of_bbox(ball_bbox)

            best_player_id = self.find_best_candidate_for_possession(
                ball_center,
                player_tracks[frame_num],
                ball_bbox
            )

            if best_player_id != -1:
                count = consecutive_possession_count.get(best_player_id, 0) + 1
                consecutive_possession_count = {best_player_id: count}

                if count >= self.min_frames:
                    possession_list[frame_num] = best_player_id
            else:
                consecutive_possession_count = {}

        return possession_list
