"""Drawer for team ball-control statistics.

Calculates cumulative ball-control percentages and renders them as a
semi-transparent overlay in the bottom-right area of each frame.
"""

import cv2
import numpy as np


class TeamBallControlDrawer:
    """
    Calculate and draw team ball-control statistics on video frames.
    """

    def get_team_ball_control(self, player_assignment, ball_acquisition):
        """
        Map per-frame ball possession to a team ID.

        Args:
            player_assignment (list[dict]): Per-frame ``{track_id: team_id}``.
            ball_acquisition (list[int]): Per-frame possessing player ID
                (``-1`` = no possession).

        Returns:
            numpy.ndarray: Per-frame team label (1, 2, or -1).
        """
        team_ball_control = []

        for assignment, possessing_player in zip(player_assignment, ball_acquisition):
            if possessing_player == -1:
                team_ball_control.append(-1)
                continue
            if possessing_player not in assignment:
                team_ball_control.append(-1)
                continue
            team_ball_control.append(assignment[possessing_player])

        return np.array(team_ball_control)

    def draw(self, video_frames, player_assignment, ball_acquisition):
        """
        Draw cumulative ball-control percentages on each frame.

        Args:
            video_frames (list): Frames (NumPy arrays) to annotate.
            player_assignment (list[dict]): Per-frame ``{track_id: team_id}``.
            ball_acquisition (list[int]): Per-frame possessing player ID.

        Returns:
            list: Annotated copies of the input frames.
        """
        team_ball_control = self.get_team_ball_control(
            player_assignment, ball_acquisition
        )

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame_drawn = self._draw_frame(frame, frame_num, team_ball_control)
            output_video_frames.append(frame_drawn)

        return output_video_frames

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _draw_frame(self, frame, frame_num, team_ball_control):
        """Render the ball-control overlay on a single frame."""
        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2

        # Overlay position (bottom-right region)
        h, w = overlay.shape[:2]
        rect_x1 = int(w * 0.60)
        rect_y1 = int(h * 0.75)
        rect_x2 = int(w * 0.99)
        rect_y2 = int(h * 0.90)

        text_x = int(w * 0.63)
        text_y1 = int(h * 0.80)
        text_y2 = int(h * 0.88)

        # Semi-transparent white rectangle
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2),
                       (255, 255, 255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Cumulative control percentages up to current frame
        control_slice = team_ball_control[: frame_num + 1]
        total = control_slice.shape[0]

        team_1_pct = (control_slice == 1).sum() / total * 100 if total else 0.0
        team_2_pct = (control_slice == 2).sum() / total * 100 if total else 0.0

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1_pct:.2f}%",
            (text_x, text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2_pct:.2f}%",
            (text_x, text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

        return frame
