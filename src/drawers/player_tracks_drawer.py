"""Drawer for player tracking visualizations.

Renders team-colored ellipses under each tracked player and a triangle
pointer above whichever player currently possesses the ball.
"""

from .utils import draw_ellipse, draw_triangle


class PlayerTracksDrawer:
    """
    Draw player tracks and ball-carrier indicators on video frames.

    Attributes:
        default_player_team_id (int): Fallback team ID when assignment is missing.
        team_1_color (list): BGR color for Team 1 players.
        team_2_color (list): BGR color for Team 2 players.
    """

    def __init__(self, team_1_color=None, team_2_color=None):
        """
        Initialize with optional custom team colors.

        Args:
            team_1_color (list, optional): BGR color for Team 1.
                Defaults to ``[255, 245, 238]`` (seashell / light).
            team_2_color (list, optional): BGR color for Team 2.
                Defaults to ``[128, 0, 0]`` (maroon / dark).
        """
        self.default_player_team_id = 1
        self.team_1_color = team_1_color or [255, 245, 238]
        self.team_2_color = team_2_color or [128, 0, 0]

    def draw(self, video_frames, tracks, player_assignment, ball_acquisition):
        """
        Draw player tracks and ball possession indicators on frames.

        Args:
            video_frames (list): Frames (NumPy arrays) to annotate.
            tracks (list[dict]): Per-frame ``{track_id: {"bbox": [x1,y1,x2,y2]}}``.
            player_assignment (list[dict]): Per-frame ``{track_id: team_id}``.
            ball_acquisition (list[int]): Per-frame possessing player ID (``-1`` = none).

        Returns:
            list: Annotated copies of the input frames.
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks[frame_num]
            assignment = player_assignment[frame_num]
            player_id_has_ball = ball_acquisition[frame_num]

            for track_id, player in player_dict.items():
                team_id = assignment.get(track_id, self.default_player_team_id)
                color = self.team_1_color if team_id == 1 else self.team_2_color

                frame = draw_ellipse(frame, player["bbox"], color, track_id)

                # Red triangle above the ball carrier
                if track_id == player_id_has_ball:
                    frame = draw_triangle(frame, player["bbox"], (0, 0, 255))

            output_video_frames.append(frame)

        return output_video_frames
