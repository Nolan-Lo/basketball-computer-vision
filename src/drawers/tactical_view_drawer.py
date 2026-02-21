"""Tactical-view overlay drawer.

Renders a semi-transparent bird's-eye court image on each video frame
and plots player dots (coloured by team) and a ball-carrier ring.

This drawer is the visual counterpart of
:class:`~src.tactical_view_converter.TacticalViewConverter`.
"""

import cv2


class TacticalViewDrawer:
    """Draw the tactical mini-map overlay.

    Args:
        team_1_color (list): BGR colour for Team 1 dots.
        team_2_color (list): BGR colour for Team 2 dots.
    """

    def __init__(
        self,
        team_1_color=None,
        team_2_color=None,
    ):
        # Position of the overlay on the output frame
        self.start_x = 20
        self.start_y = 40

        self.team_1_color = team_1_color or [255, 245, 238]   # seashell / light
        self.team_2_color = team_2_color or [128, 0, 0]       # maroon / dark

    def draw(
        self,
        video_frames,
        court_image_path,
        width,
        height,
        tactical_court_keypoints,
        tactical_player_positions=None,
        player_assignment=None,
        ball_acquisition=None,
    ):
        """Composite the tactical view onto every frame.

        Args:
            video_frames (list): Video frames to annotate.
            court_image_path (str): Path to the court background PNG/JPG.
            width (int): Pixel width of the tactical view.
            height (int): Pixel height of the tactical view.
            tactical_court_keypoints (list[tuple]): Reference keypoints
                to draw on the mini-court.
            tactical_player_positions (list[dict] | None): Per-frame
                ``{player_id: [x, y]}`` from the converter.
            player_assignment (list[dict] | None): Per-frame
                ``{player_id: team_id}``.
            ball_acquisition (list[int] | None): Per-frame possessing
                player id (-1 = none).

        Returns:
            list: Annotated frames.
        """
        court_image = cv2.imread(court_image_path)
        court_image = cv2.resize(court_image, (width, height))

        output_video_frames = []

        for frame_idx, frame in enumerate(video_frames):
            frame = frame.copy()

            y1 = self.start_y
            y2 = self.start_y + height
            x1 = self.start_x
            x2 = self.start_x + width

            # Semi-transparent court overlay
            alpha = 0.6
            overlay = frame[y1:y2, x1:x2].copy()
            cv2.addWeighted(
                court_image, alpha, overlay, 1 - alpha, 0, frame[y1:y2, x1:x2]
            )

            # Court reference keypoints (small red dots with index labels)
            for kp_idx, keypoint in enumerate(tactical_court_keypoints):
                kx, ky = keypoint
                kx += self.start_x
                ky += self.start_y
                cv2.circle(frame, (kx, ky), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(kp_idx),
                    (kx, ky),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Player dots
            if (
                tactical_player_positions
                and player_assignment
                and frame_idx < len(tactical_player_positions)
            ):
                frame_positions = tactical_player_positions[frame_idx]
                frame_assignments = (
                    player_assignment[frame_idx]
                    if frame_idx < len(player_assignment)
                    else {}
                )
                player_with_ball = (
                    ball_acquisition[frame_idx]
                    if ball_acquisition and frame_idx < len(ball_acquisition)
                    else -1
                )

                for player_id, position in frame_positions.items():
                    team_id = frame_assignments.get(player_id, 1)
                    color = (
                        self.team_1_color if team_id == 1 else self.team_2_color
                    )

                    px = int(position[0]) + self.start_x
                    py = int(position[1]) + self.start_y

                    player_radius = 8
                    cv2.circle(frame, (px, py), player_radius, color, -1)

                    # Ring around ball carrier
                    if player_id == player_with_ball:
                        cv2.circle(
                            frame, (px, py), player_radius + 3, (0, 0, 255), 2
                        )

            output_video_frames.append(frame)

        return output_video_frames
