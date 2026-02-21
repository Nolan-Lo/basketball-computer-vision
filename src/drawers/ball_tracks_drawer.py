"""Drawer for ball tracking visualizations.

Renders a coloured triangle pointer above the detected ball position
on each frame.
"""

from .utils import draw_triangle


class BallTracksDrawer:
    """
    Draw ball pointers on video frames.

    Attributes:
        ball_pointer_color (tuple): BGR colour for the ball triangle.
    """

    def __init__(self):
        self.ball_pointer_color = (0, 255, 0)  # green

    def draw(self, video_frames, tracks):
        """
        Draw ball pointers on each frame.

        Args:
            video_frames (list): Frames (NumPy arrays) to annotate.
            tracks (list[dict]): Per-frame ``{1: {"bbox": [x1,y1,x2,y2]}}``
                (key ``1`` is the single ball ID).

        Returns:
            list: Annotated copies of the input frames.
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            ball_dict = tracks[frame_num]

            for _, ball in ball_dict.items():
                if ball.get("bbox") is None:
                    continue
                frame = draw_triangle(frame, ball["bbox"], self.ball_pointer_color)

            output_video_frames.append(frame)

        return output_video_frames
