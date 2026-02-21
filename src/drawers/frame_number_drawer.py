"""Drawer for frame number overlay.

Renders the current frame index in the top-left corner of each frame.
"""

import cv2


class FrameNumberDrawer:
    """
    Draw the frame number on each video frame.
    """

    def draw(self, frames):
        """
        Draw frame numbers on a list of frames.

        Args:
            frames (list): Video frames (NumPy arrays).

        Returns:
            list: Annotated copies of the input frames.
        """
        output_frames = []

        for i, frame in enumerate(frames):
            frame = frame.copy()
            cv2.putText(
                frame,
                str(i),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            output_frames.append(frame)

        return output_frames
