"""Drawer for court keypoint visualizations.

Uses the ``supervision`` library's vertex annotators to draw labelled
court keypoints on each frame.
"""

import numpy as np
import supervision as sv


class CourtKeypointDrawer:
    """
    Draw court keypoints and their numeric labels on frames.

    Attributes:
        keypoint_color (str): Hex colour for the keypoint dots.
    """

    def __init__(self, keypoint_color="#ff2c2c"):
        """
        Args:
            keypoint_color (str): Hex colour string (e.g. ``'#ff2c2c'``).
        """
        self.keypoint_color = keypoint_color

    def draw(self, frames, court_keypoints):
        """
        Draw court keypoints on a list of frames.

        Args:
            frames (list): Video frames (NumPy arrays).
            court_keypoints (list): Per-frame list of ``(x, y, confidence)``
                tuples (as produced by ``detect_court_keypoints`` in the
                pipeline).

        Returns:
            list: Annotated copies of the input frames.
        """
        color = sv.Color.from_hex(self.keypoint_color)

        vertex_annotator = sv.VertexAnnotator(color=color, radius=8)
        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=color,
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1,
        )

        output_frames = []

        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()
            raw_keypoints = court_keypoints[index]

            if not raw_keypoints:
                output_frames.append(annotated_frame)
                continue

            # Build a (1, N, 2) array for supervision's KeyPoints
            xy = np.array(
                [[kp[0], kp[1]] for kp in raw_keypoints], dtype=np.float32
            ).reshape(1, -1, 2)

            key_points = sv.KeyPoints(xy=xy)

            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=key_points,
            )

            # Labels are the keypoint indices
            labels = [str(i) for i in range(len(raw_keypoints))]
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=key_points,
                labels=labels,
            )

            output_frames.append(annotated_frame)

        return output_frames
