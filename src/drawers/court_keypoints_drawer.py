"""Drawer for court keypoint visualizations.

Uses the ``supervision`` library's vertex annotators to draw labelled
court keypoints on each frame.  Only keypoints with non-zero coordinates
(i.e. those that passed confidence filtering) are drawn.
"""

import numpy as np
import supervision as sv


class CourtKeypointDrawer:
    """
    Draw court keypoints and their numeric labels on frames.

    Only keypoints with non-zero coordinates are drawn, so off-screen or
    low-confidence keypoints (zeroed-out during detection) are ignored.

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
                tuples.  Keypoints with ``x == 0`` and ``y == 0`` are
                treated as absent and are **not** drawn.

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

            # Filter to only valid (non-zero) keypoints
            valid_indices = [
                i for i, kp in enumerate(raw_keypoints)
                if kp[0] > 0 and kp[1] > 0
            ]

            if not valid_indices:
                output_frames.append(annotated_frame)
                continue

            # Build a (1, N, 2) array for supervision's KeyPoints
            xy = np.array(
                [[raw_keypoints[i][0], raw_keypoints[i][1]] for i in valid_indices],
                dtype=np.float32,
            ).reshape(1, -1, 2)

            key_points = sv.KeyPoints(xy=xy)

            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=key_points,
            )

            # Labels are the original keypoint indices
            labels = [str(i) for i in valid_indices]
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=key_points,
                labels=labels,
            )

            output_frames.append(annotated_frame)

        return output_frames
