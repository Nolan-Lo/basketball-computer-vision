"""Homography estimation and point transformation.

Computes a planar homography from matched 2D point correspondences
using OpenCV's ``findHomography`` and applies it via perspective
transformation.  Used by :class:`TacticalViewConverter` to map
video-frame coordinates to the standardised 2D court plane.
"""

import numpy as np
import cv2


class Homography:
    """Thin wrapper around ``cv2.findHomography`` + ``cv2.perspectiveTransform``.

    Args:
        source (np.ndarray): Source 2D points, shape ``(N, 2)``.
        target (np.ndarray): Corresponding target 2D points, shape ``(N, 2)``.

    Raises:
        ValueError: If shapes don't match, points aren't 2D, or the
            homography cannot be computed.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Project *points* through the homography.

        Args:
            points (np.ndarray): Array of shape ``(N, 2)`` with 2D coordinates.

        Returns:
            np.ndarray: Transformed points with the same shape.

        Raises:
            ValueError: If *points* are not 2D.
        """
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)
