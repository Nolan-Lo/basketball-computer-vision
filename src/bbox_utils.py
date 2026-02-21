"""Bounding box utility functions for basketball video analysis.

Provides geometric helper functions for working with bounding boxes,
including distance measurement and center-point computation.
"""

import numpy as np


def get_center_of_bbox(bbox):
    """
    Compute the center point of a bounding box.

    Args:
        bbox (tuple or list): Bounding box in (x1, y1, x2, y2) format.

    Returns:
        tuple: (center_x, center_y) coordinates.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def measure_distance(point1, point2):
    """
    Compute the Euclidean distance between two 2D points.

    Args:
        point1 (tuple): (x, y) coordinates of the first point.
        point2 (tuple): (x, y) coordinates of the second point.

    Returns:
        float: Euclidean distance between the two points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def get_bbox_width(bbox):
    """
    Compute the width of a bounding box.

    Args:
        bbox (tuple or list): Bounding box in (x1, y1, x2, y2) format.

    Returns:
        float: Width of the bounding box.
    """
    return bbox[2] - bbox[0]


def get_bbox_height(bbox):
    """
    Compute the height of a bounding box.

    Args:
        bbox (tuple or list): Bounding box in (x1, y1, x2, y2) format.

    Returns:
        float: Height of the bounding box.
    """
    return bbox[3] - bbox[1]


def get_foot_position(bbox):
    """
    Get the bottom-center point of a bounding box.

    This approximates a player's foot position on the court, which is the
    most meaningful anchor for projecting to a 2D tactical view.

    Args:
        bbox (tuple or list): Bounding box in (x1, y1, x2, y2) format.

    Returns:
        tuple: (x, y) coordinates of the bottom-center point.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)
