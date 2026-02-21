"""Low-level drawing primitives used by the drawer classes.

Provides ellipse and triangle rendering helpers that operate directly on
OpenCV frames (NumPy arrays in BGR format).
"""

import cv2
import numpy as np


def draw_ellipse(frame, bbox, color, track_id=None):
    """
    Draw a team-colored ellipse at the bottom of a player bounding box.

    The ellipse acts as a ground-plane marker beneath the player, giving a
    cleaner look than a full rectangular bounding box.

    Args:
        frame (numpy.ndarray): The BGR frame to draw on (modified in-place).
        bbox (tuple | list): Player bounding box ``(x1, y1, x2, y2)``.
        color (tuple | list): BGR color for the ellipse.
        track_id (int | None): If provided, render the ID as a label inside the
            ellipse.

    Returns:
        numpy.ndarray: The frame with the ellipse drawn.
    """
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) // 2
    width = x2 - x1

    # Ellipse sits at the bottom of the bounding box
    center = (center_x, y2)
    axes = (width // 2, int(0.35 * (width // 2)))

    # Filled semi-transparent ellipse (bottom half via angles 0°-360°)
    cv2.ellipse(frame, center, axes, angle=0.0,
                startAngle=-45, endAngle=235,
                color=color, thickness=2, lineType=cv2.LINE_4)

    # Draw track ID label centred inside the ellipse
    if track_id is not None:
        label = str(track_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Small rectangle behind the text for readability
        rect_w, rect_h = tw + 10, th + 10
        rect_x1 = center_x - rect_w // 2
        rect_y1 = y2 - rect_h // 2 + 15
        rect_x2 = center_x + rect_w // 2
        rect_y2 = y2 + rect_h // 2 + 15

        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, cv2.FILLED)
        text_x = rect_x1 + 5
        text_y = rect_y2 - 5
        cv2.putText(frame, label, (text_x, text_y), font, font_scale,
                    (0, 0, 0), thickness)

    return frame


def draw_triangle(frame, bbox, color):
    """
    Draw a downward-pointing triangle above a bounding box.

    Used as a pointer/indicator above the ball or the player carrying the
    ball.

    Args:
        frame (numpy.ndarray): The BGR frame to draw on (modified in-place).
        bbox (tuple | list): Bounding box ``(x1, y1, x2, y2)``.
        color (tuple | list): BGR color for the triangle.

    Returns:
        numpy.ndarray: The frame with the triangle drawn.
    """
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) // 2
    triangle_size = 10

    # Three vertices of a downward-pointing triangle above the box
    pts = np.array([
        [center_x, y1],
        [center_x - triangle_size, y1 - triangle_size * 2],
        [center_x + triangle_size, y1 - triangle_size * 2],
    ])

    cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)  # black outline

    return frame
