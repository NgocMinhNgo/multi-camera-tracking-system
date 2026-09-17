# -*- coding: utf-8 -*-
"""
Hungarian Algorithm Association for SORT Tracker.
Calculates IoU matrix between detection boxes and tracker state predicted boxes,
and uses scipy.optimize.linear_sum_assignment to compute optimal assignment.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(box1: list | np.ndarray, box2: list | np.ndarray) -> float:
    """
    Computes IoU between two bounding boxes [x1, y1, x2, y2].
    """
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter_area = w * h

    box1_area = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    box2_area = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def associate_detections_to_trackers(
    detections: list | np.ndarray,
    trackers: list | np.ndarray,
    low_iou_threshold: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assigns detections to tracked object bounding boxes.

    Returns:
      matches: ndarray of shape (N, 2) where each row is [det_idx, trk_idx]
      unmatched_detections: ndarray of unassigned detection indices
      unmatched_trackers: ndarray of unassigned tracker indices
    """
    if len(trackers) == 0 or len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.arange(len(trackers))
        )

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    # Hungarian assignment (maximizing total IoU score)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.column_stack((row_indices, col_indices))

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matched pairs with low IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < low_iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m)

    if len(matches) == 0:
        matches_arr = np.empty((0, 2), dtype=int)
    else:
        matches_arr = np.asarray(matches, dtype=int)

    return matches_arr, np.asarray(unmatched_detections, dtype=int), np.asarray(unmatched_trackers, dtype=int)
