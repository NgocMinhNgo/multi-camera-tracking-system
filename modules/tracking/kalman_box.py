# -*- coding: utf-8 -*-
"""
Kalman Filter Bounding Box Tracker for SORT Algorithm.
State space: [x, y, s, r, vx, vy, vs]
  - x, y: center coordinates of bounding box
  - s: scale (area = width * height)
  - r: aspect ratio (width / height)
  - vx, vy, vs: respective velocities
Measurement space: [x, y, s, r]
"""

import numpy as np


def convert_bbox_to_z(bbox: list | np.ndarray) -> np.ndarray:
    """
    Takes a bounding box in [x1, y1, x2, y2] format and returns z in [x, y, s, r]^T.
    """
    w = max(1.0, float(bbox[2] - bbox[0]))
    h = max(1.0, float(bbox[3] - bbox[1]))
    x = float(bbox[0]) + w / 2.0
    y = float(bbox[1]) + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray, score: float | None = None) -> np.ndarray:
    """
    Takes a state vector x and returns bounding box in [x1, y1, x2, y2] format.
    """
    s = max(1.0, float(x[2, 0]))
    r = max(0.01, float(x[3, 0]))
    w = np.sqrt(s * r)
    h = s / w
    x1 = float(x[0, 0]) - w / 2.0
    y1 = float(x[1, 0]) - h / 2.0
    x2 = float(x[0, 0]) + w / 2.0
    y2 = float(x[1, 0]) + h / 2.0

    if score is None:
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    else:
        return np.array([x1, y1, x2, y2, score], dtype=np.float32)


class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked objects observed as 2D bounding boxes.
    """
    count = 0

    def __init__(self, bbox: list | np.ndarray):
        """
        Initializes a tracker using initial bounding box [x1, y1, x2, y2].
        """
        # State transition matrix F (7x7)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix H (4x7)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # Measurement noise covariance matrix R (4x4)
        self.R = np.eye(4, dtype=np.float32)
        self.R[2:, 2:] *= 10.0

        # State covariance matrix P (7x7)
        self.P = np.eye(7, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 1000.0  # High initial uncertainty for unobservable velocities

        # Process noise covariance matrix Q (7x7)
        self.Q = np.eye(7, dtype=np.float32)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        # State vector x (7x1)
        self.x = np.zeros((7, 1), dtype=np.float32)
        self.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.d_box = list(map(int, bbox[:4]))

    def update(self, bbox: list | np.ndarray, use_predict: bool = False):
        """
        Updates the state vector with observed bounding box.
        """
        if use_predict:
            self.hit_streak = 0
        else:
            self.d_box = list(map(int, bbox[:4]))
            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1

        z = convert_bbox_to_z(bbox)
        # Kalman Filter Measurement Update:
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(7, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def predict(self) -> np.ndarray:
        """
        Advances state vector and returns predicted bounding box [x1, y1, x2, y2].
        """
        if (self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0

        # Kalman Filter State Prediction:
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.time_since_update += 1
        if self.time_since_update > 1:
            self.hit_streak = 0

        predicted_bbox = convert_x_to_bbox(self.x)
        self.history.append(predicted_bbox)
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """
        Returns current bounding box estimate [x1, y1, x2, y2].
        """
        return convert_x_to_bbox(self.x)
