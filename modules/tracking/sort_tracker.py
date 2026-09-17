# -*- coding: utf-8 -*-
"""
Single-Camera SORT Multi-Object Tracker.
Combines KalmanBoxTracker prediction and Hungarian algorithm IoU association
to maintain camera-level persistent object IDs across video frames.
"""

import numpy as np
from .kalman_box import KalmanBoxTracker
from .associate import associate_detections_to_trackers


class TrackingResult:
    """
    Data structure representing a tracked object instance in a frame.
    """
    def __init__(self, box: list | np.ndarray, score: float, track_id: int, cropped_img=None, distance=None):
        self.box = list(map(int, box[:4]))
        self.score = float(score)
        self.track_id = int(track_id)
        self.cropped_img = cropped_img
        self.distance = distance
        self.classname = "person"

    def is_good_box(self, center_x: int, selection_config: dict | None = None) -> bool:
        """
        Quality selection check for person cropped image:
        - Box dimensions > min_size (15px)
        - Aspect ratio (height/width) is reasonable for standing person
        """
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        if w < 15 or h < 15:
            return False
        # Ratio check: humans are typically taller than wide or square
        ratio = h / max(1.0, float(w))
        if ratio < 0.5:
            return False
        return True


class PersonTracker:
    """
    SORT Tracker Manager for a single Camera stream.
    """
    def __init__(self, max_age: int = 30, min_hits: int = 3, low_iou_threshold: float = 0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.low_iou_threshold = low_iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def track(self, detections: list, frame: np.ndarray) -> tuple[list[TrackingResult], list[int]]:
        """
        Updates trackers with frame detections.

        Args:
          detections: List of DetectionResult objects or bounding box dicts/lists.
          frame: Current frame RGB/BGR numpy array [H, W, 3].

        Returns:
          alive_tracks: List of active TrackingResult objects with persistent track_id.
          dead_tracks: List of track IDs deleted in this frame due to inactivity (> max_age).
        """
        self.frame_count += 1

        # Extract bbox array from detections
        det_boxes = []
        det_scores = []
        for det in detections:
            if hasattr(det, 'box'):
                det_boxes.append(det.box)
                det_scores.append(getattr(det, 'score', 1.0))
            elif isinstance(det, (list, tuple, np.ndarray)):
                det_boxes.append(det[:4])
                det_scores.append(det[4] if len(det) > 4 else 1.0)
            elif isinstance(det, dict) and 'box' in det:
                det_boxes.append(det['box'])
                det_scores.append(det.get('score', 1.0))

        det_boxes_arr = np.array(det_boxes, dtype=np.float32) if len(det_boxes) > 0 else np.empty((0, 4))

        # Get predicted positions from existing trackers
        predicted_trks = np.zeros((len(self.trackers), 4), dtype=np.float32)
        to_del_indices = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if isinstance(pos, np.ndarray) and pos.ndim > 1:
                pos = pos[0]
            if np.any(np.isnan(pos)):
                to_del_indices.append(t)
            else:
                predicted_trks[t] = pos[:4]

        # Purge invalid trackers
        for t in sorted(to_del_indices, reverse=True):
            self.trackers.pop(t)
            predicted_trks = np.delete(predicted_trks, t, axis=0)

        # Hungarian assignment
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_boxes_arr, predicted_trks, low_iou_threshold=self.low_iou_threshold
        )

        # Update matched trackers
        alive_tracks: list[TrackingResult] = []
        for d, t in matched:
            self.trackers[t].update(det_boxes_arr[d])
            if self.trackers[t].hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                x1, y1, x2, y2 = list(map(int, det_boxes_arr[d][:4]))
                # Crop person image safely from frame
                h_img, w_img = frame.shape[:2]
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w_img, x2), min(h_img, y2)
                crop_img = frame[y1_c:y2_c, x1_c:x2_c] if (x2_c > x1_c and y2_c > y1_c) else None

                res = TrackingResult(
                    box=[x1, y1, x2, y2],
                    score=det_scores[d],
                    track_id=self.trackers[t].id,
                    cropped_img=crop_img
                )
                alive_tracks.append(res)

        # Handle unmatched detections -> spawn new KalmanBoxTracker
        for i in unmatched_dets:
            trk = KalmanBoxTracker(det_boxes_arr[i])
            self.trackers.append(trk)
            if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                x1, y1, x2, y2 = list(map(int, det_boxes_arr[i][:4]))
                h_img, w_img = frame.shape[:2]
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w_img, x2), min(h_img, y2)
                crop_img = frame[y1_c:y2_c, x1_c:x2_c] if (x2_c > x1_c and y2_c > y1_c) else None

                res = TrackingResult(
                    box=[x1, y1, x2, y2],
                    score=det_scores[i],
                    track_id=trk.id,
                    cropped_img=crop_img
                )
                alive_tracks.append(res)

        # Handle dead trackers
        dead_tracks: list[int] = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                dead_tracks.append(trk.id)
                self.trackers.pop(i)

        return alive_tracks, dead_tracks
