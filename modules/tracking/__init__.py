# -*- coding: utf-8 -*-
"""
Single-Camera Tracking & Tracklet Packaging Module.
Exposes PersonTracker (SORT), PersonTrackManager (RAM buffer), and TrackletPackager.
"""

from .kalman_box import KalmanBoxTracker
from .associate import associate_detections_to_trackers
from .sort_tracker import PersonTracker, TrackingResult
from .track_manager import PersonTrackManager, PersonTrackInfo
from .tracklet_packager import TrackletPackager, TrackletPackage

__all__ = [
    "KalmanBoxTracker",
    "associate_detections_to_trackers",
    "PersonTracker",
    "TrackingResult",
    "PersonTrackManager",
    "PersonTrackInfo",
    "TrackletPackager",
    "TrackletPackage",
]
