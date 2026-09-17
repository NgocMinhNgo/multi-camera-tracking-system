# -*- coding: utf-8 -*-
"""
Dead Tracklet Packager.
Packages completed camera tracklets into compact serializable structures
containing metadata, trajectory boxes, and 256-dim FastReID feature vectors,
and releases memory.
"""

import json
import os
import numpy as np
from .track_manager import PersonTrackInfo, PersonTrackManager


class TrackletPackage:
    """
    Serializable payload representing a finished single-camera tracklet.
    """
    def __init__(
        self,
        cam_id: str,
        track_id: int,
        start_time: float | str,
        end_time: float | str,
        num_frames: int,
        boxes: list,
        rep_vectors: list[list[float]],
        view_labels: list[str]
    ):
        self.cam_id = cam_id
        self.track_id = track_id
        self.start_time = start_time
        self.end_time = end_time
        self.num_frames = num_frames
        self.boxes = boxes
        self.rep_vectors = rep_vectors
        self.view_labels = view_labels

    def to_dict(self) -> dict:
        return {
            "cam_id": self.cam_id,
            "track_id": self.track_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_frames": self.num_frames,
            "boxes": self.boxes,
            "num_vectors": len(self.rep_vectors),
            "rep_vectors": self.rep_vectors,
            "view_labels": self.view_labels
        }

    def __repr__(self) -> str:
        return f"<TrackletPackage {self.cam_id} Track #{self.track_id} | Frames: {self.num_frames} | Vectors: {len(self.rep_vectors)}>"


class TrackletPackager:
    """
    Handles packaging of real_dead_tracks and exporting to disk or MongoDB.
    """
    def __init__(self, output_dir: str = "output_tracklets"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def package_and_clean(self, track_info: PersonTrackInfo, track_manager: PersonTrackManager) -> TrackletPackage:
        """
        1. Tranches FastReID vectors & runs clustering for the dead track.
        2. Packages metadata into TrackletPackage.
        3. Purges track_info from RAM track_manager.
        """
        # Ensure FastReID vector extraction & clustering
        track_manager.update_emb_of_track(track_info.track_id)

        rep_vectors_list = (
            track_info.embeddings.tolist()
            if isinstance(track_info.embeddings, np.ndarray)
            else []
        )

        package = TrackletPackage(
            cam_id=track_info.cam_id,
            track_id=track_info.track_id,
            start_time=track_info.start_time,
            end_time=track_info.end_time,
            num_frames=len(track_info.boxes),
            boxes=track_info.boxes,
            rep_vectors=rep_vectors_list,
            view_labels=track_info.person_views
        )

        # Clean RAM memory
        track_manager.remove_track(track_info.track_id)
        return package

    def save_package(self, package: TrackletPackage) -> str:
        """
        Saves packaged tracklet to JSON file.
        """
        filename = f"{package.cam_id}_track_{package.track_id:04d}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(package.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"[{package.cam_id} | PACKAGER] Saved tracklet payload: {filepath}")
        return filepath
