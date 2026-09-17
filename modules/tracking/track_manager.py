# -*- coding: utf-8 -*-
"""
PersonTrackManager & RAM Buffer Management.
Manages in-memory active customer tracks (dict_tracks), crop selection,
and dead track detection (get_last_dead_tracks).
"""

import time
import numpy as np
from modules.vectorization.fastreid_vectorizer import FastReIDVectorizer
from modules.clustering.clustering import TrackletClusterer


class PersonTrackInfo:
    """
    In-RAM state data container for an individual customer tracklet in a single camera.
    """
    def __init__(self, track_id: int, timestamp: float | str, cam_id: str = "CAM_001"):
        self.track_id = int(track_id)
        self.cam_id = str(cam_id)
        self.start_time = timestamp
        self.end_time = timestamp
        self.dead_time = -1.0
        self.boxes = []
        self.scores = []
        self.cropped_imgs: list[np.ndarray] = []
        self.embeddings: np.ndarray = np.empty((0, 256), dtype=np.float32)
        self.person_views: list[str] = []

    def add_box(self, box: list, score: float, timestamp: float | str):
        self.boxes.append(list(map(int, box[:4])))
        self.scores.append(float(score))
        self.end_time = timestamp


class PersonTrackManager:
    """
    Manager class operating in RAM to store and update active/dead track session states.
    """
    def __init__(self, cam_id: str = "CAM_001", max_fragment_time: float = 3.0):
        self.cam_id = cam_id
        self.max_fragment_time = max_fragment_time
        self.dict_tracks: dict[int, PersonTrackInfo] = {}

        # Vectorization & Clustering engine
        self.vectorizer = FastReIDVectorizer(emb_size=256)
        self.clusterer = TrackletClusterer(n_clusters=5, batch_size=20)

    def update_session_tracks(self, timestamp: float, alive_tracks: list, dead_tracks: list[int], frame: np.ndarray):
        """
        Updates session state in RAM for each frame.
        """
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2

        # 1. Update alive tracks
        for track_data in alive_tracks:
            t_id = track_data.track_id
            if t_id not in self.dict_tracks:
                new_track = PersonTrackInfo(track_id=t_id, timestamp=timestamp, cam_id=self.cam_id)
                self.dict_tracks[t_id] = new_track

            self.dict_tracks[t_id].add_box(track_data.box, track_data.score, timestamp)

            # Crop selection (is_good_box)
            if hasattr(track_data, 'is_good_box') and track_data.is_good_box(center_x):
                if track_data.cropped_img is not None and track_data.cropped_img.size > 0:
                    self.dict_tracks[t_id].cropped_imgs.append(track_data.cropped_img)

        # 2. Update dead tracks timestamp
        for t_id in dead_tracks:
            if t_id in self.dict_tracks and self.dict_tracks[t_id].dead_time < 0:
                self.dict_tracks[t_id].dead_time = timestamp

    def get_last_dead_tracks(self, timestamp: float) -> list[PersonTrackInfo]:
        """
        Returns tracks that have been dead for > max_fragment_time seconds.
        """
        real_dead_tracks = []
        for t_id, track_info in self.dict_tracks.items():
            if track_info.dead_time > 0 and (timestamp - track_info.dead_time) >= self.max_fragment_time:
                real_dead_tracks.append(track_info)
        return real_dead_tracks

    def update_emb_of_track(self, track_id: int):
        """
        Extracts FastReID 256-dim feature vectors for crops, performs clustering,
        keeps top representative vectors, and clears crop image buffer from RAM.
        """
        if track_id not in self.dict_tracks:
            return

        track_info = self.dict_tracks[track_id]
        if not track_info.cropped_imgs:
            return

        # 1. Extract 256-dim feature vectors
        embs, person_views, _ = self.vectorizer.get_infer_result(track_info.cropped_imgs)

        # Combine with existing embeddings
        if len(track_info.embeddings) > 0:
            all_embs = np.vstack([track_info.embeddings, embs])
            all_views = track_info.person_views + person_views
        else:
            all_embs = embs
            all_views = person_views

        # 2. Clustering to select top medoid representative vectors
        selected_embs, selected_views, _ = self.clusterer.fit_and_select(all_embs, all_views)

        track_info.embeddings = selected_embs
        track_info.person_views = selected_views

        # 3. Purge raw crop images to free RAM
        track_info.cropped_imgs = []

    def remove_track(self, track_id: int):
        """
        Purges dead track from RAM dict_tracks completely.
        """
        if track_id in self.dict_tracks:
            del self.dict_tracks[track_id]
            print(f"[{self.cam_id} | RAM CLEANUP] Purged track #{track_id} from memory.")
