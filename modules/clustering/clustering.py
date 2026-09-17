# -*- coding: utf-8 -*-
"""
Tracklet Feature Vector Clustering.
Uses sklearn.cluster.MiniBatchKMeans or DBSCAN to cluster feature vectors of accumulated
tracklet crops, selecting top medoid/centroid representative vectors and removing noise.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans


class TrackletClusterer:
    def __init__(self, n_clusters: int = 5, batch_size: int = 20):
        self.n_clusters = n_clusters
        self.batch_size = batch_size

    def fit_and_select(
        self,
        embeddings: np.ndarray,
        person_views: list[str] | None = None,
        cropped_imgs: list[np.ndarray] | None = None
    ) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
        """
        Clusters embeddings and returns representative medoid vectors, corresponding views, and crops.

        Args:
          embeddings: ndarray of shape (N, D)
          person_views: Optional list of view labels
          cropped_imgs: Optional list of image crop arrays

        Returns:
          selected_embs: ndarray of shape (K, D) where K <= n_clusters
          selected_views: Selected view labels
          selected_crops: Selected image crop arrays
        """
        N = len(embeddings)
        if N == 0:
            return (
                np.empty((0, 256), dtype=np.float32),
                [],
                []
            )

        if N <= self.n_clusters:
            # Fewer vectors than clusters -> return all
            views = person_views if person_views is not None else ["front"] * N
            crops = cropped_imgs if cropped_imgs is not None else []
            return embeddings, views, crops

        # Perform MiniBatchKMeans clustering
        actual_clusters = min(self.n_clusters, N)
        kmeans = MiniBatchKMeans(n_clusters=actual_clusters, random_state=42, batch_size=max(10, self.batch_size))
        labels = kmeans.fit_predict(embeddings)

        selected_indices = []
        # Find medoid (closest vector to center) for each cluster
        for k in range(actual_clusters):
            cluster_mask = (labels == k)
            if not np.any(cluster_mask):
                continue
            cluster_indices = np.where(cluster_mask)[0]
            cluster_center = kmeans.cluster_centers_[k]

            # Compute Euclidean distances to centroid
            dists = np.linalg.norm(embeddings[cluster_indices] - cluster_center, axis=1)
            medoid_idx = cluster_indices[np.argmin(dists)]
            selected_indices.append(medoid_idx)

        selected_indices = sorted(selected_indices)
        selected_embs = embeddings[selected_indices]

        selected_views = []
        if person_views is not None:
            selected_views = [person_views[idx] for idx in selected_indices if idx < len(person_views)]

        selected_crops = []
        if cropped_imgs is not None:
            selected_crops = [cropped_imgs[idx] for idx in selected_indices if idx < len(cropped_imgs)]

        return selected_embs, selected_views, selected_crops
