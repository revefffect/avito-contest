"""Мягкая география из координат объявлений и обучающих переходов поиска."""

import numpy as np
import pandas as pd


def haversine(lat_a, lon_a, lat_b, lon_b):
    """Расстояние в километрах, скаляры и массивы поддерживаются одинаково."""
    lat_a, lon_a, lat_b, lon_b = [
        np.radians(value) for value in (lat_a, lon_a, lat_b, lon_b)
    ]
    value = (
        np.sin((lat_b - lat_a) / 2) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin((lon_b - lon_a) / 2) ** 2
    )
    return 12742 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    index = np.searchsorted(np.cumsum(weights), weights.sum() / 2, side="left")
    return float(values[index])


def coordinates(frame):
    values = frame[["item_latitude", "item_longitude"]].astype(float).to_numpy()
    valid = (
        np.isfinite(values).all(axis=1)
        & (np.abs(values[:, 0]) <= 90)
        & (np.abs(values[:, 1]) <= 180)
        & np.any(values != 0, axis=1)
    )
    values[~valid] = np.nan
    return values


class Geography:
    def fit(self, train, items, metadata_items=None):
        """В train должны быть только разрешённые обучающие пары, без eval."""
        self.item_locations = items["item_location_id"].to_numpy(copy=True)
        self.item_coordinates = coordinates(items)
        self.location_ids, self.location_indices = np.unique(
            self.item_locations, return_inverse=True
        )

        columns = ["item_id", "item_location_id", "item_latitude", "item_longitude"]
        metadata = items[columns]
        if metadata_items is not None:
            metadata = pd.concat([metadata, metadata_items[columns]], ignore_index=True)
        metadata = metadata.drop_duplicates("item_id").copy()
        metadata[["item_latitude", "item_longitude"]] = coordinates(metadata)
        centers = (
            metadata.groupby("item_location_id")[["item_latitude", "item_longitude"]]
            .median()
            .dropna()
        )
        self.centers = {
            location: point
            for location, point in zip(centers.index, centers.to_numpy())
        }

        marginal = train["item_location_id"].value_counts()
        if len(train):
            self.location_prior = (marginal / len(train)).to_dict()
        else:
            self.location_prior = (
                items["item_location_id"].value_counts(normalize=True).to_dict()
            )
        self.corpus_prior = np.array(
            [self.location_prior.get(location, 0.0) for location in self.location_ids]
        )

        # Сглаживание к общей частоте города уменьшает вес случайных единичных кликов.
        self.smoothing = 5.0
        self.transitions = {}
        pairs = train.groupby(["search_location_id", "item_location_id"]).size()
        for query_location, group in pairs.groupby(level=0, sort=False):
            self.transitions[query_location] = group.droplevel(0).to_dict()
        self.query_states = {}
        return self

    def _query_state(self, query_location):
        if query_location in self.query_states:
            return self.query_states[query_location]

        counts = self.transitions.get(query_location, {})
        train_count = sum(counts.values())
        known_locations = [location for location in counts if location in self.centers]
        point = self.centers.get(query_location)
        scope = 0.0
        if known_locations:
            points = np.array([self.centers[location] for location in known_locations])
            weights = np.array([counts[location] for location in known_locations])
            if point is None:
                # Локация поиска может обозначать область, которой нет у объявлений.
                point = np.array(
                    [
                        weighted_median(points[:, 0], weights),
                        weighted_median(points[:, 1], weights),
                    ]
                )
            distances = haversine(point[0], point[1], points[:, 0], points[:, 1])
            scope = weighted_median(distances, weights)

        observed = np.array([counts.get(location, 0) for location in self.location_ids])
        probability = (observed + self.smoothing * self.corpus_prior) / (
            train_count + self.smoothing
        )
        lift = np.divide(
            probability,
            self.corpus_prior,
            out=np.ones_like(probability),
            where=self.corpus_prior > 0,
        )
        state = {
            "point": point,
            "scope": scope,
            "train_count": train_count,
            "observed": observed,
            "probability": probability,
            "lift": lift,
        }
        self.query_states[query_location] = state
        return state

    def _distances(self, state, candidate_indices):
        points = self.item_coordinates[candidate_indices]
        has_center = np.isfinite(points).all(axis=1)
        if state["point"] is None:
            return np.zeros(len(points)), np.zeros(len(points), dtype=bool)
        point = state["point"]
        distances = haversine(point[0], point[1], points[:, 0], points[:, 1])
        return np.where(has_center, distances, 0.0), has_center

    def features(self, search_location_id, candidate_indices):
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        state = self._query_state(search_location_id)
        distances, has_center = self._distances(state, candidate_indices)
        locations = self.location_indices[candidate_indices]
        size = len(candidate_indices)
        result = {
            "same_location": self.item_locations[candidate_indices]
            == search_location_id,
            "log_distance": np.log1p(distances),
            "has_center": has_center,
            "transition_probability": state["probability"][locations],
            "transition_lift": state["lift"][locations],
            "scope_km": np.full(size, state["scope"]),
            "train_count": np.full(size, state["train_count"]),
        }
        return {name: values.astype(np.float32) for name, values in result.items()}

    def affinity(self, search_location_id):
        """Множитель текстовой близости в [0, 1], без жёсткого отсечения городов."""
        state = self._query_state(search_location_id)
        if state["point"] is None:
            return np.ones(len(self.item_locations), dtype=np.float32)

        indices = np.arange(len(self.item_locations))
        distances, has_center = self._distances(state, indices)
        scale = 25.0 + min(state["scope"], 150.0)
        nearby = 1.0 / (1.0 + distances / scale)
        nearby[~has_center] = 0.5

        probability = state["probability"]
        relative = probability / max(float(probability.max(initial=0)), 1e-12)
        lift = state["lift"]
        support = state["observed"] / (state["observed"] + self.smoothing)
        transition = support * (0.7 * np.sqrt(relative) + 0.3 * lift / (1 + lift))
        affinity = np.maximum(nearby, 0.85 * transition[self.location_indices])

        # Общероссийский поиск не должен становиться московским из-за медианы кликов.
        floor = 0.9 if state["scope"] > 300 else 0.02
        affinity = floor + (1 - floor) * affinity
        affinity[self.item_locations == search_location_id] = 1.0
        return np.clip(affinity, 0, 1).astype(np.float32)
