import numpy as np
import pandas as pd

from avito_retrieval.geography import Geography


def item_frame():
    return pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "item_location_id": [10, 20, 30, 40],
            "item_latitude": [55.75, 55.8, 59.94, np.nan],
            "item_longitude": [37.62, 37.9, 30.32, np.nan],
        }
    )


def test_known_city_prefers_exact_and_nearby_without_excluding_far_items():
    train = pd.DataFrame(
        {"search_location_id": [10] * 11, "item_location_id": [10] * 10 + [20]}
    )
    geography = Geography().fit(train, item_frame())
    affinity = geography.affinity(10)
    assert affinity.dtype == np.float32
    assert affinity[0] == 1
    assert 0 < affinity[2] < affinity[1] < affinity[0]
    features = geography.features(10, [2, 0, 1])
    assert features["same_location"].tolist() == [0, 1, 0]
    assert features["transition_probability"][1] > features["transition_probability"][2]
    assert all(np.isfinite(value).all() for value in features.values())


def test_query_only_region_uses_training_transitions_and_no_numeric_id_proximity():
    train = pd.DataFrame(
        {"search_location_id": [999] * 10, "item_location_id": [10] * 8 + [20] * 2}
    )
    geography = Geography().fit(train, item_frame())
    affinity = geography.affinity(999)
    assert affinity[0] > affinity[2]
    assert geography.features(999, [0])["has_center"].tolist() == [1]
    assert geography.features(999, [0])["train_count"].tolist() == [10]
    assert np.all(geography.affinity(998) == 1)


def test_broad_scope_weakens_geographic_penalty():
    items = item_frame()
    items.loc[1, ["item_latitude", "item_longitude"]] = [56.8, 60.6]
    train = pd.DataFrame(
        {"search_location_id": [999] * 3, "item_location_id": [10, 20, 30]}
    )
    geography = Geography().fit(train, items)
    assert geography.features(999, [0])["scope_km"][0] > 300
    assert np.all(geography.affinity(999) >= 0.9)


def test_missing_coordinates_do_not_turn_into_zero_distance_observations():
    train = pd.DataFrame({"search_location_id": [10], "item_location_id": [10]})
    geography = Geography().fit(train, item_frame())
    features = geography.features(10, [3])
    assert features["has_center"].tolist() == [0]
    assert features["log_distance"].tolist() == [0]
    assert np.isfinite(geography.affinity(10)).all()


def test_optional_metadata_recovers_center_outside_corpus_and_fit_resets_state():
    items = item_frame()
    metadata = pd.DataFrame(
        {
            "item_id": ["e"],
            "item_location_id": [50],
            "item_latitude": [55.76],
            "item_longitude": [37.63],
        }
    )
    empty = pd.DataFrame(columns=["search_location_id", "item_location_id"])
    geography = Geography().fit(empty, items, metadata_items=metadata)
    assert geography.features(50, [0])["has_center"].tolist() == [1]
    assert geography.affinity(50)[0] > geography.affinity(50)[2]
    geography.fit(empty, items)
    assert np.all(geography.affinity(50) == 1)
