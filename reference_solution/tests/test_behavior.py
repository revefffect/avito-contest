import numpy as np
import pandas as pd

from avito_retrieval.behavior import BehaviorIndex


def training_frames():
    train = pd.DataFrame(
        {
            "search_query": [
                "сантехник",
                "сантехник",
                "сантехника ремонт",
                "торт на заказ",
                "кондитер торты",
            ],
            "search_location_id": [10, 10, 20, 10, 20],
            "search_is_delivery_search": [0] * 5,
            "search_infm_params_text": [
                "Вид услуги Ремонт и отделка",
                "Вид услуги Ремонт и отделка",
                "Вид услуги Ремонт и отделка",
                "",
                "",
            ],
            "search_category": [114] * 5,
            "item_id": ["a", "a", "b", "c", "c"],
            "item_microcat_id": [1, 1, 1, 2, 2],
            "item_location_id": [10, 10, 20, 10, 10],
        }
    )
    items = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "held_out", "new_category"],
            "item_microcat_id": [1, 1, 2, 1, 99],
        }
    )
    return train, items


def query_row(text, location=10, filters=""):
    return {
        "search_query": text,
        "search_location_id": location,
        "search_is_delivery_search": 0,
        "search_infm_params_text": filters,
        "search_category": 114,
    }


def test_unseen_query_can_transfer_category_without_inventing_exact_counts():
    train, items = training_frames()
    model = BehaviorIndex().fit(train, items)
    query = query_row("сантехник срочный")
    info = model.query_info(query)
    features = model.features(query, np.arange(len(items)), info)
    assert info["text_count"] == 0
    assert np.all(features["text_item_count"] == 0)
    assert np.all(features["context_item_count"] == 0)
    assert features["microcat_probability"][3] > features["microcat_probability"][2]
    assert features["microcat_probability"][4] == 0
    # Категорию нового объявления можно предсказать, его несуществующие клики — нельзя.
    assert features["popularity"][3] == 0
    assert features["memory_score"][3] == 0


def test_exact_text_and_exact_context_have_different_meaning():
    train, items = training_frames()
    model = BehaviorIndex().fit(train, items)
    query = query_row("Сантехник!", location=999, filters="Вид услуги Ремонт и отделка")
    features = model.features(query, np.arange(len(items)), model.query_info(query))
    assert np.isclose(features["text_item_count"][0], np.log1p(2))
    assert np.all(features["context_item_count"] == 0)
    query["search_location_id"] = 10
    features = model.features(query, np.arange(len(items)), model.query_info(query))
    assert np.isclose(features["context_item_count"][0], np.log1p(2))
    assert features["context_item_count"][3] == 0


def test_unrelated_query_has_no_neighbor_evidence_and_empty_filter_is_neutral():
    train, items = training_frames()
    model = BehaviorIndex().fit(train, items)
    query = query_row("xyz987654")
    info = model.query_info(query)
    features = model.features(query, np.arange(len(items)), info)
    assert info["nearest_similarity"] == 0
    assert np.all(info["category_probs"] == 0)
    assert np.all(info["memory"] == 0)
    assert np.all(features["filter_microcat_probability"] == 0)
    assert np.all(features["filter_train_count"] == 0)
    assert all(np.isfinite(values).all() for values in features.values())


def test_known_filter_prior_respects_categories_without_inventing_item_history():
    train, items = training_frames()
    model = BehaviorIndex().fit(train, items)
    query = query_row("сантехника", filters="Вид услуги Ремонт и отделка")
    features = model.features(query, np.arange(len(items)), model.query_info(query))
    assert features["filter_microcat_probability"].tolist() == [1, 1, 0, 1, 0]
    assert features["text_item_count"][3] == 0
    assert features["context_item_count"][3] == 0
