import numpy as np
import pandas as pd
import pytest

from scripts.predict import (
    normalize_weights,
    standardize_by_query,
    top_predictions,
    validate_candidates,
)


def fixtures():
    queries = pd.DataFrame({"query_id": ["0000000000000000", "0000000000000001"]})
    items = pd.DataFrame({"item_id": [f"{value:016x}" for value in range(3)]})
    candidates = pd.DataFrame({"qid": [0, 0, 1], "item_idx": [0, 1, 2]})
    return candidates, queries, items


@pytest.mark.parametrize(
    "violation", ["missing_query", "duplicate", "tail_item", "wrong_prefix"]
)
def test_prediction_rejects_incompatible_candidates(violation):
    candidates, queries, items = fixtures()
    corpus = items.copy()
    if violation == "missing_query":
        candidates = candidates.iloc[:2]
    elif violation == "duplicate":
        candidates = pd.concat([candidates, candidates.iloc[:1]], ignore_index=True)
    elif violation == "tail_item":
        corpus.loc[3, "item_id"] = "0000000000000003"
        candidates.loc[2, "item_idx"] = 3
    else:
        corpus = corpus.iloc[::-1]
    with pytest.raises(ValueError):
        validate_candidates(candidates, queries, corpus, items)


def test_prediction_uses_stable_ties_and_preserves_string_ids():
    candidates, queries, items = fixtures()
    candidates = candidates.iloc[::-1]
    validate_candidates(candidates, queries, items, items)
    result = top_predictions(candidates, np.ones(3), items["item_id"].to_numpy(), 2)
    assert result == [["0000000000000000", "0000000000000001"], ["0000000000000002"]]


def test_model_standardization_is_per_query_and_constant_scores_are_neutral():
    qids = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    raw = np.array([10.0, 200.0, 20.0, 100.0, 42.0])
    result = standardize_by_query(raw, qids, 3)
    np.testing.assert_allclose(result, [-1, 1, 1, -1, 0])
    np.testing.assert_allclose(standardize_by_query(7 * raw + 3, qids, 3), result)


def test_weight_sum_overflow_is_rejected_before_it_zeroes_predictions():
    with pytest.raises(ValueError, match="positive finite sum"):
        normalize_weights([1e308, 1e308], 2)
    np.testing.assert_array_equal(normalize_weights([0.25, 0.75], 2), [0.25, 0.75])


def test_missing_nullable_item_position_is_rejected():
    candidates, queries, items = fixtures()
    candidates["item_idx"] = pd.array([0, 1, None], dtype="Int64")
    with pytest.raises(ValueError, match="item_idx"):
        validate_candidates(candidates, queries, items, items)
