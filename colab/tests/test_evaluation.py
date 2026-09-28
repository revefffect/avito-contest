import numpy as np
import pandas as pd
import pytest

from avito_retrieval.evaluation import evaluate_candidates


def data():
    queries = pd.DataFrame(
        {
            "qid": [0, 1, 2],
            "query_id": ["a", "b", "c"],
            "seen_text": [True, False, False],
        }
    )
    corpus = pd.DataFrame({"item_id": ["x", "y", "z", "w"]})
    truth = pd.DataFrame(
        {"query_id": ["a", "a", "b", "c"], "item_id": ["x", "y", "z", "w"]}
    )
    candidates = pd.DataFrame(
        {
            "qid": [0, 0, 1],
            "item_idx": [0, 2, 2],
            "score": [1.0, 0.0, 1.0],
            "label": [1, 0, 1],
        }
    )
    return candidates, queries, truth, corpus


def test_full_truth_denominator_and_missing_queries_remain_in_macro_mean():
    candidates, queries, truth, corpus = data()
    summary, per_query = evaluate_candidates(candidates, queries, truth, corpus)
    assert summary["recall@50"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert summary["recall@union"] == pytest.approx(0.5)
    assert per_query["candidate_count"].tolist() == [2, 1, 0]
    assert per_query["relevant_count"].tolist() == [2, 1, 1]
    candidates["label"] = 0
    assert (
        evaluate_candidates(candidates, queries, truth, corpus)[0]["recall@50"]
        == summary["recall@50"]
    )


def test_score_ties_use_corpus_position_and_not_input_order():
    candidates, queries, truth, corpus = data()
    candidates.loc[:, "score"] = 1.0
    forward = evaluate_candidates(candidates, queries, truth, corpus, ks=(1,))[1]
    reverse = evaluate_candidates(
        candidates.iloc[::-1], queries, truth, corpus, ks=(1,)
    )[1]
    np.testing.assert_array_equal(forward["recall@1"], reverse["recall@1"])
    assert forward.loc[0, "recall@1"] == 0.5


def test_duplicate_candidates_are_rejected():
    candidates, queries, truth, corpus = data()
    repeated = pd.concat([candidates, candidates.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate query-item"):
        evaluate_candidates(repeated, queries, truth, corpus)


def test_poststratification_uses_reference_filter_mix_without_changing_macro():
    candidates, queries, truth, corpus = data()
    queries["search_infm_params_text"] = ["", "", "service filter"]
    summary, _ = evaluate_candidates(
        candidates, queries, truth, corpus, reference_empty_fraction=0.8
    )
    assert summary["recall@50"] == pytest.approx(0.5)
    assert summary["poststratified_recall@50"] == pytest.approx(0.8 * 0.75 + 0.2 * 0)


def test_poststratification_reports_unknown_when_a_required_group_is_absent():
    candidates, queries, truth, corpus = data()
    summary, _ = evaluate_candidates(
        candidates, queries, truth, corpus, reference_empty_fraction=0.8
    )
    assert summary["poststratified_recall@50"] is None
