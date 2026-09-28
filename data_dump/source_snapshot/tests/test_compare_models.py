import pandas as pd
import pytest

from scripts.compare_models import paired_bootstrap


def test_paired_bootstrap_matches_ids_and_constant_differences():
    current = pd.DataFrame(
        {
            "query_id": ["a", "b", "c"],
            "recall@50": [0.5, 1.0, 1.0],
            "filter_empty": [True, False, False],
        }
    )
    baseline = pd.DataFrame({"query_id": ["c", "a", "b"], "recall@50": [0.5, 0.0, 0.5]})
    report = paired_bootstrap(current, baseline, 0.63, iterations=2000)
    for metric in report.values():
        assert metric["delta"] == pytest.approx(0.5)
        assert metric["ci95"] == pytest.approx([0.5, 0.5])
        assert metric["bootstrap_positive_fraction"] == 1.0


def test_bootstrap_rejects_missing_baseline_queries():
    current = pd.DataFrame(
        {"query_id": ["a", "b"], "recall@50": [1.0, 1.0], "filter_empty": [True, False]}
    )
    baseline = pd.DataFrame({"query_id": ["a"], "recall@50": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        paired_bootstrap(current, baseline, 0.63)


def test_weighted_bootstrap_uses_reference_mix_and_fixed_seed():
    current = pd.DataFrame(
        {
            "query_id": ["a", "b", "c"],
            "recall@50": [1.0, 0.0, 0.5],
            "filter_empty": [True, False, False],
        }
    )
    baseline = pd.DataFrame({"query_id": ["a", "b", "c"], "recall@50": [0.0, 0.5, 0.5]})
    report = paired_bootstrap(current, baseline, 0.8, iterations=2000)
    assert report["recall@50"]["delta"] == pytest.approx(1 / 6)
    assert report["poststratified_recall@50"]["delta"] == pytest.approx(0.75)
    assert paired_bootstrap(current, baseline, 0.8, iterations=2000) == report
