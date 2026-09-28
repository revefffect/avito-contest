"""Macro Recall с полной разметкой, включая не найденные генератором объявления."""

from pathlib import Path

import numpy as np
import pandas as pd


RESERVED_COLUMNS = {
    "qid",
    "item_idx",
    "label",
    "query_id",
    "item_id",
    "split",
    "target",
}


def load_split_queries(path: str | Path, splits: list[str]) -> pd.DataFrame:
    # Позиция qid привязана к исходному файлу и не меняется после фильтрации.
    identities = pd.read_parquet(path, columns=["query_id", "split"])
    identities["qid"] = np.arange(len(identities), dtype=np.int32)
    selected = pd.read_parquet(path, filters=[("split", "in", splits)])
    return selected.merge(
        identities[["query_id", "qid"]], on="query_id", validate="one_to_one"
    )


def feature_columns(frame: pd.DataFrame) -> list[str]:
    names = [name for name in frame if name not in RESERVED_COLUMNS]
    if not names:
        raise ValueError("no model features found")
    invalid = [name for name in names if not pd.api.types.is_numeric_dtype(frame[name])]
    if invalid:
        raise ValueError(f"model features must be numeric: {invalid}")
    return names


def reference_empty_fraction(path: str | Path) -> float:
    """Вычислить долю запросов без фильтров по признакам, без тестовой разметки."""
    frame = pd.read_parquet(path, columns=["search_infm_params_text"])
    if frame.empty:
        raise ValueError("reference queries must not be empty")
    return float(
        frame["search_infm_params_text"]
        .map(lambda value: not isinstance(value, str) or not value.strip())
        .mean()
    )


def evaluate_candidates(
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    truth: pd.DataFrame,
    corpus: pd.DataFrame,
    score_column="score",
    ks=(10, 50, 100),
    reference_empty_fraction=None,
) -> tuple[dict, pd.DataFrame]:
    """Вернуть сводку и строки запросов, порядок ничьих фиксирован по item_idx.

    queries содержит qid исходного eval_queries и query_id. Пустые пулы остаются
    в среднем с нулевым Recall. label из candidates в расчёте не используется.
    """
    ks = tuple(sorted(set(ks)))
    if not ks or any(not isinstance(k, (int, np.integer)) or k < 1 for k in ks):
        raise ValueError("ks must contain positive integer cutoffs")
    if queries["qid"].duplicated().any() or queries["query_id"].duplicated().any():
        raise ValueError("queries must have unique qid and query_id")
    if candidates[["qid", "item_idx"]].duplicated().any():
        raise ValueError("duplicate query-item candidate")
    if not set(candidates["qid"]).issubset(set(queries["qid"])):
        raise ValueError("candidate qid outside selected queries")
    if not pd.api.types.is_integer_dtype(candidates["item_idx"]):
        raise ValueError("item_idx must be integer")
    indices = candidates["item_idx"].to_numpy()
    if np.any(indices < 0) or np.any(indices >= len(corpus)):
        raise ValueError("candidate item_idx outside corpus")
    values = candidates[score_column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("candidate scores must be finite")
    if corpus["item_id"].duplicated().any():
        raise ValueError("corpus item_id must be unique")

    truth_by_query = truth.groupby("query_id", sort=False)["item_id"].agg(set).to_dict()
    seen_by_query = None
    if "seen_item" in truth:
        seen_by_query = (
            truth.groupby("query_id", sort=False)["seen_item"]
            .agg(
                lambda values: (
                    "all_seen"
                    if values.all()
                    else "mixed" if values.any() else "all_unseen"
                )
            )
            .to_dict()
        )
    order = np.lexsort((indices, -values, candidates["qid"].to_numpy()))
    ranked = candidates.iloc[order]
    ranked_by_query = {
        int(qid): group["item_idx"].to_numpy()
        for qid, group in ranked.groupby("qid", sort=False)
    }
    item_ids = corpus["item_id"].to_numpy()
    rows = []
    for query in queries.to_dict("records"):
        qid, query_id = query["qid"], query["query_id"]
        relevant = truth_by_query.get(query_id, set())
        if not relevant:
            raise ValueError(f"query {query_id!r} has no ground-truth relevant items")
        found = item_ids[ranked_by_query.get(qid, np.empty(0, dtype=np.int32))]
        hits = np.fromiter((item_id in relevant for item_id in found), dtype=np.int32)
        row = {
            "qid": qid,
            "query_id": query_id,
            "relevant_count": len(relevant),
            "candidate_count": len(found),
            "recall@union": float(hits.sum() / len(relevant)),
        }
        row.update({f"recall@{k}": float(hits[:k].sum() / len(relevant)) for k in ks})
        for name in ("regime", "seen_text", "seen_context"):
            if name in query:
                row[name] = query[name]
        filters = query.get("search_infm_params_text", "")
        row["filter_empty"] = not isinstance(filters, str) or not filters.strip()
        if seen_by_query is not None:
            row["seen_item_group"] = seen_by_query[query_id]
        rows.append(row)
    per_query = pd.DataFrame(rows)
    metric_names = [f"recall@{k}" for k in ks] + ["recall@union"]
    summary = {"queries": len(queries)}
    summary.update(
        {
            name: float(per_query[name].mean()) if len(per_query) else 0.0
            for name in metric_names
        }
    )
    summary["candidate_count_mean"] = (
        float(per_query["candidate_count"].mean()) if len(per_query) else 0.0
    )
    summary["groups"] = {}
    for column in (
        "regime",
        "seen_text",
        "seen_context",
        "filter_empty",
        "seen_item_group",
    ):
        if column in per_query:
            summary["groups"][column] = {
                str(value): {
                    "queries": len(group),
                    **{name: float(group[name].mean()) for name in metric_names},
                }
                for value, group in per_query.groupby(column, dropna=False)
            }
    if reference_empty_fraction is not None:
        fraction = float(reference_empty_fraction)
        if not 0 <= fraction <= 1:
            raise ValueError("reference_empty_fraction must be between 0 and 1")
        summary["reference_empty_fraction"] = fraction
        groups = summary["groups"].get("filter_empty", {})
        for name in metric_names:
            weighted = 0.0
            for empty, weight in (("True", fraction), ("False", 1 - fraction)):
                if not weight:
                    continue
                if empty not in groups:
                    weighted = None
                    break
                weighted += weight * groups[empty][name]
            # Отсутствующую в dev группу нельзя заменить нулём или выдуманной оценкой.
            summary[f"poststratified_{name}"] = weighted
    return summary, per_query
