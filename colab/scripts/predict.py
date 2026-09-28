"""Получить answer.csv из кандидатов и проверить файл перед отправкой."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoost

from avito_retrieval.evaluation import RESERVED_COLUMNS
from avito_retrieval.submission import validate_submission, write_submission


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidates(features, queries, corpus, allowed_items):
    """Проверить позиционные индексы до обращения к модели."""
    if queries.empty:
        raise ValueError("benchmark queries must not be empty")
    empty = pd.DataFrame({"query_id": queries["query_id"], "answer": ""})
    validate_submission(empty, queries, allowed_items)
    allowed_count = len(allowed_items)
    if (
        corpus["item_id"].iloc[:allowed_count].tolist()
        != allowed_items["item_id"].tolist()
    ):
        raise ValueError("corpus prefix does not match the benchmark item order")
    for name in ("qid", "item_idx"):
        if (
            name not in features
            or not pd.api.types.is_integer_dtype(features[name])
            or features[name].isna().any()
        ):
            raise ValueError(f"{name} must contain integer row positions")
    actual = set(features["qid"])
    expected = set(range(len(queries)))
    if actual != expected:
        raise ValueError(
            f"candidate qid coverage mismatch: {len(expected - actual)} missing, "
            f"{len(actual - expected)} extra"
        )
    if features[["qid", "item_idx"]].duplicated().any():
        raise ValueError("duplicate query-item candidate")
    if not features["item_idx"].between(0, allowed_count - 1).all():
        raise ValueError("candidate item_idx outside the benchmark corpus prefix")


def standardize_by_query(scores, qids, query_count):
    """Привести модели к общей шкале только внутри каждого запроса."""
    counts = np.bincount(qids, minlength=query_count)
    means = np.bincount(qids, weights=scores, minlength=query_count) / counts
    centered = scores - means[qids]
    variance = (
        np.bincount(qids, weights=centered * centered, minlength=query_count) / counts
    )
    scale = np.sqrt(variance[qids])
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)


def normalize_weights(values, model_count):
    weights = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        total = weights.sum()
    if (
        weights.ndim != 1
        or len(weights) != model_count
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isfinite(total)
        or total <= 0
    ):
        raise ValueError(
            "weights must match the model count, be finite and non-negative, "
            "and have a positive finite sum"
        )
    return weights / total


def score_models(features, model_paths, weights, query_count, threads):
    qids = features["qid"].to_numpy()
    combined = np.zeros(len(features), dtype=np.float64)
    records = []
    for path, weight in zip(model_paths, weights):
        names_path = path.with_suffix(".feature_names.json")
        names = json.loads(names_path.read_text(encoding="utf-8"))
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) for name in names)
        ):
            raise ValueError(f"invalid feature-name list: {names_path}")
        if len(names) != len(set(names)) or set(names) & RESERVED_COLUMNS:
            raise ValueError(f"duplicate or reserved model feature names: {names_path}")
        missing = set(names) - set(features.columns)
        if missing:
            raise ValueError(f"missing model features: {sorted(missing)}")
        for name in names:
            values = features[name]
            if (
                not pd.api.types.is_numeric_dtype(values)
                or not np.isfinite(values.to_numpy()).all()
            ):
                raise ValueError(f"model feature {name!r} must be numeric and finite")
        model = CatBoost()
        model.load_model(path)
        if model.feature_names_ != names:
            raise ValueError(f"feature-name sidecar does not match model {path}")
        scores = np.asarray(
            model.predict(
                features[names], prediction_type="RawFormulaVal", thread_count=threads
            ),
            dtype=np.float64,
        ).reshape(-1)
        if scores.shape != combined.shape or not np.isfinite(scores).all():
            raise ValueError(f"model {path} returned invalid prediction scores")
        if len(model_paths) > 1:
            scores = standardize_by_query(scores, qids, query_count)
        combined += weight * scores
        records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "weight": float(weight),
                "feature_names_sha256": sha256_file(names_path),
                "feature_count": len(names),
                "trees": model.tree_count_,
            }
        )
    if not np.isfinite(combined).all():
        raise ValueError("combined prediction scores must be finite")
    return combined, records


def top_predictions(features, scores, item_ids, query_count):
    qids = features["qid"].to_numpy()
    item_indices = features["item_idx"].to_numpy()
    # Порядок кандидатов и разбиение предсказаний на потоки не должны менять ничьи.
    order = np.lexsort((item_indices, -scores, qids))
    sorted_qids = qids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_qids)) + 1]
    ends = np.r_[starts[1:], len(order)]
    predictions = [[] for _ in range(query_count)]
    for start, end in zip(starts, ends):
        qid = sorted_qids[start]
        selected = item_indices[order[start : min(start + 50, end)]]
        predictions[qid] = item_ids[selected].tolist()
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        required=True,
        help="Repeat this option to combine several models",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help="One non-negative weight per model, equal weights by default",
    )
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument(
        "--queries", type=Path, default=Path("data/raw/benchmark_queries.parquet")
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/processed/corpus.parquet")
    )
    parser.add_argument(
        "--allowed-items", type=Path, default=Path("data/raw/benchmark_items.parquet")
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        weights = normalize_weights(
            args.weights if args.weights is not None else [1.0] * len(args.model),
            len(args.model),
        )
    except ValueError as error:
        parser.error(str(error))
    queries = pd.read_parquet(args.queries, columns=["query_id"], use_threads=False)
    corpus = pd.read_parquet(args.corpus, columns=["item_id"], use_threads=False)
    allowed_items = pd.read_parquet(
        args.allowed_items, columns=["item_id"], use_threads=False
    )
    features = pd.read_parquet(args.features, use_threads=False)
    validate_candidates(features, queries, corpus, allowed_items)
    scores, models = score_models(
        features, args.model, weights, len(queries), args.threads
    )
    predictions = top_predictions(
        features, scores, corpus["item_id"].to_numpy(), len(queries)
    )
    write_submission(queries["query_id"], predictions, args.output, allowed_items)
    # Повторное чтение проверяет сохранённый CSV, а не только данные в памяти.
    restored = pd.read_csv(
        args.output, dtype=str, keep_default_na=False, encoding="utf-8"
    )
    submission = validate_submission(restored, queries, allowed_items)
    counts = np.bincount(features["qid"].to_numpy(), minlength=len(queries))
    report = {
        "submission": submission,
        "output_path": str(args.output),
        "csv_sha256": sha256_file(args.output),
        "features_path": str(args.features),
        "features_sha256": sha256_file(args.features),
        "models": models,
        "combination": "within_query_zscore" if len(models) > 1 else "raw_model_score",
        "queries_sha256": sha256_file(args.queries),
        "allowed_items_sha256": sha256_file(args.allowed_items),
        "candidate_pool": {
            "rows": len(features),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "queries_with_fewer_than_50": int((counts < 50).sum()),
        },
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
