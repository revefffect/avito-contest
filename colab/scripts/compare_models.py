"""Сравнить модели и их небольшие ансамбли на одном наборе кандидатов."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from avito_retrieval.evaluation import (
    RESERVED_COLUMNS,
    evaluate_candidates,
    load_split_queries,
    reference_empty_fraction,
)

# Соседний CLI содержит ту же нормализацию, которая применяется при отправке.
if __package__:
    from .predict import score_models, sha256_file, standardize_by_query
else:
    from predict import score_models, sha256_file, standardize_by_query


def paired_bootstrap(current, baseline, fraction, iterations=2000, seed=42):
    """Сравнивать одинаковые запросы, сохраняя парность предсказаний."""
    if baseline["query_id"].duplicated().any():
        raise ValueError("baseline must have exactly one row per query_id")
    reference = baseline.set_index("query_id")
    if not set(current["query_id"]).issubset(reference.index):
        raise ValueError("baseline is missing selected query IDs")
    old = reference.loc[current["query_id"], "recall@50"].to_numpy(dtype=np.float64)
    if not np.isfinite(old).all() or (old < 0).any() or (old > 1).any():
        raise ValueError("baseline Recall@50 must be finite and between zero and one")
    delta = current["recall@50"].to_numpy() - old
    if not len(delta):
        raise ValueError("bootstrap requires at least one query")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(delta), size=(iterations, len(delta)))
    draws = delta[samples].mean(axis=1)
    result = {
        "recall@50": {
            "delta": float(delta.mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "bootstrap_positive_fraction": float((draws > 0).mean()),
        }
    }
    del samples
    # Для переноса на benchmark сохраняем его доли групп и ресэмплируем внутри групп.
    weighted_delta, weighted_draws = 0.0, np.zeros(iterations)
    empty = current["filter_empty"].to_numpy(dtype=bool)
    for mask, weight in ((empty, fraction), (~empty, 1 - fraction)):
        if weight == 0:
            continue
        values = delta[mask]
        if not len(values):
            result["poststratified_recall@50"] = None
            return result
        samples = rng.integers(0, len(values), size=(iterations, len(values)))
        weighted_delta += weight * values.mean()
        weighted_draws += weight * values[samples].mean(axis=1)
    result["poststratified_recall@50"] = {
        "delta": float(weighted_delta),
        "ci95": np.quantile(weighted_draws, [0.025, 0.975]).tolist(),
        "bootstrap_positive_fraction": float((weighted_draws > 0).mean()),
    }
    return result


def required_features(paths):
    result = {"qid", "item_idx"}
    for path in paths:
        names = json.loads(
            path.with_suffix(".feature_names.json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) for name in names)
        ):
            raise ValueError(f"invalid model feature names for {path}")
        if len(names) != len(set(names)) or set(names) & RESERVED_COLUMNS:
            raise ValueError(f"duplicate or reserved model feature names for {path}")
        result.update(names)
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/comparison.json"))
    parser.add_argument("--split", choices=["dev", "audit"], default="dev")
    parser.add_argument(
        "--queries", type=Path, default=Path("data/processed/eval_queries.parquet")
    )
    parser.add_argument(
        "--truth", type=Path, default=Path("data/processed/eval_truth.parquet")
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/processed/corpus.parquet")
    )
    parser.add_argument(
        "--reference-queries",
        type=Path,
        default=Path("data/raw/benchmark_queries.parquet"),
    )
    parser.add_argument("--baseline-per-query", type=Path)
    parser.add_argument(
        "--blend-grid",
        type=float,
        nargs="+",
        help="Weights of the first model, two models only, default: 0.25 0.5 0.75",
    )
    parser.add_argument("--no-blends", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    names = [path.stem for path in args.model]
    if len(names) != len(set(names)):
        parser.error("model filenames must have different stems")
    if args.blend_grid is not None and len(args.model) != 2:
        parser.error("blend-grid requires exactly two models")
    grid = args.blend_grid if args.blend_grid is not None else [0.25, 0.5, 0.75]
    if any(not np.isfinite(value) or not 0 < value < 1 for value in grid):
        parser.error("blend-grid weights must be finite and between zero and one")
    queries = load_split_queries(args.queries, [args.split])
    if queries.empty:
        raise ValueError("the selected split contains no queries")
    features = pd.read_parquet(
        args.features,
        columns=required_features(args.model),
        filters=[("qid", "in", queries["qid"].tolist())],
        use_threads=False,
    )
    truth = pd.read_parquet(
        args.truth, filters=[("split", "=", args.split)], use_threads=False
    )
    corpus = pd.read_parquet(args.corpus, columns=["item_id"], use_threads=False)
    fraction = reference_empty_fraction(args.reference_queries)
    candidates = features[["qid", "item_idx"]].copy()
    scores, model_metadata = {}, {}
    for name, path in zip(names, args.model):
        scores[name], records = score_models(
            features, [path], [1.0], len(queries), args.threads
        )
        model_metadata[name] = {"kind": "model", "models": records}
    if len(names) > 1 and not args.no_blends:
        present, dense_qids = np.unique(features["qid"].to_numpy(), return_inverse=True)
        normalized = {
            name: standardize_by_query(scores[name], dense_qids, len(present))
            for name in names
        }
        weights_to_try = (
            [[value, 1 - value] for value in grid]
            if len(names) == 2
            else [[1 / len(names)] * len(names)]
        )
        for weights in weights_to_try:
            blend_name = "blend_" + "_".join(f"{weight:g}" for weight in weights)
            scores[blend_name] = sum(
                weight * normalized[name] for weight, name in zip(weights, names)
            )
            model_metadata[blend_name] = {
                "kind": "within_query_zscore",
                "model_names": names,
                "weights": weights,
            }
    summaries, per_query_tables = {}, {}
    for name, values in scores.items():
        candidates["score"] = values
        summary, per_query = evaluate_candidates(
            candidates,
            queries,
            truth,
            corpus,
            reference_empty_fraction=fraction,
        )
        summaries[name], per_query_tables[name] = summary, per_query
        print(
            json.dumps(
                {
                    "model": name,
                    "recall@50": summary["recall@50"],
                    "poststratified_recall@50": summary["poststratified_recall@50"],
                }
            ),
            flush=True,
        )
    if args.baseline_per_query:
        baseline = pd.read_parquet(args.baseline_per_query, use_threads=False)
        baseline_metadata = {
            "path": str(args.baseline_per_query),
            "sha256": sha256_file(args.baseline_per_query),
        }
    else:
        baseline = per_query_tables[names[0]]
        baseline_metadata = {"model_name": names[0]}
    for name, table in per_query_tables.items():
        summaries[name]["paired_delta_vs_baseline"] = paired_bootstrap(
            table,
            baseline,
            fraction,
            iterations=args.bootstrap,
            seed=args.seed,
        )
    report = {
        "split": args.split,
        "query_count": len(queries),
        "candidate_rows": len(features),
        "features_path": str(args.features),
        "features_sha256": sha256_file(args.features),
        "reference_empty_fraction": fraction,
        "baseline": baseline_metadata,
        "bootstrap": {
            "samples": args.bootstrap,
            "seed": args.seed,
            "interval": "paired percentile 95%, poststratified resampling within filter groups",
        },
        "model_metadata": model_metadata,
        "results": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.concat(
        [table.assign(model_name=name) for name, table in per_query_tables.items()],
        ignore_index=True,
    ).to_parquet(args.output.with_suffix(".per_query.parquet"), index=False)


if __name__ == "__main__":
    main()
