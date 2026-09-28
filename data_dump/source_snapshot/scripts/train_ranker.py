"""Обучить ранжирование на fit и выбрать число деревьев по Recall@50 на dev."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from avito_retrieval.evaluation import (
    evaluate_candidates,
    feature_columns,
    load_split_queries,
    reference_empty_fraction,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        "--loss", choices=["QuerySoftMax", "Logloss"], default="QuerySoftMax"
    )
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.07)
    parser.add_argument("--checkpoint-period", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drop-features", default="", help="Comma-separated feature names"
    )
    parser.add_argument("--fit-splits", default="fit", help="fit or fit,dev")
    parser.add_argument(
        "--fixed-iterations",
        action="store_true",
        help="Use the requested tree count without dev selection",
    )
    parser.add_argument(
        "--reference-queries",
        type=Path,
        default=Path("data/raw/benchmark_queries.parquet"),
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.checkpoint_period < 1:
        parser.error("iterations and checkpoint-period must be positive")
    fit_splits = [value.strip() for value in args.fit_splits.split(",")]
    if set(fit_splits) not in ({"fit"}, {"fit", "dev"}) or len(fit_splits) != len(
        set(fit_splits)
    ):
        parser.error(
            "fit-splits must be fit or fit,dev; audit is never a training split"
        )
    if "dev" in fit_splits and not args.fixed_iterations:
        parser.error("including dev in training requires --fixed-iterations")
    start = time.monotonic()
    queries = load_split_queries(
        args.queries, fit_splits if args.fixed_iterations else ["fit", "dev"]
    )
    dev_queries = queries.loc[queries["split"].eq("dev")].copy()
    fit_qids = set(queries.loc[queries["split"].isin(fit_splits), "qid"])
    dev_qids = set() if args.fixed_iterations else set(dev_queries["qid"])
    if not fit_qids or (not args.fixed_iterations and not dev_qids):
        raise ValueError("both fit and dev queries are required")
    # Audit-строки не поступают ни в обучение, ни в подбор числа деревьев.
    features = pd.read_parquet(
        args.features, filters=[("qid", "in", sorted(fit_qids | dev_qids))]
    )
    expected_qids = fit_qids | dev_qids
    if set(features["qid"]) != expected_qids:
        missing = expected_qids - set(features["qid"])
        raise ValueError(
            f"candidate qid coverage mismatch: {len(missing)} missing; examples={sorted(missing)[:5]}"
        )
    for name in ("qid", "item_idx"):
        if not pd.api.types.is_integer_dtype(features[name]):
            raise ValueError(f"{name} must be integer")
    if features[["qid", "item_idx"]].duplicated().any():
        raise ValueError("duplicate query-item candidate")
    names = feature_columns(features)
    dropped = {name.strip() for name in args.drop_features.split(",") if name.strip()}
    unknown = dropped - set(names)
    if unknown:
        raise ValueError(f"unknown drop-features: {sorted(unknown)}")
    names = [name for name in names if name not in dropped]
    if not names:
        raise ValueError("drop-features removed every model feature")
    invalid = [
        name for name in names if not np.isfinite(features[name].to_numpy()).all()
    ]
    if invalid:
        raise ValueError(f"non-finite model features: {invalid}")
    if not features["label"].isin([0, 1]).all():
        raise ValueError("label must be binary")
    fit = features.loc[features["qid"].isin(fit_qids)].sort_values(["qid", "item_idx"])
    dev = features.loc[features["qid"].isin(dev_qids)].sort_values(["qid", "item_idx"])
    del features
    corpus = pd.read_parquet(args.corpus, columns=["item_id"])
    for frame in (fit, dev):
        if not frame["item_idx"].between(0, len(corpus) - 1).all():
            raise ValueError("candidate item_idx outside corpus")
    parameters = dict(
        loss_function=args.loss,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        random_seed=args.seed,
        thread_count=args.threads,
        allow_writing_files=False,
    )
    if args.loss == "QuerySoftMax":
        model = CatBoostRanker(**parameters)
        fit_pool = Pool(fit[names], label=fit["label"], group_id=fit["qid"])
    else:
        model = CatBoostClassifier(**parameters)
        fit_pool = Pool(fit[names], label=fit["label"])
    dev_pool = Pool(dev[names]) if not args.fixed_iterations else None
    print(
        json.dumps(
            {"fit_rows": len(fit), "dev_rows": len(dev), "features": len(names)}
        ),
        flush=True,
    )
    model.fit(fit_pool, verbose=100)
    del fit, fit_pool
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "fit_query_count": len(fit_qids),
            "dev_query_count": len(dev_qids),
            "dropped_features": sorted(dropped),
        }
    )
    if args.fixed_iterations:
        # Финальный retrain использует заранее выбранное число деревьев, без повторного подбора.
        config.update(
            {
                "selected_iteration": model.tree_count_,
                "selected_dev_recall@50": None,
                "seconds": round(time.monotonic() - start, 3),
            }
        )
        save_model(model, names, config, [], args.output)
        print(json.dumps(config, indent=2), flush=True)
        return
    truth = pd.read_parquet(args.truth, filters=[("split", "=", "dev")])
    fraction = reference_empty_fraction(args.reference_queries)
    predictions = model.staged_predict(
        dev_pool,
        eval_period=args.checkpoint_period,
        thread_count=args.threads,
        **({"prediction_type": "RawFormulaVal"} if args.loss == "Logloss" else {}),
    )
    candidates = dev[["qid", "item_idx"]].copy()
    checkpoints = []
    best_recall, best_iteration = -1.0, 0
    for number, scores in enumerate(predictions, start=1):
        iteration = min(number * args.checkpoint_period, model.tree_count_)
        candidates["score"] = np.asarray(scores).reshape(-1)
        summary, _ = evaluate_candidates(
            candidates, dev_queries, truth, corpus, reference_empty_fraction=fraction
        )
        record = {"iteration": iteration, **summary}
        checkpoints.append(record)
        print(
            json.dumps({"iteration": iteration, "dev_recall@50": summary["recall@50"]}),
            flush=True,
        )
        # Равный Recall оставляет более простую модель с меньшим числом деревьев.
        if summary["recall@50"] > best_recall:
            best_recall, best_iteration = summary["recall@50"], iteration
    if best_iteration == 0:
        raise RuntimeError("CatBoost produced no evaluation checkpoints")
    model.shrink(ntree_end=best_iteration)
    config.update(
        {
            "selected_iteration": best_iteration,
            "selected_dev_recall@50": best_recall,
            "reference_empty_fraction": fraction,
            "seconds": round(time.monotonic() - start, 3),
        }
    )
    scores = model.predict(
        dev_pool,
        thread_count=args.threads,
        **({"prediction_type": "RawFormulaVal"} if args.loss == "Logloss" else {}),
    )
    candidates["score"] = np.asarray(scores).reshape(-1)
    selected_summary, per_query = evaluate_candidates(
        candidates, dev_queries, truth, corpus, reference_empty_fraction=fraction
    )
    if not np.isclose(selected_summary["recall@50"], best_recall, rtol=0, atol=1e-12):
        raise RuntimeError("shrunk model does not reproduce selected Recall@50")
    save_model(model, names, config, checkpoints, args.output)
    per_query.to_parquet(args.output.with_suffix(".dev_queries.parquet"), index=False)
    print(json.dumps(config, indent=2), flush=True)


def save_model(model, names, config, checkpoints, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output)
    output.with_suffix(".feature_names.json").write_text(
        json.dumps(names, indent=2) + "\n"
    )
    output.with_suffix(".train_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    output.with_suffix(".evaluation.json").write_text(
        json.dumps(
            {
                "selected_iteration": config["selected_iteration"],
                "checkpoints": checkpoints,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
