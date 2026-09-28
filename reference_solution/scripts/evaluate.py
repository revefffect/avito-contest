"""Оценить готовую модель или эвристику на явно выбранном подмножестве запросов."""

import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoost

from avito_retrieval.evaluation import (
    evaluate_candidates,
    load_split_queries,
    reference_empty_fraction,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--score-column")
    parser.add_argument("--feature-names", type=Path)
    parser.add_argument("--split", choices=["fit", "dev", "audit"], default="dev")
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
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--reference-queries",
        type=Path,
        default=Path("data/raw/benchmark_queries.parquet"),
    )
    args = parser.parse_args()
    queries = load_split_queries(args.queries, [args.split])
    features = pd.read_parquet(
        args.features, filters=[("qid", "in", queries["qid"].tolist())]
    )
    truth = pd.read_parquet(args.truth, filters=[("split", "=", args.split)])
    corpus = pd.read_parquet(args.corpus, columns=["item_id"])
    candidates = features[["qid", "item_idx"]].copy()
    if args.model:
        names_path = args.feature_names or args.model.with_suffix(".feature_names.json")
        names = json.loads(names_path.read_text())
        model = CatBoost()
        model.load_model(args.model)
        candidates["score"] = model.predict(
            features[names], prediction_type="RawFormulaVal", thread_count=args.threads
        )
    else:
        candidates["score"] = features[args.score_column]
    fraction = reference_empty_fraction(args.reference_queries)
    summary, per_query = evaluate_candidates(
        candidates, queries, truth, corpus, reference_empty_fraction=fraction
    )
    summary.update(
        {
            "split": args.split,
            "features_file": str(args.features),
            "model": str(args.model) if args.model else None,
            "score_column": args.score_column,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    per_query.to_parquet(args.output.with_suffix(".queries.parquet"), index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
