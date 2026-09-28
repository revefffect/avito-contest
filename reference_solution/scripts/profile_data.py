"""Проверяем структуру данных до выбора схемы валидации."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def normalize_text(values):
    return values.fillna("").str.lower().str.replace("ё", "е").str.split().str.join(" ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/data_profile.json")
    )
    args = parser.parse_args()

    schemas = {}
    for name in ["train", "benchmark_queries", "benchmark_items"]:
        file = pq.ParquetFile(args.data_dir / f"{name}.parquet")
        schemas[name] = {
            "rows": file.metadata.num_rows,
            "schema": str(file.schema_arrow),
        }

    query_columns = [
        c
        for c in pq.read_schema(args.data_dir / "train.parquet").names
        if c.startswith("search_")
    ]
    extra = ["item_id", "item_location_id", "item_category_id", "item_microcat_id"]
    train = pd.read_parquet(
        args.data_dir / "train.parquet", columns=query_columns + extra
    )
    queries = pd.read_parquet(args.data_dir / "benchmark_queries.parquet")
    items = pd.read_parquet(args.data_dir / "benchmark_items.parquet", columns=extra)
    train["normalized_query"] = normalize_text(train["search_query"])
    queries["normalized_query"] = normalize_text(queries["search_query"])
    in_corpus = train["item_id"].isin(items["item_id"])
    contexts = train[query_columns].fillna("").astype(str).agg("\x1f".join, axis=1)
    test_contexts = (
        queries[query_columns].fillna("").astype(str).agg("\x1f".join, axis=1)
    )
    pairs = train.assign(context=contexts).drop_duplicates(["context", "item_id"])
    counts = pairs.groupby("context").size()

    report = {
        "schemas": schemas,
        "train_nulls": train.isna().sum().to_dict(),
        "queries_nulls": queries.isna().sum().to_dict(),
        "unique_train_items": train["item_id"].nunique(),
        "unique_corpus_items": items["item_id"].nunique(),
        "unique_train_texts": train["normalized_query"].nunique(),
        "unique_benchmark_texts": queries["normalized_query"].nunique(),
        "unique_train_contexts": contexts.nunique(),
        "unique_benchmark_contexts": test_contexts.nunique(),
        "unique_context_item_pairs": len(pairs),
        "train_rows_in_corpus": int(in_corpus.sum()),
        "train_items_in_corpus": train.loc[in_corpus, "item_id"].nunique(),
        "corpus_items_seen_in_train": float(
            items["item_id"].isin(train["item_id"]).mean()
        ),
        "benchmark_seen_text_fraction": float(
            queries["normalized_query"].isin(train["normalized_query"]).mean()
        ),
        "benchmark_seen_context_fraction": float(test_contexts.isin(contexts).mean()),
        "train_search_item_same_location": float(
            train["search_location_id"].eq(train["item_location_id"]).mean()
        ),
        "items_per_context_quantiles": counts.quantile(
            [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
        ).to_dict(),
        "search_categories_train": train["search_category"]
        .value_counts(dropna=False)
        .head(30)
        .to_dict(),
        "search_categories_benchmark": queries["search_category"]
        .value_counts(dropna=False)
        .head(30)
        .to_dict(),
        "top_query_texts": train["normalized_query"].value_counts().head(40).to_dict(),
        "benchmark_sample": queries.sample(min(15, len(queries)), random_state=42)
        .fillna("")
        .to_dict("records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"benchmark_sample", "schemas"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
