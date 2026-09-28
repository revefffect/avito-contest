"""Готовим признаки кандидатов; обучение и выбор конфигурации идут отдельно."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from avito_retrieval.behavior import BehaviorIndex
from avito_retrieval.candidates import CandidateBuilder
from avito_retrieval.geography import Geography


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["validation", "benchmark"], default="validation"
    )
    parser.add_argument("--dense", action="store_true")
    parser.add_argument(
        "--plain-query",
        action="store_true",
        help="Добавить E5-представление запроса без фильтров",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.plain_query and not args.dense:
        parser.error("--plain-query requires --dense")
    threadpool_limits(limits=3)
    items = pd.read_parquet("data/processed/corpus.parquet")
    if args.mode == "validation":
        train_path = "data/processed/retrieval_train.parquet"
        train = pd.read_parquet(train_path)
        queries = pd.read_parquet("data/processed/eval_queries.parquet")
        truth = pd.read_parquet("data/processed/eval_truth.parquet")
        prefix = None
    else:
        train_path = "data/raw/train.parquet"
        columns = [
            "search_query",
            "search_location_id",
            "search_is_delivery_search",
            "search_infm_params_text",
            "search_category",
            "item_id",
            "item_microcat_id",
            "item_location_id",
        ]
        train = pd.read_parquet("data/raw/train.parquet", columns=columns)
        queries = pd.read_parquet("data/raw/benchmark_queries.parquet")
        truth = None
        prefix = len(
            pd.read_parquet("data/raw/benchmark_items.parquet", columns=["item_id"])
        )
    queries = queries.iloc[: args.limit] if args.limit else queries
    knowledge_path = Path(f"artifacts/knowledge_{args.mode}.joblib")
    metadata_path = knowledge_path.with_suffix(".json")
    fingerprint = {
        "train_sha256": file_hash(train_path),
        "corpus_sha256": file_hash("data/processed/corpus.parquet"),
        "behavior_code_sha256": file_hash("src/avito_retrieval/behavior.py"),
        "geography_code_sha256": file_hash("src/avito_retrieval/geography.py"),
    }
    if (
        knowledge_path.exists()
        and metadata_path.exists()
        and json.loads(metadata_path.read_text()) == fingerprint
    ):
        behavior, geography = joblib.load(knowledge_path)
    else:
        print("Обучаем статистики запросов и географии", flush=True)
        behavior = BehaviorIndex().fit(train, items)
        geography = Geography().fit(train, items)
        joblib.dump((behavior, geography), knowledge_path)
        metadata_path.write_text(json.dumps(fingerprint, indent=2) + "\n")
    print("Загружаем текстовый индекс", flush=True)
    lexical = joblib.load("artifacts/lexical.joblib", mmap_mode="r")
    item_embeddings = query_embeddings = None
    if args.dense:
        from avito_retrieval.dense import load_embeddings

        item_embeddings, item_ids = load_embeddings(Path("artifacts/dense"), "items")
        query_name = (
            "eval_queries" if args.mode == "validation" else "benchmark_queries"
        )
        query_embeddings, query_ids = load_embeddings(
            Path("artifacts/dense"), query_name
        )
        if (
            list(item_ids) != items["item_id"].tolist()
            or list(query_ids[: len(queries)]) != queries["query_id"].tolist()
        ):
            raise ValueError("Embedding IDs differ from source data")
        query_embeddings = {"dense": query_embeddings[: len(queries)]}
        if args.plain_query:
            plain, plain_ids = load_embeddings(
                Path("artifacts/dense"), query_name + "_plain"
            )
            if list(plain_ids[: len(queries)]) != queries["query_id"].tolist():
                raise ValueError("Plain query embedding IDs differ from source data")
            query_embeddings["dense_plain"] = plain[: len(queries)]
    builder = CandidateBuilder(items, lexical, behavior, geography, item_embeddings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("Собираем кандидатов и признаки", flush=True)
    builder.write_features(queries, args.output, truth, query_embeddings, prefix)
    query_path = (
        "data/processed/eval_queries.parquet"
        if args.mode == "validation"
        else "data/raw/benchmark_queries.parquet"
    )
    metadata = {
        "mode": args.mode,
        "dense": args.dense,
        "plain_query": args.plain_query,
        "query_count": len(queries),
        "allowed_n_items": prefix or len(items),
        "queries_sha256": file_hash(query_path),
        "knowledge": fingerprint,
        "candidate_code_sha256": file_hash("src/avito_retrieval/candidates.py"),
        "features_sha256": file_hash(args.output),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
