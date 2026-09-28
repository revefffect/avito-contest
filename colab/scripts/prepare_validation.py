"""Готовим отложенные запросы с новыми текстами и объявлениями."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from avito_retrieval.text import normalize, query_keys


SEARCH_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
META_COLUMNS = ["item_id", "item_location_id", "item_category_id", "item_microcat_id"]


def make_validation(data_dir, output_dir, count=6000, seed=20260928):
    rng = np.random.default_rng(seed)
    train = pd.read_parquet(
        data_dir / "train.parquet", columns=SEARCH_COLUMNS + META_COLUMNS
    )
    train["text_key"] = train["search_query"].map(normalize)
    train["context_key"] = query_keys(train)
    pairs = train.drop_duplicates(["context_key", "item_id"])
    contexts = pairs.groupby("context_key", sort=True).agg(
        text_key=("text_key", "first"),
        positives=("item_id", list),
    )
    text_contexts = contexts.groupby("text_key", sort=True).size()
    selected = []
    used_texts = set()

    # Доли отражают наблюдаемую новизну benchmark, но не используют его ответы.
    for mode, size in [
        ("warm_context", round(count * 0.044)),
        ("warm_text", round(count * 0.33)),
        ("cold_text", count - round(count * 0.044) - round(count * 0.33)),
    ]:
        available = contexts[~contexts["text_key"].isin(used_texts)]
        if mode == "warm_context":
            available = available[available["positives"].map(len) >= 2]
        elif mode == "warm_text":
            available = available[available["text_key"].map(text_contexts) >= 2]
        by_text = available.groupby("text_key", sort=True).indices
        chosen_texts = rng.choice(sorted(by_text), size=size, replace=False)
        for text in chosen_texts:
            positions = by_text[text]
            row = available.iloc[int(rng.choice(positions))]
            targets = sorted(row["positives"])
            if mode == "warm_context":
                targets = sorted(
                    rng.choice(
                        targets, max(1, len(targets) // 2), replace=False
                    ).tolist()
                )
            selected.append(
                {
                    "context_key": row.name,
                    "text_key": text,
                    "regime": mode,
                    "targets": targets,
                }
            )
        used_texts.update(chosen_texts)

    selected = (
        pd.DataFrame(selected).iloc[rng.permutation(count)].reset_index(drop=True)
    )
    selected["split"] = np.repeat(
        ["fit", "dev", "audit"],
        [count // 2, count // 4, count - count // 2 - count // 4],
    )
    selected["query_id"] = selected["context_key"].map(
        lambda key: hashlib.sha256(key.encode()).hexdigest()[:16]
    )
    query_features = train.drop_duplicates("context_key").set_index("context_key")[
        SEARCH_COLUMNS
    ]
    queries = selected.drop(columns="targets").join(query_features, on="context_key")
    truth = (
        selected[["query_id", "split", "targets"]]
        .explode("targets")
        .rename(columns={"targets": "item_id"})
    )

    cold_texts = set(selected.loc[selected["regime"] == "cold_text", "text_key"])
    target_ids = sorted(set(truth["item_id"]))
    cold_items = set(
        rng.choice(target_ids, round(len(target_ids) * 0.904), replace=False)
    )
    hidden_pairs = set(
        zip(
            selected.loc[
                selected.index.repeat(selected["targets"].map(len)), "context_key"
            ],
            selected["targets"].explode(),
        )
    )
    forbidden_pair = pd.MultiIndex.from_frame(train[["context_key", "item_id"]]).isin(
        hidden_pairs
    )
    keep = (
        ~train["text_key"].isin(cold_texts)
        & ~train["item_id"].isin(cold_items)
        & ~forbidden_pair
    )
    retrieval_train = train.loc[keep].copy()

    # Корпус сохраняет тексты отложенных объявлений, но их выборы скрыты от обучения.
    # Негативы - весь реальный корпус, а не облегчённая случайная подвыборка.
    items = pd.read_parquet(data_dir / "benchmark_items.parquet")
    extra_ids = sorted(set(target_ids) - set(items["item_id"]))
    extra_items = pd.read_parquet(
        data_dir / "train.parquet", filters=[("item_id", "in", extra_ids)]
    )
    extra_items = extra_items[items.columns].drop_duplicates("item_id")
    corpus = pd.concat([items, extra_items], ignore_index=True)
    queries["seen_text"] = queries["text_key"].isin(retrieval_train["text_key"])
    queries["seen_context"] = queries["context_key"].isin(
        retrieval_train["context_key"]
    )
    truth["seen_item"] = truth["item_id"].isin(retrieval_train["item_id"])

    assert (
        not pd.MultiIndex.from_frame(retrieval_train[["context_key", "item_id"]])
        .isin(hidden_pairs)
        .any()
    )
    assert corpus["item_id"].is_unique
    assert truth["item_id"].isin(corpus["item_id"]).all()
    assert queries["query_id"].is_unique
    assert not set(retrieval_train["text_key"]) & cold_texts
    assert not set(retrieval_train["item_id"]) & cold_items

    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_train.to_parquet(output_dir / "retrieval_train.parquet", index=False)
    queries.to_parquet(output_dir / "eval_queries.parquet", index=False)
    truth.to_parquet(output_dir / "eval_truth.parquet", index=False)
    corpus.to_parquet(output_dir / "corpus.parquet", index=False)
    report = {
        "seed": seed,
        "queries": len(queries),
        "truth_pairs": len(truth),
        "retrieval_train_rows": len(retrieval_train),
        "benchmark_corpus_size": len(items),
        "validation_corpus_size": len(corpus),
        "extra_validation_items": len(extra_items),
        "seen_text_fraction": queries["seen_text"].mean(),
        "seen_context_fraction": queries["seen_context"].mean(),
        "seen_target_item_fraction": truth["seen_item"].mean(),
        "split_counts": queries["split"].value_counts().to_dict(),
        "regimes": queries.groupby(["split", "regime"]).size().to_dict(),
        "positives_per_query": truth.groupby("query_id").size().describe().to_dict(),
        "note": "Корпус расширен отложенными объявлениями. Более плотный корпус и неполная разметка ограничивают перенос локальной оценки на платформу.",
    }
    report["regimes"] = {
        "/".join(key): value for key, value in report["regimes"].items()
    }
    Path("reports/validation_split.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--queries", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    make_validation(args.data_dir, args.output_dir, args.queries, args.seed)
