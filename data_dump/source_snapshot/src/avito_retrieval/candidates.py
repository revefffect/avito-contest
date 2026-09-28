"""Собираем кандидатов из независимых источников и признаки для отбора 50."""

import time
import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from avito_retrieval.lexical import _topk
from avito_retrieval.text import normalize, tokenize


def top_indices(scores, count, positive_only=True):
    keep = np.isfinite(scores)
    if positive_only:
        keep &= scores > 0
    indices = np.flatnonzero(keep)
    return _topk(indices, scores[indices], count)[0].astype(np.int32)


class CandidateBuilder:
    def __init__(self, items, lexical, behavior, geography, item_embeddings=None):
        if list(items["item_id"]) != list(lexical.item_ids_):
            raise ValueError("Lexical index and corpus have different item order")
        if list(items["item_id"]) != list(behavior.item_ids):
            raise ValueError("Behavior index and corpus have different item order")
        self.items = items
        self.lexical = lexical
        bm25 = lexical.matrices_["bm25"]
        self.term_presence = sparse.csr_matrix(
            (np.ones_like(bm25.data), bm25.indices, bm25.indptr), shape=bm25.shape
        )
        self.behavior = behavior
        self.geography = geography
        self.item_embeddings = item_embeddings
        self.titles = items["item_title_raw"].fillna("").map(normalize).to_numpy()
        self.title_tokens = [set(tokenize(value)) for value in self.titles]
        self.params = (
            items["item_infm_params_text"]
            .fillna("")
            .map(lambda value: " " + " ".join(tokenize(value)) + " ")
            .to_numpy()
        )
        self.numeric = {}
        for name in ["item_price", "item_rating_reviews_count", "item_rating"]:
            values = (
                pd.to_numeric(items[name], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=np.float32)
            )
            if name != "item_rating":
                values = np.log1p(np.maximum(values, 0))
            self.numeric[name] = values
        for name in ["item_is_phone_hidden", "item_is_message_forbidden"]:
            self.numeric[name] = items[name].fillna(False).to_numpy(dtype=np.float32)
        self.numeric["title_length"] = np.array(
            [len(value) for value in self.titles], dtype=np.float32
        )
        self.numeric["description_length"] = (
            items["item_description_raw"]
            .fillna("")
            .str.len()
            .to_numpy(dtype=np.float32)
        )
        self.affinity = lru_cache(maxsize=64)(geography.affinity)

    def _lexical_scores(self, queries):
        scores = {}
        texts = queries["search_query"].fillna("").tolist()
        for name, vectorizer in self.lexical.vectorizers_.items():
            vectors = vectorizer.transform(texts)
            if name == "bm25":
                vectors.data.fill(1)
                coverage = (vectors @ self.term_presence).toarray()
                lengths = np.array([max(len(set(tokenize(text))), 1) for text in texts])
                scores["full_coverage"] = coverage / lengths[:, None]
            scores[name] = (vectors @ self.lexical.matrices_[name]).toarray()
        context = (
            queries["search_query"].fillna("")
            + " "
            + queries["search_infm_params_text"].fillna("")
        ).tolist()
        vectors = self.lexical.vectorizers_["word"].transform(context)
        scores["context_word"] = (vectors @ self.lexical.matrices_["word"]).toarray()
        return scores

    def _one_query(self, query, raw_scores, dense_scores, allowed_n_items):
        n_items = allowed_n_items or len(self.items)
        geo = self.affinity(query["search_location_id"])[:n_items]
        raw = {name: values[:n_items] for name, values in raw_scores.items()}
        info = self.behavior.query_info(query)
        memory = info["memory"][:n_items]
        channels = {}
        geo_limits = {"word": 220, "bm25": 500, "char": 350}
        for name in ["word", "bm25", "char"]:
            channels[name] = (raw[name], 100)
            channels[f"geo_{name}"] = (
                raw[name] * (0.015 + 0.985 * geo) ** 0.7,
                geo_limits[name],
            )
        channels["geo_context"] = (raw["context_word"] * geo**0.7, 220)
        channels["memory"] = (memory, 60)
        channels["geo_memory"] = (memory * geo, 120)
        if dense_scores is not None:
            for name, values in dense_scores.items():
                raw[name] = values[:n_items]
                channels[name] = (raw[name], 120)
                # У E5 близкие косинусы; поправка сохраняет этот масштаб.
                channels[f"geo_{name}"] = (
                    raw[name] + 0.055 * np.log(0.005 + 0.995 * geo),
                    300,
                )

        sources = {
            name: top_indices(values, count, positive_only="dense" not in name)
            for name, (values, count) in channels.items()
        }
        candidates = np.unique(np.concatenate(list(sources.values())))
        if len(candidates) < 50:
            fallback = top_indices(
                (1 + self.numeric["item_rating_reviews_count"][:n_items]) * geo, 50
            )
            candidates = np.union1d(candidates, fallback)
        result = {"item_idx": candidates.astype(np.int32)}
        for name, values in raw.items():
            result[name] = values[candidates]
            result[f"relative_{name}"] = values[candidates] / max(
                float(values.max()), 1e-6
            )
        for name, indices in sources.items():
            ranks = np.zeros(len(candidates), dtype=np.float32)
            ranks[np.searchsorted(candidates, indices)] = 1 / np.sqrt(
                1 + np.arange(len(indices))
            )
            result[f"rank_{name}"] = ranks
        result["geo_affinity"] = geo[candidates]
        result.update(self.geography.features(query["search_location_id"], candidates))
        result.update(self.behavior.features(query, candidates, info))
        result.update(
            {name: values[candidates] for name, values in self.numeric.items()}
        )
        query_tokens = set(tokenize(query["search_query"]))
        query_text = normalize(query["search_query"])
        filter_tokens = set(tokenize(query["search_infm_params_text"]))
        size = len(candidates)
        result["title_coverage"] = np.array(
            [
                len(query_tokens & self.title_tokens[i]) / max(len(query_tokens), 1)
                for i in candidates
            ]
        )
        result["title_precision"] = np.array(
            [
                len(query_tokens & self.title_tokens[i])
                / max(len(self.title_tokens[i]), 1)
                for i in candidates
            ]
        )
        result["title_exact_phrase"] = np.array(
            [bool(query_text) and query_text in self.titles[i] for i in candidates]
        )
        result["filter_coverage"] = np.array(
            [
                sum(f" {token} " in self.params[i] for token in filter_tokens)
                / max(len(filter_tokens), 1)
                for i in candidates
            ]
        )
        result["query_words"] = np.full(size, len(query_tokens))
        result["query_length"] = np.full(size, len(query_text))
        result["has_filters"] = np.full(size, bool(filter_tokens))
        result["delivery_search"] = np.full(size, query["search_is_delivery_search"])
        result["category_match"] = (
            self.items["item_category_id"].to_numpy()[candidates]
            == query["search_category"]
        )
        result["lexical_heuristic"] = (
            result["relative_word"]
            + 0.7 * result["relative_bm25"]
            + 0.5 * result["relative_char"]
        ) * result["geo_affinity"] ** 0.7
        if dense_scores is not None:
            result["hybrid_heuristic"] = (
                result["dense"]
                + 0.055 * np.log(0.005 + 0.995 * result["geo_affinity"])
                + 0.035 * result["lexical_heuristic"]
            )
        return {
            name: values.astype(np.float32) if name != "item_idx" else values
            for name, values in result.items()
        }

    def write_features(
        self,
        queries,
        output_path,
        truth=None,
        query_embeddings=None,
        allowed_n_items=None,
        batch_size=16,
    ):
        ids = self.items["item_id"].to_numpy()
        targets = (
            {}
            if truth is None
            else truth.groupby("query_id")["item_id"].agg(set).to_dict()
        )
        writer = None
        output_path = Path(output_path)
        temporary_path = output_path.with_suffix(output_path.suffix + ".part")
        start_time = time.monotonic()
        total_rows = 0
        try:
            for start in range(0, len(queries), batch_size):
                batch = queries.iloc[start : start + batch_size]
                lexical_scores = self._lexical_scores(batch)
                dense_scores = None
                if query_embeddings is not None:
                    dense_scores = {
                        name: values[start : start + len(batch)]
                        @ self.item_embeddings.T
                        for name, values in query_embeddings.items()
                    }
                frames = []
                for offset, (_, query) in enumerate(batch.iterrows()):
                    one_dense = (
                        None
                        if dense_scores is None
                        else {
                            name: values[offset]
                            for name, values in dense_scores.items()
                        }
                    )
                    result = self._one_query(
                        query,
                        {key: value[offset] for key, value in lexical_scores.items()},
                        one_dense,
                        allowed_n_items,
                    )
                    frame = pd.DataFrame(result)
                    frame.insert(0, "qid", np.int32(start + offset))
                    correct = targets.get(query["query_id"], set())
                    frame["label"] = np.array(
                        [
                            value in correct
                            for value in ids[frame["item_idx"].to_numpy()]
                        ],
                        dtype=np.int8,
                    )
                    frames.append(frame)
                table = pa.Table.from_pandas(
                    pd.concat(frames, ignore_index=True), preserve_index=False
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_path, table.schema, compression="zstd"
                    )
                writer.write_table(table)
                total_rows += table.num_rows
                if start == 0 or (start // batch_size + 1) % 10 == 0:
                    elapsed = time.monotonic() - start_time
                    done = start + len(batch)
                    print(
                        f"{done}/{len(queries)} запросов, {total_rows:,} кандидатов, {elapsed:.1f}с",
                        flush=True,
                    )
        finally:
            if writer is not None:
                writer.close()
        os.replace(temporary_path, output_path)
