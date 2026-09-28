"""Три независимых текстовых индекса и поиск без плотной матрицы всего корпуса."""

from __future__ import annotations

import hashlib
import logging
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from avito_retrieval.text import normalize, tokenize


LOGGER = logging.getLogger(__name__)
TEXT_COLUMNS = ("item_title_raw", "item_infm_params_text", "item_description_raw")


def _text_column(frame, name):
    if name not in frame:
        return [""] * len(frame)
    return [value if isinstance(value, str) else "" for value in frame[name]]


def _locations(frame, name):
    if name not in frame:
        return None
    result = []
    for value in frame[name]:
        if pd.isna(value):
            result.append("")
        elif isinstance(value, (float, np.floating)) and value.is_integer():
            result.append(str(int(value)))
        else:
            result.append(str(value))
    return np.asarray(result)


def _topk(indices, scores, k):
    """Разрешать ничьи по позиции объявления, включая границу top-k."""
    if len(scores) > k:
        threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        tied = tied[np.argsort(indices[tied], kind="stable")[: k - len(above)]]
        selected = np.concatenate((above, tied))
        indices, scores = indices[selected], scores[selected]
    order = np.lexsort((indices, -scores))
    return indices[order], scores[order]


class LexicalIndex:
    """Словный TF-IDF, BM25 описаний и символьный TF-IDF заголовков.

    Матрицы хранятся как CSR float32: признаки × объявления. Поиск возвращает
    позиции строк исходного корпуса; -1 и score=0 обозначают отсутствие кандидата.
    """

    def __init__(
        self, batch_size=16, char_max_features=250_000, char_min_df=2, k1=1.2, b=0.65
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.batch_size = batch_size
        self.char_max_features = char_max_features
        self.char_min_df = char_min_df
        self.k1, self.b = k1, b

    def fit(self, items: pd.DataFrame) -> "LexicalIndex":
        if "item_id" not in items or items.empty:
            raise ValueError("items must contain a nonempty item_id column")
        ids = items["item_id"].tolist()
        if any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(
            ids
        ):
            raise ValueError("item_id must contain unique strings")
        self.item_ids_ = np.asarray(ids)
        self.item_locations_ = _locations(items, "item_location_id")
        titles, params, descriptions = (
            _text_column(items, name) for name in TEXT_COLUMNS
        )
        descriptions = [value[:3000] for value in descriptions]
        short_texts = [f"{title} {param}" for title, param in zip(titles, params)]

        fingerprint = hashlib.sha256()
        locations = (
            self.item_locations_
            if self.item_locations_ is not None
            else [""] * len(ids)
        )
        for row in zip(ids, titles, params, descriptions, locations):
            for value in row:
                encoded = value.encode("utf-8")
                fingerprint.update(len(encoded).to_bytes(8, "little"))
                fingerprint.update(encoded)

        self.vectorizers_, self.matrices_ = {}, {}
        word_options = dict(
            tokenizer=tokenize, token_pattern=None, lowercase=False, dtype=np.float32
        )
        LOGGER.info("Building word TF-IDF for %d items", len(items))
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, **word_options
        )
        self.matrices_["word"] = vectorizer.fit_transform(short_texts).T.tocsr()
        self.vectorizers_["word"] = vectorizer

        LOGGER.info("Building BM25")
        # Повторы усиливают заголовок и параметры относительно рекламного описания.
        full_texts = (
            f"{title} {title} {title} {param} {param} {description}"
            for title, param, description in zip(titles, params, descriptions)
        )
        vectorizer = CountVectorizer(**word_options)
        counts = vectorizer.fit_transform(full_texts)
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = max(float(lengths.mean()), 1.0)
        df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float32)
        idf = np.log1p((len(items) - df + 0.5) / (df + 0.5)).astype(np.float32)
        norms = self.k1 * (1 - self.b + self.b * lengths / average_length)
        # Положительный IDF и насыщение TF по формуле Lucene BM25:
        # https://www.elastic.co/blog/found-similarity-in-elasticsearch
        for start in range(0, len(items), 4096):
            end = min(start + 4096, len(items))
            left, right = counts.indptr[start], counts.indptr[end]
            values = counts.data[left:right]
            row_norms = np.repeat(
                norms[start:end], np.diff(counts.indptr[start : end + 1])
            )
            values *= (self.k1 + 1) / (values + row_norms)
            values *= idf[counts.indices[left:right]]
        self.matrices_["bm25"] = counts.T.tocsr()
        self.vectorizers_["bm25"] = vectorizer
        del counts, descriptions

        LOGGER.info("Building character TF-IDF")
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            preprocessor=normalize,
            lowercase=False,
            max_features=self.char_max_features,
            min_df=self.char_min_df,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.matrices_["char"] = vectorizer.fit_transform(short_texts).T.tocsr()
        self.vectorizers_["char"] = vectorizer
        matrix_bytes = sum(
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
            for matrix in self.matrices_.values()
        )
        self.metadata_ = {
            "schema_version": 1,
            "n_items": len(items),
            "corpus_sha256": fingerprint.hexdigest(),
            "fingerprint_fields": ["item_id", *TEXT_COLUMNS, "item_location_id"],
            "description_char_limit": 3000,
            "k1": self.k1,
            "b": self.b,
            "char_min_df": self.char_min_df,
            "char_max_features": self.char_max_features,
            "matrix_bytes": matrix_bytes,
            "channels": {
                name: {"features": matrix.shape[0], "nonzero": matrix.nnz}
                for name, matrix in self.matrices_.items()
            },
        }
        LOGGER.info(
            "Index ready; sparse matrices occupy %.2f GiB", matrix_bytes / 2**30
        )
        return self

    def search(
        self, queries: pd.DataFrame, top_k=300, allowed_n_items=None, include_local=True
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Искать глобально и внутри локации, не обрезая локальный поиск global top-k.

        allowed_n_items ограничивает поиск префиксом корпуса. При локальной
        проверке префикс позволяет исключить добавленные обучающие объявления.
        """
        if not hasattr(self, "matrices_"):
            raise ValueError("fit the index before search")
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        n_items = len(self.item_ids_)
        if allowed_n_items is not None:
            if (
                not isinstance(allowed_n_items, Integral)
                or not 0 <= allowed_n_items <= n_items
            ):
                raise ValueError("allowed_n_items must be a valid corpus prefix length")
            n_items = int(allowed_n_items)
        query_texts = [
            f"{text} {param}"
            for text, param in zip(
                _text_column(queries, "search_query"),
                _text_column(queries, "search_infm_params_text"),
            )
        ]
        query_locations = _locations(queries, "search_location_id")
        local = (
            include_local
            and self.item_locations_ is not None
            and query_locations is not None
        )
        results = {}
        for name, matrix in self.matrices_.items():
            channel_names = [name, f"local_{name}"] if local else [name]
            for channel_name in channel_names:
                results[channel_name] = (
                    np.full((len(queries), top_k), -1, dtype=np.int32),
                    np.zeros((len(queries), top_k), dtype=np.float32),
                )
            for start in range(0, len(queries), self.batch_size):
                text_batch = query_texts[start : start + self.batch_size]
                query_matrix = self.vectorizers_[name].transform(text_batch)
                if name == "bm25":
                    query_matrix.data.fill(1)
                # В памяти только ненулевые совпадения небольшой пачки запросов.
                scores = (query_matrix @ matrix).tocsr()
                for offset in range(len(text_batch)):
                    row = start + offset
                    left, right = scores.indptr[offset : offset + 2]
                    indices, values = (
                        scores.indices[left:right],
                        scores.data[left:right],
                    )
                    keep = (indices < n_items) & (values > 0)
                    indices, values = indices[keep], values[keep]
                    best_indices, best_scores = _topk(indices, values, top_k)
                    size = len(best_indices)
                    results[name][0][row, :size] = best_indices
                    results[name][1][row, :size] = best_scores
                    if local and query_locations[row]:
                        keep = self.item_locations_[indices] == query_locations[row]
                        best_indices, best_scores = _topk(
                            indices[keep], values[keep], top_k
                        )
                        size = len(best_indices)
                        results[f"local_{name}"][0][row, :size] = best_indices
                        results[f"local_{name}"][1][row, :size] = best_scores
        return results
