"""Статистики выбора и перенос знаний с похожих обучающих запросов."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as normalize_matrix

from avito_retrieval.text import normalize, query_keys, tokenize


class BehaviorIndex:
    def fit(self, train, items):
        self.item_ids = items["item_id"].to_numpy()
        self.item_categories = items["item_microcat_id"].to_numpy()
        self.category_ids = np.sort(train["item_microcat_id"].unique())
        self.category_lookup = {value: i for i, value in enumerate(self.category_ids)}
        self.item_category_indices = np.array(
            [self.category_lookup.get(value, -1) for value in self.item_categories]
        )
        train = train.copy()
        train["text_key"] = train["search_query"].map(normalize)
        train["context_key"] = query_keys(train)
        self.texts = np.sort(train["text_key"].unique())
        text_lookup = {value: i for i, value in enumerate(self.texts)}
        item_lookup = {value: i for i, value in enumerate(self.item_ids)}
        text_rows = train["text_key"].map(text_lookup).to_numpy()
        categories = train["item_microcat_id"].map(self.category_lookup).to_numpy()
        counts = sparse.coo_matrix(
            (np.ones(len(train), dtype=np.float32), (text_rows, categories)),
            shape=(len(self.texts), len(self.category_ids)),
        ).tocsr()
        self.category_matrix = normalize_matrix(counts, norm="l1")
        same_location = train["search_location_id"].eq(train["item_location_id"])
        self.global_local_rate = float(same_location.mean())
        location_stats = (
            train.assign(same_location=same_location)
            .groupby("item_microcat_id")["same_location"]
            .agg(["sum", "count"])
        )
        local_rate = (location_stats["sum"] + 20 * same_location.mean()) / (
            location_stats["count"] + 20
        )
        self.category_local_rate = local_rate.reindex(self.category_ids).to_numpy(
            dtype=np.float32
        )

        self.word_vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=1,
            max_features=180_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=150_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(self.texts).T.tocsr()
        self.char_matrix = self.char_vectorizer.fit_transform(self.texts).T.tocsr()

        in_corpus = train["item_id"].isin(item_lookup)
        matched = train.loc[in_corpus]
        rows = matched["text_key"].map(text_lookup).to_numpy()
        cols = matched["item_id"].map(item_lookup).to_numpy()
        self.query_items = sparse.coo_matrix(
            (np.ones(len(matched), dtype=np.float32), (rows, cols)),
            shape=(len(self.texts), len(items)),
        ).tocsr()
        self.query_items.data = np.log1p(self.query_items.data)
        self.popularity = (
            items["item_id"]
            .map(train["item_id"].value_counts())
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        self.text_item_counts = (
            matched.groupby(["text_key", "item_id"]).size().to_dict()
        )
        self.context_item_counts = (
            matched.groupby(["context_key", "item_id"]).size().to_dict()
        )
        self.text_count = train["text_key"].value_counts().to_dict()

        filter_counts = (
            train.loc[train["search_infm_params_text"].ne("")]
            .groupby(["search_infm_params_text", "item_microcat_id"])
            .size()
        )
        filter_totals = filter_counts.groupby(level=0).sum()
        self.filter_probs = (filter_counts / filter_totals).to_dict()
        self.filter_counts = filter_totals.to_dict()
        return self

    def query_info(self, query, neighbors=40):
        text = normalize(query["search_query"])
        word = self.word_vectorizer.transform([text]) @ self.word_matrix
        char = self.char_vectorizer.transform([text]) @ self.char_matrix
        similarities = (word * 0.55 + char * 0.45).toarray().ravel()
        k = min(neighbors, len(similarities))
        indices = np.argpartition(similarities, -k)[-k:]
        indices = indices[np.lexsort((indices, -similarities[indices]))]
        weights = np.maximum(similarities[indices], 0) ** 4
        total = weights.sum()
        if total:
            weights /= total
        category_probs = np.asarray(weights @ self.category_matrix[indices]).ravel()
        memory = sparse.csr_matrix(weights.reshape(1, -1)) @ self.query_items[indices]
        return {
            "text": text,
            "category_probs": category_probs,
            "memory": memory.toarray().ravel(),
            "nearest_similarity": (
                float(similarities[indices[0]]) if len(indices) else 0.0
            ),
            "text_count": self.text_count.get(text, 0),
            "query_local_rate": float(category_probs @ self.category_local_rate),
        }

    def features(self, query, candidates, info):
        ids = self.item_ids[candidates]
        cats = self.item_categories[candidates]
        category_indices = self.item_category_indices[candidates]
        category_probs = np.zeros(len(candidates), dtype=np.float32)
        known = category_indices >= 0
        category_probs[known] = info["category_probs"][category_indices[known]]
        local_rate = np.full(len(candidates), self.global_local_rate, dtype=np.float32)
        local_rate[known] = self.category_local_rate[category_indices[known]]
        context = query_keys(pd.DataFrame([query])).iloc[0]
        text = info["text"]
        filters = query["search_infm_params_text"]
        return {
            "popularity": np.log1p(self.popularity[candidates]),
            "memory_score": info["memory"][candidates],
            "text_item_count": np.log1p(
                [self.text_item_counts.get((text, value), 0) for value in ids]
            ),
            "context_item_count": np.log1p(
                [self.context_item_counts.get((context, value), 0) for value in ids]
            ),
            "microcat_probability": category_probs,
            "category_local_rate": local_rate,
            "query_local_rate": np.full(len(candidates), info["query_local_rate"]),
            "filter_microcat_probability": np.array(
                [self.filter_probs.get((filters, cat), 0) for cat in cats]
            ),
            "filter_train_count": np.full(
                len(candidates), np.log1p(self.filter_counts.get(filters, 0))
            ),
            "query_train_count": np.full(len(candidates), np.log1p(info["text_count"])),
            "nearest_query_similarity": np.full(
                len(candidates), info["nearest_similarity"]
            ),
        }
