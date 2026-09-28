import numpy as np
import pandas as pd
import pytest

from avito_retrieval.behavior import BehaviorIndex
from avito_retrieval.candidates import CandidateBuilder
from avito_retrieval.geography import Geography
from avito_retrieval.lexical import LexicalIndex


@pytest.fixture
def tiny_builder():
    items = pd.DataFrame(
        {
            "item_id": [f"{number:016x}" for number in range(6)],
            "item_title_raw": [
                "Ремонт сантехники",
                "Сантехник установка крана",
                "Торт на заказ",
                "Ремонт квартиры",
                "Помощь по дому",
                "Устранение редкого засора",
            ],
            "item_description_raw": [
                "Ремонт сантехники и труб",
                "Установка крана и ремонт сантехники",
                "Пеку торты на заказ",
                "Ремонт стен и полов",
                "Замена сифона и устранение засора",
                "Редкий засор устраняем быстро",
            ],
            "item_infm_params_text": [
                "Вид услуги Ремонт и отделка",
                "Вид услуги Ремонт и отделка",
                "Вид услуги Доставка еды",
                "Вид услуги Ремонт и отделка",
                "Вид услуги Ремонт и отделка",
                "Вид услуги Ремонт и отделка",
            ],
            "item_location_id": [10] * 6,
            "item_microcat_id": [1, 1, 2, 1, 1, 1],
            "item_category_id": [114] * 6,
            "item_latitude": [55.75] * 6,
            "item_longitude": [37.62] * 6,
            "item_price": [1000, 1500, 2000, 3000, 500, 5000],
            "item_rating_reviews_count": [1, 2, 3, 4, 5, 1000],
            "item_rating": [4.8] * 6,
            "item_is_phone_hidden": [False] * 6,
            "item_is_message_forbidden": [False] * 6,
        }
    )
    train = pd.DataFrame(
        {
            "search_query": [
                "сантехник",
                "ремонт сантехники",
                "торт",
                "торт на заказ",
                "редкий засор",
            ],
            "search_location_id": [10] * 5,
            "search_is_delivery_search": [0] * 5,
            "search_infm_params_text": [""] * 5,
            "search_category": [114] * 5,
            "item_id": items.loc[[0, 1, 2, 2, 5], "item_id"].tolist(),
            "item_location_id": [10] * 5,
            "item_microcat_id": [1, 1, 2, 2, 1],
        }
    )
    embeddings = np.array(
        [
            [-0.2, -0.7],
            [-0.4, -0.1],
            [-0.5, -0.2],
            [-0.6, -0.3],
            [-0.1, -0.4],
            [0.99, 0.95],
        ],
        dtype=np.float32,
    )
    return CandidateBuilder(
        items,
        LexicalIndex(char_min_df=1).fit(items),
        BehaviorIndex().fit(train, items),
        Geography().fit(train, items),
        item_embeddings=embeddings,
    )


def queries(texts):
    return pd.DataFrame(
        {
            "query_id": [f"query_{number:010d}" for number in range(len(texts))],
            "search_query": texts,
            "search_location_id": [10] * len(texts),
            "search_is_delivery_search": [0] * len(texts),
            "search_infm_params_text": [""] * len(texts),
            "search_category": [114] * len(texts),
        }
    )


def test_corpus_prefix_excludes_strong_tail_from_every_candidate_source(
    tiny_builder, tmp_path
):
    query = queries(["редкий засор"])
    vectors = {"dense": np.array([[1, 0]], dtype=np.float32)}
    full_path, prefix_path = tmp_path / "full.parquet", tmp_path / "prefix.parquet"
    tiny_builder.write_features(query, full_path, query_embeddings=vectors)
    tiny_builder.write_features(
        query, prefix_path, query_embeddings=vectors, allowed_n_items=5
    )
    full = pd.read_parquet(full_path).set_index("item_idx")
    prefix = pd.read_parquet(prefix_path)
    # Хвост побеждает по смысловому каналу, памяти и запасному порядку по отзывам.
    assert full.loc[5, "rank_dense"] == 1
    assert full.loc[5, "memory_score"] > 0
    assert set(prefix["item_idx"]) == set(range(5))
    assert 5 not in prefix["item_idx"].values


def test_negative_dense_channels_keep_their_own_ranks_and_description_matches(
    tiny_builder, tmp_path
):
    query = queries(["сифон"])
    vectors = {
        "dense": np.array([[1, 0]], dtype=np.float32),
        "dense_plain": np.array([[0, 1]], dtype=np.float32),
    }
    output = tmp_path / "negative.parquet"
    tiny_builder.write_features(
        query, output, query_embeddings=vectors, allowed_n_items=5
    )
    frame = pd.read_parquet(output).set_index("item_idx")
    assert (frame["dense"] < 0).all()
    assert (frame["rank_dense"] > 0).all()
    assert (frame["rank_dense_plain"] > 0).all()
    assert frame["rank_dense"].idxmax() == 4
    assert frame["rank_dense_plain"].idxmax() == 1
    assert frame.loc[4, "title_coverage"] == 0
    assert frame.loc[4, "full_coverage"] == 1
    assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()


def test_labels_follow_query_ids_without_affecting_features_and_export_is_repeatable(
    tiny_builder, tmp_path
):
    query = queries(["сантехник", "сантехник"])
    query.index = [71, 3]
    truth = pd.DataFrame(
        {
            "query_id": [query.iloc[1]["query_id"], query.iloc[0]["query_id"]],
            "item_id": tiny_builder.items.loc[[4, 1], "item_id"].tolist(),
        }
    )
    first, second, unlabelled = [
        tmp_path / name for name in ["a.parquet", "b.parquet", "c.parquet"]
    ]
    tiny_builder.write_features(query, first, truth=truth, batch_size=1)
    tiny_builder.write_features(query, second, truth=truth, batch_size=1)
    tiny_builder.write_features(query, unlabelled, batch_size=1)
    labelled = pd.read_parquet(first)
    positive_pairs = set(
        map(
            tuple, labelled.loc[labelled["label"].eq(1), ["qid", "item_idx"]].to_numpy()
        )
    )
    assert positive_pairs == {(0, 1), (1, 4)}
    no_truth = pd.read_parquet(unlabelled)
    assert not no_truth["label"].any()
    pd.testing.assert_frame_equal(
        labelled.drop(columns="label"), no_truth.drop(columns="label")
    )
    assert first.read_bytes() == second.read_bytes()


def test_interrupted_export_leaves_only_partial_file_and_preserves_completed_output(
    tiny_builder, tmp_path, monkeypatch
):
    output = tmp_path / "features.parquet"
    previous = b"previous completed result"
    output.write_bytes(previous)
    original = tiny_builder._one_query
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted feature generation")
        return original(*args, **kwargs)

    monkeypatch.setattr(tiny_builder, "_one_query", interrupted)
    with pytest.raises(RuntimeError, match="interrupted feature generation"):
        tiny_builder.write_features(
            queries(["сантехник", "торт"]), output, batch_size=1
        )
    assert output.read_bytes() == previous
    partial = output.with_suffix(".parquet.part")
    assert partial.is_file()
    assert pd.read_parquet(partial)["qid"].unique().tolist() == [0]
