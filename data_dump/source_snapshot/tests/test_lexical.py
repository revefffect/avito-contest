import joblib
import numpy as np
import pandas as pd

from avito_retrieval.lexical import LexicalIndex, _topk


def corpus():
    return pd.DataFrame(
        {
            "item_id": [f"{value:016x}" for value in range(5)],
            "item_title_raw": [
                "Ремонт телевизоров",
                "Ремонт телевизоров",
                "Ремонт квартир",
                "Монтаж домофонов",
                "Монтаж домофонов",
            ],
            "item_infm_params_text": [
                "Техника",
                "Техника",
                "Отделка",
                "Техника",
                "Техника",
            ],
            "item_description_raw": [
                "Экран и электроника",
                "Экран и электроника",
                "Стены и пол",
                "Установка видеодомофонов",
                "Установка видеодомофонов",
            ],
            "item_location_id": [1, 2, 1, 1, 2],
        }
    )


def test_retrieval_local_filter_prefix_and_serialization(tmp_path):
    index = LexicalIndex(batch_size=1).fit(corpus())
    queries = pd.DataFrame(
        {
            "search_query": ["ремонт телевизора", "qwertyasdfzxcv"],
            "search_location_id": [2, 1],
        }
    )
    found = index.search(queries, top_k=8)
    assert set(found) == {
        "word",
        "bm25",
        "char",
        "local_word",
        "local_bm25",
        "local_char",
    }
    for name, (indices, scores) in found.items():
        assert indices.shape == scores.shape == (2, 8)
        assert indices.dtype == np.int32 and scores.dtype == np.float32
        assert indices[0, 0] == (1 if name.startswith("local_") else 0)
        assert np.all(indices[1] == -1)
        assert np.all(scores[1] == 0)
    restricted = index.search(queries.iloc[:1], top_k=8, allowed_n_items=1)
    for name, (indices, _) in restricted.items():
        assert np.all(indices < 1)
        if name.startswith("local_"):
            assert np.all(indices == -1)
    path = tmp_path / "index.joblib"
    joblib.dump(index, path)
    restored = joblib.load(path, mmap_mode="r").search(queries, top_k=8)
    for name in found:
        np.testing.assert_array_equal(found[name][0], restored[name][0])
        np.testing.assert_allclose(found[name][1], restored[name][1])


def test_bm25_matches_hand_computed_document_score():
    items = corpus()
    items["item_title_raw"] = ["ремонт", "ремонт ремонт", "кот", "кот", "кот"]
    items["item_infm_params_text"] = ""
    items["item_description_raw"] = ""
    index = LexicalIndex().fit(items)
    query = pd.DataFrame({"search_query": ["ремонт"]})
    indices, scores = index.search(query, top_k=5)["bm25"]
    lengths = np.array([3, 6, 3, 3, 3], dtype=np.float32)
    idf = np.log1p((5 - 2 + 0.5) / (2 + 0.5))
    expected = (
        idf
        * (2.2 * lengths[:2])
        / (lengths[:2] + 1.2 * (0.35 + 0.65 * lengths[:2] / lengths.mean()))
    )
    actual = dict(zip(indices[0], scores[0]))
    np.testing.assert_allclose([actual[0], actual[1]], expected, rtol=1e-6)


def test_topk_uses_document_position_to_break_boundary_ties():
    indices = np.array([8, 4, 3, 9, 2], dtype=np.int32)
    scores = np.array([1, 2, 1, 1, 1], dtype=np.float32)
    chosen, values = _topk(indices, scores, 3)
    np.testing.assert_array_equal(chosen, [4, 2, 3])
    np.testing.assert_array_equal(values, [2, 1, 1])
