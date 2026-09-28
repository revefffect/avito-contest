"""Одинаковая обработка текстов при обучении и поиске."""

import re
import unicodedata
from functools import lru_cache

from nltk.stem.snowball import RussianStemmer


TOKEN = re.compile(r"[a-zа-я0-9]+")
STEMMER = RussianStemmer()


def normalize(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return " ".join(TOKEN.findall(text))


@lru_cache(maxsize=200_000)
def stem(token):
    return STEMMER.stem(token)


def tokenize(text):
    # Числа и латиница нужны для марок, моделей и размеров техники.
    return [stem(token) for token in normalize(text).split()]


def query_keys(frame):
    columns = [
        "search_query",
        "search_location_id",
        "search_is_delivery_search",
        "search_infm_params_text",
        "search_category",
    ]
    values = frame[columns].fillna("").astype(str)
    values["search_query"] = values["search_query"].map(normalize)
    values["search_infm_params_text"] = values["search_infm_params_text"].map(normalize)
    return values.agg("\x1f".join, axis=1)
