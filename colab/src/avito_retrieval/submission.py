"""Проверка и сохранение ответа без потери регистра и ведущих нулей."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from numbers import Integral
from pathlib import Path

import pandas as pd


ITEM_ID_PATTERN = re.compile(r"[0-9a-f]{16}")


def _query_ids(frame: pd.DataFrame, source: str) -> list[str]:
    if list(frame.columns).count("query_id") != 1:
        raise ValueError(f"{source}: expected one query_id column")
    values = frame["query_id"].tolist()
    for row, value in enumerate(values):
        if not isinstance(value, str):
            raise ValueError(f"{source}, row {row}: query_id must be a string")
        if len(value) != 16 or any(char.isspace() for char in value):
            raise ValueError(
                f"{source}, row {row}: query_id must contain exactly 16 non-space characters"
            )
    if len(values) != len(set(values)):
        raise ValueError(f"{source}: duplicate query_id")
    return values


def _item_ids(items: pd.DataFrame) -> set[str]:
    if list(items.columns).count("item_id") != 1:
        raise ValueError("items: expected one item_id column")
    values = items["item_id"].tolist()
    for row, value in enumerate(values):
        if not isinstance(value, str) or ITEM_ID_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"items, row {row}: item_id must be a string of 16 lowercase hex characters"
            )
    if len(values) != len(set(values)):
        raise ValueError("items: duplicate item_id")
    return set(values)


def validate_submission(
    answer: pd.DataFrame,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    max_candidates: int = 50,
) -> dict:
    """Проверить формат и покрытие запросов, вернуть статистику кандидатов.

    Пустая строка answer разрешена. Пропуски, приведение чисел к строкам и
    исправление регистра намеренно не поддерживаются: потерянный ID не восстановить.
    Все ошибки входных данных приводят к ValueError.
    """
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, Integral):
        raise ValueError("max_candidates must be a non-negative integer")
    if max_candidates < 0:
        raise ValueError("max_candidates must be a non-negative integer")
    if len(answer.columns) != 2 or set(answer.columns) != {"query_id", "answer"}:
        raise ValueError("answer: expected exactly the columns query_id and answer")

    expected = set(_query_ids(queries, "queries"))
    actual = set(_query_ids(answer, "answer"))
    if expected != actual:
        missing, extra = expected - actual, actual - expected
        raise ValueError(
            f"query coverage mismatch: {len(missing)} missing, {len(extra)} extra, "
            f"missing examples={sorted(missing)[:3]}, extra examples={sorted(extra)[:3]}"
        )
    corpus = _item_ids(items)
    counts = []
    for row, value in enumerate(answer["answer"].tolist()):
        if not isinstance(value, str):
            raise ValueError(
                f"answer, row {row}: answer must be a string, not a missing value"
            )

        # split(" ") сохраняет лишние пробелы как пустые токены, чтобы их заметить.
        candidates = value.split(" ") if value else []
        if len(candidates) > max_candidates:
            raise ValueError(
                f"answer, row {row}: more than {max_candidates} candidates"
            )
        for item_id in candidates:
            if ITEM_ID_PATTERN.fullmatch(item_id) is None:
                raise ValueError(
                    f"answer, row {row}: invalid item_id {item_id!r}, "
                    "use lowercase 16-character hex IDs separated by one space"
                )
        if len(candidates) != len(set(candidates)):
            raise ValueError(f"answer, row {row}: duplicate item_id")
        unknown = set(candidates) - corpus
        if unknown:
            raise ValueError(
                f"answer, row {row}: unknown item_id {sorted(unknown)[:3]}"
            )
        counts.append(len(candidates))

    return {
        "rows": len(counts),
        "empty_answers": counts.count(0),
        "candidates_total": sum(counts),
        "candidates_min": min(counts, default=0),
        "candidates_max": max(counts, default=0),
        "candidates_mean": sum(counts) / len(counts) if counts else 0.0,
        "max_candidates": int(max_candidates),
    }


def write_submission(
    query_ids: Iterable[str],
    predictions: Iterable[Sequence[str]],
    output_path: str | Path,
    items: pd.DataFrame,
) -> dict:
    """Сохранить проверенный UTF-8 CSV, порядок кандидатов оставить прежним.

    Полное покрытие здесь проверяется относительно переданного query_ids.
    Для независимой проверки относительно бенчмарка используйте CLI.
    """
    ids = list(query_ids)
    prediction_rows = list(predictions)
    if len(ids) != len(prediction_rows):
        raise ValueError("query_ids and predictions must have the same number of rows")
    joined = []
    for row, candidates in enumerate(prediction_rows):
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Iterable):
            raise ValueError(
                f"predictions, row {row}: expected a sequence of item_id strings"
            )
        candidate_ids = list(candidates)
        if any(not isinstance(item_id, str) for item_id in candidate_ids):
            raise ValueError(f"predictions, row {row}: item_id must be a string")
        if any(ITEM_ID_PATTERN.fullmatch(item_id) is None for item_id in candidate_ids):
            raise ValueError(f"predictions, row {row}: invalid item_id")
        joined.append(" ".join(candidate_ids))

    answer = pd.DataFrame({"query_id": ids, "answer": joined})
    queries = pd.DataFrame({"query_id": ids})
    report = validate_submission(answer, queries, items)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # При сбое записи уже готовый ответ останется целым.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            answer.to_csv(handle, index=False)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {**report, "output_path": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an Avito contest submission")
    parser.add_argument("--answer", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    args = parser.parse_args()
    try:
        # Отключаем угадывание типов и NA: пустой ответ допустим, ID всегда строка.
        answer = pd.read_csv(
            args.answer, dtype=str, keep_default_na=False, encoding="utf-8"
        )
        # Одного потока достаточно для ID, дочерний CLI не создаёт лишний Arrow pool.
        queries = pd.read_parquet(args.queries, columns=["query_id"], use_threads=False)
        items = pd.read_parquet(args.items, columns=["item_id"], use_threads=False)
        report = validate_submission(answer, queries, items)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
