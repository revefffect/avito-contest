import subprocess
import sys

import pandas as pd
import pytest

from avito_retrieval.submission import validate_submission, write_submission


QUERY_A = "00WuFMaXSFZBxSzT"
QUERY_B = "0000000000000001"
ITEM_A = "0000000000000001"
ITEM_B = "abcdef0123456789"


@pytest.fixture
def frames():
    queries = pd.DataFrame({"query_id": [QUERY_A, QUERY_B]})
    items = pd.DataFrame({"item_id": [ITEM_A, ITEM_B]})
    answer = pd.DataFrame({"query_id": [QUERY_B, QUERY_A], "answer": [ITEM_A, ITEM_B]})
    return answer, queries, items


def test_valid_ids_are_case_sensitive_and_keep_leading_zeros(frames, tmp_path):
    answer, queries, items = frames
    assert validate_submission(answer, queries, items)["rows"] == 2
    path = tmp_path / "answer.csv"
    report = write_submission(queries["query_id"], [[ITEM_A, ITEM_B], []], path, items)
    restored = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(restored.columns) == ["query_id", "answer"]
    assert restored["query_id"].tolist() == [QUERY_A, QUERY_B]
    assert restored["answer"].tolist() == [f"{ITEM_A} {ITEM_B}", ""]
    assert report["empty_answers"] == 1
    assert validate_submission(restored, queries, items)["candidates_total"] == 2


@pytest.mark.parametrize("bad_id", [1, "1", QUERY_A.lower(), " " + QUERY_A[1:], None])
def test_invalid_or_changed_query_id_fails(frames, bad_id):
    answer, queries, items = frames
    answer.loc[1, "query_id"] = bad_id
    with pytest.raises(ValueError):
        validate_submission(answer, queries, items)


def test_missing_query_fails(frames):
    answer, queries, items = frames
    with pytest.raises(ValueError, match="1 missing"):
        validate_submission(answer.iloc[:1], queries, items)


def test_duplicate_query_fails(frames):
    answer, queries, items = frames
    answer.loc[1, "query_id"] = answer.loc[0, "query_id"]
    with pytest.raises(ValueError, match="duplicate query_id"):
        validate_submission(answer, queries, items)


@pytest.mark.parametrize(
    "bad_answer, reason",
    [
        (f"{ITEM_A} {ITEM_A}", "duplicate item_id"),
        ("ffffffffffffffff", "unknown item_id"),
        (ITEM_B.upper(), "invalid item_id"),
        (f" {ITEM_A}", "invalid item_id"),
        (f"{ITEM_A} ", "invalid item_id"),
        (f"{ITEM_A}  {ITEM_B}", "invalid item_id"),
        (f"{ITEM_A}\t{ITEM_B}", "invalid item_id"),
        (f"{ITEM_A},{ITEM_B}", "invalid item_id"),
        (str([ITEM_A]), "invalid item_id"),
        (None, "must be a string"),
        (float("nan"), "must be a string"),
        (1, "must be a string"),
    ],
)
def test_invalid_candidate_representations_fail(frames, bad_answer, reason):
    answer, queries, items = frames
    answer.loc[0, "answer"] = bad_answer
    with pytest.raises(ValueError, match=reason):
        validate_submission(answer, queries, items)


def test_candidate_limit_accepts_50_and_rejects_51(frames):
    answer, queries, _ = frames
    ids = [f"{number:016x}" for number in range(51)]
    items = pd.DataFrame({"item_id": ids + [ITEM_B]})
    answer.loc[0, "answer"] = " ".join(ids[:50])
    assert validate_submission(answer, queries, items)["candidates_max"] == 50
    answer.loc[0, "answer"] = " ".join(ids)
    with pytest.raises(ValueError, match="more than 50"):
        validate_submission(answer, queries, items)


def test_empty_answers_are_valid_and_reported(frames):
    answer, queries, items = frames
    answer["answer"] = ""
    report = validate_submission(answer, queries, items)
    assert report["rows"] == 2
    assert report["empty_answers"] == 2
    assert report["candidates_min"] == report["candidates_max"] == 0
    assert report["candidates_mean"] == 0


@pytest.mark.parametrize(
    "columns", [["query_id", "answer", "index"], ["query_id"], ["query_id", "query_id"]]
)
def test_column_contract(frames, columns):
    _, queries, items = frames
    answer = pd.DataFrame(columns=columns)
    with pytest.raises(ValueError, match="exactly the columns"):
        validate_submission(answer, queries, items)


def test_writer_rejects_mismatched_lengths_without_overwriting(frames, tmp_path):
    _, queries, items = frames
    path = tmp_path / "answer.csv"
    path.write_text("previous answer", encoding="utf-8")
    with pytest.raises(ValueError, match="same number"):
        write_submission(queries["query_id"], [[ITEM_A]], path, items)
    assert path.read_text(encoding="utf-8") == "previous answer"


def test_numeric_corpus_ids_are_rejected(frames):
    answer, queries, _ = frames
    with pytest.raises(ValueError, match="lowercase hex"):
        validate_submission(answer, queries, pd.DataFrame({"item_id": [1]}))


@pytest.mark.parametrize("bad_row", [[""], [f"{ITEM_A} {ITEM_B}"], [1], ITEM_A, None])
def test_writer_rejects_ambiguous_candidate_values(frames, tmp_path, bad_row):
    _, queries, items = frames
    path = tmp_path / "answer.csv"
    with pytest.raises(ValueError):
        write_submission(queries["query_id"], [bad_row, []], path, items)
    assert not path.exists()


def test_cli_keeps_numeric_looking_ids_and_empty_answers(frames, tmp_path):
    _, queries, items = frames
    path = tmp_path / "answer.csv"
    write_submission(queries["query_id"], [[ITEM_A], []], path, items)
    query_path, item_path = tmp_path / "queries.parquet", tmp_path / "items.parquet"
    queries.to_parquet(query_path)
    items.to_parquet(item_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "avito_retrieval.submission",
            "--answer",
            str(path),
            "--queries",
            str(query_path),
            "--items",
            str(item_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '"empty_answers": 1' in result.stdout
