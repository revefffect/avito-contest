"""Повторить сохранение ответа или полностью пересобрать решение без сетевых API."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(config):
    for name, expected in config.get("input_sha256", {}).items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Input path is outside the project: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {name}")


def run(script, *arguments):
    command = [sys.executable, "-u", str(ROOT / "scripts" / script)]
    command.extend(str(value) for value in arguments)
    print("Запуск:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument("--output", type=Path, default=ROOT / "answer.csv")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Повторить подготовку, обучение и поиск, данные и веса должны быть уже скачаны",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    args = parser.parse_args()
    # Явные относительные пути относятся к папке запуска пользователя.
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    config = json.loads(args.config.read_text())
    if not args.rebuild:
        verify_inputs(config)

    if args.rebuild:
        run(
            "prepare_validation.py",
            "--seed",
            config["split_seed"],
            "--queries",
            config["validation_count"],
        )
        run(
            "prepare_lexical.py",
            "--items",
            "data/processed/corpus.parquet",
            "--output",
            "artifacts/lexical.joblib",
        )
        dense = config["dense"]
        dense_args = [
            "--device",
            args.device,
            "--batch-size",
            dense["batch_size"],
            "--checkpoint-rows",
            dense["checkpoint_rows"],
            "--max-length",
            dense["item_max_length"],
            "--query-max-length",
            dense["query_max_length"],
            "--precision",
            dense["precision"],
            "--seed",
            dense["seed"],
        ]
        run(
            "prepare_dense.py",
            "--kind",
            "items",
            "--items-file",
            "data/processed/corpus.parquet",
            *dense_args,
        )
        for name, source in [
            ("eval_queries", "data/processed/eval_queries.parquet"),
            ("benchmark_queries", "data/raw/benchmark_queries.parquet"),
        ]:
            run(
                "prepare_dense.py",
                "--kind",
                "queries",
                "--queries-file",
                source,
                "--query-name",
                name,
                *dense_args,
            )
            run(
                "prepare_dense.py",
                "--kind",
                "queries",
                "--queries-file",
                source,
                "--query-name",
                name + "_plain",
                "--query-text-only",
                *dense_args,
            )
        run(
            "build_features.py",
            "--dense",
            "--plain-query",
            "--output",
            config["validation_features"],
        )
        for model in config["models"]:
            options = [
                "--features",
                config["validation_features"],
                "--output",
                model["path"],
                "--loss",
                model["loss"],
                "--iterations",
                model["iterations"],
                "--depth",
                model["depth"],
                "--learning-rate",
                model["learning_rate"],
                "--seed",
                model["seed"],
                "--threads",
                model["threads"],
                "--fit-splits",
                "fit,dev",
                "--fixed-iterations",
            ]
            if model.get("drop_features"):
                options += ["--drop-features", ",".join(model["drop_features"])]
            run("train_ranker.py", *options)
        run(
            "build_features.py",
            "--mode",
            "benchmark",
            "--dense",
            "--plain-query",
            "--output",
            config["benchmark_features"],
        )
        run("package_solution.py", "--prepare-ids")

    options = [
        "--features",
        config["benchmark_features"],
        "--output",
        args.output,
        "--queries",
        config.get("queries_ids", "data/raw/benchmark_queries.parquet"),
        "--corpus",
        config.get("corpus_ids", "data/processed/corpus.parquet"),
        "--allowed-items",
        config.get("allowed_items_ids", "data/raw/benchmark_items.parquet"),
    ]
    for model in config["models"]:
        options += ["--model", model["path"]]
    options += ["--weights", *config["weights"]]
    run("predict.py", *options)
    expected = config.get("answer_sha256")
    actual = sha256_file(args.output)
    if expected and actual != expected:
        raise ValueError(
            f"Answer differs from the released file: expected {expected}, got {actual}"
        )
    status = "SHA256 ответа подтверждён" if expected else "SHA256 созданного ответа"
    print(f"{status}: {actual}", flush=True)


if __name__ == "__main__":
    main()
