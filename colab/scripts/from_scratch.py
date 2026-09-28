"""Запустить один этап полного обучения, не подгружая готовые модели конкурса."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    "download",
    "split",
    "lexical",
    "model",
    "embeddings",
    "train_features",
    "train",
    "audit",
    "benchmark",
    "predict",
]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def commands(stage, config, device, archive=None):
    dense = config["dense"]
    model_args = []
    for model in config["models"]:
        model_args.extend(["--model", model["path"]])
    if stage == "download":
        return [("download_data.py", *(["--archive", str(archive)] if archive else []))]
    if stage == "split":
        return [
            (
                "prepare_validation.py",
                "--seed",
                str(config["split_seed"]),
                "--queries",
                str(config["validation_count"]),
            )
        ]
    if stage == "lexical":
        return [
            (
                "prepare_lexical.py",
                "--items",
                "data/processed/corpus.parquet",
                "--output",
                "artifacts/lexical.joblib",
            )
        ]
    if stage == "model":
        return [("prepare_dense.py", "--download-model", "--download-only")]
    if stage == "embeddings":
        common = [
            "--device",
            device,
            "--batch-size",
            str(dense["batch_size"]),
            "--checkpoint-rows",
            str(dense["checkpoint_rows"]),
            "--max-length",
            str(dense["item_max_length"]),
            "--query-max-length",
            str(dense["query_max_length"]),
            "--precision",
            dense["precision"],
            "--seed",
            str(dense["seed"]),
        ]
        result = [
            (
                "prepare_dense.py",
                "--kind",
                "items",
                "--items-file",
                "data/processed/corpus.parquet",
                *common,
            )
        ]
        for name, source in [
            ("eval_queries", "data/processed/eval_queries.parquet"),
            ("benchmark_queries", "data/raw/benchmark_queries.parquet"),
        ]:
            for plain in [False, True]:
                result.append(
                    (
                        "prepare_dense.py",
                        "--kind",
                        "queries",
                        "--queries-file",
                        source,
                        "--query-name",
                        name + ("_plain" if plain else ""),
                        *(["--query-text-only"] if plain else []),
                        *common,
                    )
                )
        return result
    if stage in ["train_features", "benchmark"]:
        return [
            (
                "build_features.py",
                "--mode",
                "validation" if stage == "train_features" else "benchmark",
                "--dense",
                "--plain-query",
                "--output",
                (
                    config["validation_features"]
                    if stage == "train_features"
                    else config["benchmark_features"]
                ),
            )
        ]
    if stage == "train":
        result = []
        for model in config["models"]:
            args = [
                "train_ranker.py",
                "--features",
                config["validation_features"],
                "--output",
                model["path"],
                "--fit-splits",
                "fit,dev",
                "--fixed-iterations",
            ]
            for key in [
                "loss",
                "iterations",
                "depth",
                "learning_rate",
                "seed",
                "threads",
            ]:
                args.extend(["--" + key.replace("_", "-"), str(model[key])])
            args.extend(["--drop-features", ",".join(model["drop_features"])])
            result.append(tuple(args))
        return result
    if stage == "audit":
        return [
            (
                "compare_models.py",
                "--features",
                config["validation_features"],
                *model_args,
                "--split",
                "audit",
                "--blend-grid",
                "0.5",
                "--output",
                "reports/audit.json",
            )
        ]
    if stage == "predict":
        return [
            (
                "predict.py",
                "--features",
                config["benchmark_features"],
                *model_args,
                "--weights",
                *map(str, config["weights"]),
                "--queries",
                "data/raw/benchmark_queries.parquet",
                "--corpus",
                "data/processed/corpus.parquet",
                "--allowed-items",
                "data/raw/benchmark_items.parquet",
                "--output",
                "answer.csv",
            )
        ]
    raise ValueError(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES + ["all"], required=True)
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu", "mps"], default="auto"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument("--dataset-archive", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    archive = args.dataset_archive.resolve() if args.dataset_archive else None
    config = json.loads(config_path.read_text())
    os.chdir(ROOT)
    os.environ.update(
        PYTHONPATH=str(ROOT / "src"),
        PYTHONHASHSEED="42",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="3",
        TOKENIZERS_PARALLELISM="false",
    )
    for name in ["artifacts", "reports", "data"]:
        (ROOT / name).mkdir(exist_ok=True)
    path = ROOT / "reports/run_steps.json"
    history = json.loads(path.read_text()) if path.exists() else []
    for stage in STAGES if args.stage == "all" else [args.stage]:
        record = {
            "stage": stage,
            "started_at_unix": time.time(),
            "config_sha256": sha256(config_path),
            "commands": [],
            "status": "running",
        }
        history.append(record)
        path.write_text(json.dumps(history, indent=2) + "\n")
        try:
            with (ROOT / "reports" / f"{stage}.log").open("w") as log:
                for script, *arguments in commands(stage, config, args.device, archive):
                    command = [
                        sys.executable,
                        "-u",
                        str(ROOT / "scripts" / script),
                        *arguments,
                    ]
                    record["commands"].append([script, *arguments])
                    print("Запуск:", script, " ".join(arguments), flush=True)
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    if process.wait():
                        raise RuntimeError(
                            f"Этап {stage} завершился с ошибкой. См. reports/{stage}.log"
                        )
            record["status"] = "passed"
        except BaseException:
            record["status"] = "failed"
            raise
        finally:
            record["elapsed_seconds"] = round(
                time.time() - record["started_at_unix"], 3
            )
            path.write_text(json.dumps(history, indent=2) + "\n")
    if args.stage in ["all", "predict"]:
        actual = sha256(ROOT / "answer.csv")
        result = {
            "answer_sha256": actual,
            "reference_answer_sha256": config["reference_answer_sha256"],
            "matches_previous_mps_answer": actual == config["reference_answer_sha256"],
            "python": sys.version,
            "platform": platform.platform(),
            "config": config,
            "packages": {
                name: importlib.metadata.version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "scikit-learn",
                    "catboost",
                    "torch",
                    "transformers",
                    "huggingface-hub",
                ]
            },
            "source_sha256": {
                str(p.relative_to(ROOT)): sha256(p)
                for folder in ["scripts", "src"]
                for p in sorted((ROOT / folder).rglob("*.py"))
            },
        }
        (ROOT / "reports/reproducibility.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        print("SHA256 нового ответа:", actual)
        print(
            "Совпадение с прежним ответом MPS:", result["matches_previous_mps_answer"]
        )


if __name__ == "__main__":
    main()
