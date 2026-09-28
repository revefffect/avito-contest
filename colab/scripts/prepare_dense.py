#!/usr/bin/env python3
"""Подготовка E5 и возобновляемый расчёт эмбеддингов корпуса и запросов."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_retrieval.dense import E5Encoder, download_model, prepare_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument(
        "--items-file",
        type=Path,
        help="Корпус для локальной оценки, включая heldout объявления",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        help="Отдельный Parquet с query_id и признаками запроса",
    )
    parser.add_argument(
        "--query-name",
        default="benchmark_queries",
        help="Имя выходных файлов запросов без расширения",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/dense")
    parser.add_argument(
        "--query-text-only",
        action="store_true",
        help="Кодировать текст запроса без фильтров поиска",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "artifacts/models/multilingual-e5-small",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument(
        "--precision", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--checkpoint-rows", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, help="Число первых строк для замера скорости"
    )
    parser.add_argument("--kind", choices=("items", "queries", "both"), default="both")
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Однократно загрузить открытую модель",
    )
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.download_only and not args.download_model:
        parser.error("--download-only требует --download-model")
    if args.download_model:
        download_model(args.model_dir)
    if args.download_only:
        return
    encoder = E5Encoder(
        args.model_dir, args.device, args.seed, args.threads, args.precision
    )
    print(
        f"E5: устройство {encoder.device}, вычисления {encoder.precision}, размерность {encoder.dimension}",
        flush=True,
    )
    kinds = ("items", "queries") if args.kind == "both" else (args.kind,)
    for kind in kinds:
        source = args.items_file if kind == "items" else args.queries_file
        source = source or args.data_dir / f"benchmark_{kind}.parquet"
        result = prepare_embeddings(
            source,
            args.output_dir,
            encoder,
            kind=kind,
            batch_size=args.batch_size,
            max_length=args.max_length if kind == "items" else args.query_max_length,
            limit=args.limit,
            checkpoint_rows=args.checkpoint_rows,
            output_name="items" if kind == "items" else args.query_name,
            query_text_only=args.query_text_only and kind == "queries",
        )
        print(
            json.dumps(
                {
                    "kind": kind,
                    "complete": result["complete"],
                    "rows": result["completed_rows"],
                    "seconds": result["elapsed_seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
