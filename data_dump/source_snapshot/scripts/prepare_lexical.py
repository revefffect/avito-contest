"""Построить воспроизводимые текстовые индексы из локального корпуса."""

import argparse
import json
import logging
import os
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np
import pandas as pd

from avito_retrieval.lexical import LexicalIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items", type=Path, default=Path("data/raw/benchmark_items.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/lexical.joblib"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    start = time.monotonic()
    columns = [
        "item_id",
        "item_title_raw",
        "item_infm_params_text",
        "item_description_raw",
        "item_location_id",
    ]
    items = pd.read_parquet(args.items, columns=columns)
    index = LexicalIndex(batch_size=args.batch_size).fit(items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=args.output.parent, suffix=".joblib", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        # Без сжатия индекс быстрее загружается и поддерживает mmap_mode="r".
        joblib.dump(index, temporary_path, compress=0)
        os.replace(temporary_path, args.output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    np.save(
        args.output.with_suffix(".item_ids.npy"), index.item_ids_, allow_pickle=False
    )
    metadata = {
        **index.metadata_,
        "source": str(args.items),
        "build_seconds": round(time.monotonic() - start, 3),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
