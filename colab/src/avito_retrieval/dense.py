"""Локальные эмбеддинги E5 с фиксированной моделью и продолжением расчёта.

Pooling и префиксы соответствуют официальной карточке модели:
https://huggingface.co/intfloat/multilingual-e5-small
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "README.md",
)
ITEM_COLUMNS = (
    "item_id",
    "item_title_raw",
    "item_infm_params_text",
    "item_description_raw",
)
QUERY_COLUMNS = ("query_id", "search_query", "search_infm_params_text")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def download_model(model_dir: str | Path) -> Path:
    """Единственный сетевой шаг, последующий инференс работает только локально."""
    from huggingface_hub import snapshot_download

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(model_dir),
        allow_patterns=list(MODEL_FILES),
        max_workers=3,
    )
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Неполная загрузка модели: {missing}")
    _save_json(
        model_dir / "provenance.json",
        {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "model_card": f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/README.md",
            "sha256": {name: file_sha256(model_dir / name) for name in MODEL_FILES},
        },
    )
    return model_dir


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def query_text(row: dict, include_filters: bool = True) -> str:
    parts = [_text(row.get("search_query"))]
    if include_filters:
        parts.append(_text(row.get("search_infm_params_text")))
    return "\n".join(filter(None, parts))


def item_text(row: dict, description_chars: int = 2000) -> str:
    # Заголовок и параметры идут первыми: длинное описание обрезается токенизатором.
    return "\n".join(
        filter(
            None,
            (
                _text(row.get("item_title_raw")),
                _text(row.get("item_infm_params_text")),
                _text(row.get("item_description_raw"))[:description_chars],
            ),
        )
    )


class E5Encoder:
    def __init__(
        self,
        model_dir: str | Path,
        device: str = "auto",
        seed: int = 42,
        threads: int = 4,
        precision: str = "float32",
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        if device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device должен быть auto, cpu, mps или cuda")
        if precision not in {"float32", "float16"}:
            raise ValueError("precision должен быть float32 или float16")
        model_dir = Path(model_dir)
        provenance_path = model_dir / "provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(
                "Модель не подготовлена. Сначала выполните --download-model."
            )
        self.provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (self.provenance.get("model_id"), self.provenance.get("revision")) != (
            MODEL_ID,
            MODEL_REVISION,
        ):
            raise ValueError(
                "Локальная модель не соответствует закреплённой ревизии E5"
            )
        for name in MODEL_FILES:
            if file_sha256(model_dir / name) != self.provenance["sha256"].get(name):
                raise ValueError(f"Контрольная сумма модели не совпала: {name}")
        self.device = device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA недоступна. В Colab включите GPU в настройках среды."
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS недоступен в текущем процессе. Выберите --device cpu."
            )
        if self.device == "cpu" and precision == "float16":
            raise ValueError("Для CPU используйте float32")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        self.seed = seed
        self.precision = precision
        self.versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        dtype = torch.float16 if precision == "float16" else torch.float32
        self.model = (
            AutoModel.from_pretrained(
                model_dir,
                local_files_only=True,
                dtype=dtype,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )
        self.dimension = int(self.model.config.hidden_size)

    def encode(
        self,
        texts: Sequence[str],
        kind: str = "query",
        max_length: int = 256,
        batch_size: int = 32,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        if kind not in {"query", "passage"}:
            raise ValueError("kind должен быть query или passage")
        if batch_size < 1 or not 1 <= max_length <= 512:
            raise ValueError("Нужны batch_size > 0 и max_length в пределах 1..512")
        embeddings = np.empty((len(texts), self.dimension), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = [
                    f"{kind}: {text}" for text in texts[start : start + batch_size]
                ]
                encoded = self.tokenizer(
                    batch,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**encoded).last_hidden_state.float()
                mask = encoded["attention_mask"].unsqueeze(-1).bool()
                pooled = hidden.masked_fill(~mask, 0).sum(1) / mask.sum(1).clamp_min(1)
                normalized = functional.normalize(pooled, p=2, dim=1)
                embeddings[start : start + len(batch)] = normalized.cpu().numpy()
        if not np.isfinite(embeddings).all():
            raise FloatingPointError("Модель вернула нечисловые эмбеддинги")
        return embeddings


def prepare_embeddings(
    parquet_path: str | Path,
    output_dir: str | Path,
    encoder: E5Encoder,
    kind: str = "items",
    batch_size: int = 32,
    max_length: int = 256,
    limit: int | None = None,
    checkpoint_rows: int = 256,
    output_name: str | None = None,
    query_text_only: bool = False,
) -> dict:
    """Порядок строк сохраняется, после сбоя повторяются только незафиксированные строки."""
    import pyarrow.parquet as pq

    if kind not in {"items", "queries"}:
        raise ValueError("kind должен быть items или queries")
    if query_text_only and kind != "queries":
        raise ValueError("query_text_only применим только к запросам")
    output_name = output_name or kind
    if not output_name.replace("_", "").isalnum():
        raise ValueError("output_name должен состоять из букв, цифр и подчёркиваний")
    if checkpoint_rows < 1 or batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Размеры порций и limit должны быть положительными")
    parquet_path, output_dir = Path(parquet_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ITEM_COLUMNS if kind == "items" else QUERY_COLUMNS
    source = pq.ParquetFile(parquet_path)
    missing = set(columns) - set(source.schema_arrow.names)
    if missing:
        raise ValueError(
            f"В {parquet_path.name} отсутствуют колонки: {sorted(missing)}"
        )
    ids = source.read(columns=[columns[0]]).column(0).to_pylist()
    if limit is not None:
        ids = ids[:limit]
    if not ids or any(not isinstance(value, str) for value in ids):
        raise ValueError("Нужен непустой набор строковых идентификаторов")
    if len(set(ids)) != len(ids):
        raise ValueError("Идентификаторы должны быть уникальными")
    ids_sha = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    config = {
        "format_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": encoder.provenance["sha256"]["model.safetensors"],
        "source_name": parquet_path.name,
        "source_sha256": file_sha256(parquet_path),
        "ids_sha256": ids_sha,
        "rows": len(ids),
        "dimension": encoder.dimension,
        "dtype": "float32",
        "precision": encoder.precision,
        "device": encoder.device,
        "seed": encoder.seed,
        "versions": encoder.versions,
        "kind": kind,
        "batch_size": batch_size,
        "max_length": max_length,
        "checkpoint_rows": checkpoint_rows,
        "text_version": 1,
        "description_chars": 2000,
        "prefix": "passage: " if kind == "items" else "query: ",
        "pooling": "masked_mean_l2",
    }
    if query_text_only:
        config["query_text_only"] = True
    metadata_path = output_dir / f"{output_name}.meta.json"
    embedding_path = output_dir / f"{output_name}.npy"
    ids_path = output_dir / f"{output_name}.ids.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["config"] != config:
            raise ValueError(
                "Конфигурация или входные данные изменились. Выберите другую output-dir."
            )
        if json.loads(ids_path.read_text(encoding="utf-8")) != ids:
            raise ValueError("Порядок идентификаторов в кеше изменился")
        output = np.load(embedding_path, mmap_mode="r+")
        if output.shape != (len(ids), encoder.dimension) or output.dtype != np.float32:
            raise ValueError("Форма или тип кеша эмбеддингов не совпали")
        if metadata["complete"]:
            if file_sha256(embedding_path) != metadata["embeddings_sha256"]:
                raise ValueError("Контрольная сумма эмбеддингов не совпала")
            print(f"{kind}: готовый кеш, {len(ids):,} строк", flush=True)
            return metadata
    else:
        if embedding_path.exists() or ids_path.exists():
            raise FileExistsError(
                "Обнаружен неполный кеш без метаданных. Выберите другую output-dir."
            )
        output = np.lib.format.open_memmap(
            embedding_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(ids), encoder.dimension),
        )
        _save_json(ids_path, ids)
        metadata = {
            "config": config,
            "completed_rows": 0,
            "complete": False,
            "elapsed_seconds": 0.0,
        }
        _save_json(metadata_path, metadata)
    offset = 0
    resumed_at = int(metadata["completed_rows"])
    if not 0 <= resumed_at <= len(ids):
        raise ValueError("Некорректная позиция продолжения расчёта")
    started = time.monotonic()
    previous_seconds = float(metadata["elapsed_seconds"])
    for records in source.iter_batches(
        batch_size=checkpoint_rows, columns=list(columns)
    ):
        end = min(offset + len(records), len(ids))
        if end <= resumed_at:
            offset += len(records)
            continue
        if offset >= len(ids):
            break
        begin = max(offset, resumed_at)
        rows = records.slice(begin - offset, end - begin).to_pylist()
        texts = [
            (
                item_text(row)
                if kind == "items"
                else query_text(row, include_filters=not query_text_only)
            )
            for row in rows
        ]
        output[begin:end] = encoder.encode(
            texts,
            kind="passage" if kind == "items" else "query",
            max_length=max_length,
            batch_size=batch_size,
        )
        # Сначала сохраняем числа, затем позицию: сбой не оставит «готовые» пустые строки.
        output.flush()
        elapsed = time.monotonic() - started
        metadata.update(completed_rows=end, elapsed_seconds=previous_seconds + elapsed)
        _save_json(metadata_path, metadata)
        speed = (end - resumed_at) / max(elapsed, 1e-6)
        eta = (len(ids) - end) / max(speed, 1e-6)
        print(
            f"{kind}: {end:,}/{len(ids):,}, {speed:.1f} строк/с, осталось {eta / 60:.1f} мин",
            flush=True,
        )
        offset += len(records)
    metadata.update(complete=True, embeddings_sha256=file_sha256(embedding_path))
    _save_json(metadata_path, metadata)
    return metadata


def load_embeddings(
    output_dir: str | Path,
    kind: str = "items",
    verify_hash: bool = True,
) -> tuple[np.ndarray, list[str]]:
    output_dir = Path(output_dir)
    metadata = json.loads(
        (output_dir / f"{kind}.meta.json").read_text(encoding="utf-8")
    )
    if not metadata["complete"]:
        raise ValueError("Расчёт эмбеддингов ещё не завершён")
    embedding_path = output_dir / f"{kind}.npy"
    if verify_hash and file_sha256(embedding_path) != metadata["embeddings_sha256"]:
        raise ValueError("Контрольная сумма эмбеддингов не совпала")
    values = np.load(embedding_path, mmap_mode="r")
    ids = json.loads((output_dir / f"{kind}.ids.json").read_text(encoding="utf-8"))
    ids_sha = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    if ids_sha != metadata["config"]["ids_sha256"]:
        raise ValueError("Контрольная сумма идентификаторов не совпала")
    if (
        values.shape != (len(ids), metadata["config"]["dimension"])
        or values.dtype != np.float32
        or metadata["completed_rows"] != len(ids)
    ):
        raise ValueError("Идентификаторы и эмбеддинги не совпадают по размеру")
    return values, ids
