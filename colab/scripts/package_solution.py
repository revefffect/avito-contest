"""Собрать локальный архив для повторного получения answer.csv без нейросети и сети."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
ID_TABLES = {
    "queries_ids": (
        "data/raw/benchmark_queries.parquet",
        "query_id",
        "data/ids/benchmark_queries.parquet",
    ),
    "allowed_items_ids": (
        "data/raw/benchmark_items.parquet",
        "item_id",
        "data/ids/benchmark_items.parquet",
    ),
    "corpus_ids": (
        "data/processed/corpus.parquet",
        "item_id",
        "data/ids/corpus.parquet",
    ),
}


def project_path(value, must_exist=True):
    """Путь должен оставаться в проекте и не проходить через символические ссылки."""
    if "\\" in str(value):
        raise ValueError(f"Обратные разделители пути запрещены: {value}")
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    try:
        relative = path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"Путь вне проекта: {value}") from error
    if ".." in relative.parts:
        raise ValueError(f"Переходы к родительской папке запрещены: {value}")
    current = ROOT
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Символическая ссылка запрещена: {current}")
    resolved = path.resolve(strict=must_exist)
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Путь вне проекта: {value}")
    if must_exist and not resolved.is_file():
        raise ValueError(f"Нужен обычный файл: {value}")
    return resolved


def prepare_id_tables():
    """Сохранить исходный порядок ID отдельно от текстов объявлений и запросов."""
    result = {}
    for key, (source, column, destination) in ID_TABLES.items():
        source_path = project_path(source)
        destination_path = project_path(destination, must_exist=False)
        table = pq.read_table(
            source_path, columns=[column], use_threads=False
        ).replace_schema_metadata(None)
        ids = table.column(column).to_pylist()
        if not ids or any(not isinstance(value, str) for value in ids):
            raise ValueError(f"Ожидались непустые строковые ID: {source}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"ID должны быть уникальными: {source}")
        # Не переписываем равную таблицу: её хеш уже мог попасть в финальную конфигурацию.
        unchanged = destination_path.exists() and pq.read_table(
            destination_path, use_threads=False
        ).equals(table)
        if not unchanged:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=destination_path.parent, suffix=".parquet", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                pq.write_table(table, temporary, compression="zstd")
                os.replace(temporary, destination_path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        with destination_path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        result[key] = {"path": destination, "rows": len(ids), "sha256": digest}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "deliverables/avito_solution.zip"
    )
    parser.add_argument(
        "--prepare-ids",
        action="store_true",
        help="Подготовить только три ID-таблицы, без финального config и архива",
    )
    args = parser.parse_args()
    if args.prepare_ids:
        print(json.dumps(prepare_id_tables(), ensure_ascii=False, indent=2))
        return

    config_path = project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    answer_path = project_path("answer.csv")
    if not isinstance(config.get("models"), list) or not config["models"]:
        raise ValueError("Финальный config должен содержать непустой список models")
    for key, (_, _, destination) in ID_TABLES.items():
        if config.get(key, destination) != destination:
            raise ValueError(f"Для переносимого архива {key} должен быть {destination}")

    files = {
        config_path,
        answer_path,
        project_path("pyproject.toml"),
        project_path("README.md"),
    }
    configurations = [(config_path, config, answer_path)]
    alternatives = config.get("alternatives", [])
    if not isinstance(alternatives, list):
        raise ValueError("alternatives должен быть списком config/answer")
    for alternative in alternatives:
        if not isinstance(alternative, dict) or not {"config", "answer"}.issubset(
            alternative
        ):
            raise ValueError("У альтернативы должны быть поля config и answer")
        if any(Path(alternative[key]).is_absolute() for key in ("config", "answer")):
            raise ValueError("Пути альтернатив должны быть относительными")
        alternative_config_path = project_path(alternative["config"])
        alternative_answer_path = project_path(alternative["answer"])
        if (
            alternative_config_path.suffix != ".json"
            or alternative_answer_path.suffix != ".csv"
        ):
            raise ValueError("Для альтернативы нужны JSON-конфигурация и CSV-ответ")
        alternative_config = json.loads(
            alternative_config_path.read_text(encoding="utf-8")
        )
        configurations.append(
            (alternative_config_path, alternative_config, alternative_answer_path)
        )
        files.update((alternative_config_path, alternative_answer_path))
    selection_path = ROOT / "configs/selection.json"
    if selection_path.exists():
        files.add(project_path(selection_path))
    for folder, suffixes in (
        ("src", {".py"}),
        ("scripts", {".py"}),
        ("tests", {".py"}),
        ("docs", {".md"}),
        ("reports", {".json", ".md"}),
    ):
        directory = ROOT / folder
        if directory.is_symlink():
            raise ValueError(f"Символическая ссылка запрещена: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Символическая ссылка запрещена: {path}")
            if any(part.startswith(".") for part in path.relative_to(directory).parts):
                continue
            if path.is_file() and path.suffix in suffixes:
                if folder == "reports" and path.stat().st_size > 10 * 1024 * 1024:
                    continue
                files.add(project_path(path))
    files.update(project_path(path) for path in ROOT.glob("requirements*.txt"))

    # В artifacts берём только явно выбранные модели и числовые признаки benchmark.
    selected = [(config["benchmark_features"], ".parquet")]
    selected.extend((model["path"], ".cbm") for model in config["models"])
    for value, suffix in selected:
        if Path(value).is_absolute():
            raise ValueError(f"В config нужны относительные пути: {value}")
        path = project_path(value)
        if not path.is_relative_to(ROOT / "artifacts") or path.suffix != suffix:
            raise ValueError(f"Недопустимый выбранный артефакт: {value}")
        files.add(path)
        if suffix == ".cbm":
            files.add(project_path(path.with_suffix(".feature_names.json")))
            files.add(project_path(path.with_suffix(".train_config.json")))
        else:
            import pyarrow as pa

            schema = pq.read_schema(path)
            if not {"qid", "item_idx"}.issubset(schema.names) or not all(
                pa.types.is_integer(field.type)
                or pa.types.is_floating(field.type)
                or pa.types.is_boolean(field.type)
                for field in schema
            ):
                raise ValueError(
                    "benchmark_features должен содержать только числовые признаки кандидатов"
                )
            sidecar = path.with_suffix(".metadata.json")
            if sidecar.exists():
                files.add(project_path(sidecar))
    for _, _, selected_answer in configurations:
        report_path = selected_answer.with_suffix(".report.json")
        if report_path.exists():
            files.add(project_path(report_path))

    id_tables = prepare_id_tables()
    files.update(project_path(record["path"]) for record in id_tables.values())
    records = {}
    for path in sorted(files, key=lambda value: value.relative_to(ROOT).as_posix()):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        records[path.relative_to(ROOT).as_posix()] = {
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    if "manifest.json" in records:
        raise ValueError("Имя manifest.json зарезервировано для описи архива")
    for selected_config_path, selected_config, selected_answer in configurations:
        answer_name = selected_answer.relative_to(ROOT).as_posix()
        config_name = selected_config_path.relative_to(ROOT).as_posix()
        if (
            selected_config.get("answer_sha256")
            and selected_config["answer_sha256"] != records[answer_name]["sha256"]
        ):
            raise ValueError(
                f"{answer_name} не соответствует answer_sha256 в {config_name}"
            )
        for name, expected in selected_config.get("input_sha256", {}).items():
            if name not in records or records[name]["sha256"] != expected:
                raise ValueError(
                    f"Входной файл отсутствует в архиве или его хеш изменился: {name} ({config_name})"
                )

    manifest = {
        "format_version": 1,
        "purpose": "offline_quick_reproduction",
        "config": config_path.relative_to(ROOT).as_posix(),
        "answer_sha256": records["answer.csv"]["sha256"],
        "files": records,
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output = project_path(args.output, must_exist=False)
    if output.suffix != ".zip":
        raise ValueError("Выходной файл должен иметь расширение .zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, suffix=".zip", delete=False
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name in sorted([*records, "manifest.json"]):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                if name == "manifest.json":
                    archive.writestr(info, manifest_bytes)
                else:
                    with (
                        project_path(name).open("rb") as source,
                        archive.open(info, "w", force_zip64=True) as target,
                    ):
                        shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        # Проверяем именно сохранённый архив до замены предыдущего готового результата.
        with zipfile.ZipFile(temporary) as archive:
            if sorted(archive.namelist()) != sorted([*records, "manifest.json"]):
                raise ValueError("Набор файлов в архиве не совпал с manifest")
            if archive.read("manifest.json") != manifest_bytes:
                raise ValueError("Содержимое manifest в архиве изменилось")
            for name, record in records.items():
                with archive.open(name) as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if (
                    digest != record["sha256"]
                    or archive.getinfo(name).file_size != record["bytes"]
                ):
                    raise ValueError(f"Проверка файла в архиве не пройдена: {name}")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    with output.open("rb") as handle:
        archive_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    print(
        json.dumps(
            {
                "archive": str(output),
                "sha256": archive_sha,
                "files": len(records),
                "bytes": output.stat().st_size,
                "verified": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
