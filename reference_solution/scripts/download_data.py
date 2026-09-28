"""Получить исходный архив конкурса или проверить его локальную копию."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
import tempfile
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_KEY = "https://disk.yandex.ru/d/sNhfo0YOjGtufg"
DOWNLOAD_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
ARCHIVE_SHA256 = "8dd3cba59201bae333c11db89c5111198fa10cc52a70c70bdc57bd6a248fb777"
EXPECTED_ROWS = {
    "train.parquet": 497673,
    "benchmark_queries.parquet": 2452,
    "benchmark_items.parquet": 189212,
}
CHUNK_SIZE = 4 * 1024 * 1024


def check_archive_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != ARCHIVE_SHA256:
        raise ValueError(f"SHA256 архива не совпал: {actual}")
    return actual


def save_archive(stream, destination):
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with temporary.open("wb") as output:
            shutil.copyfileobj(stream, output, length=CHUNK_SIZE)
        check_archive_hash(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_archive(data_dir, source=None):
    destination = data_dir / "dataset.zip"
    if source is not None:
        source = source.expanduser().resolve()
        if source == destination.resolve():
            check_archive_hash(destination)
        else:
            with source.open("rb") as stream:
                save_archive(stream, destination)
        return destination

    if destination.exists():
        check_archive_hash(destination)
        return destination

    # Сеть используется только при явном запуске этого скрипта без готового архива.
    api_url = DOWNLOAD_API + "?" + urlencode({"public_key": PUBLIC_KEY})
    with urlopen(api_url, timeout=60) as response:
        href = json.load(response)["href"]
    if urlparse(href).scheme != "https":
        raise ValueError("Сервис загрузки вернул ссылку без HTTPS")
    with urlopen(href, timeout=120) as stream:
        save_archive(stream, destination)
    return destination


def archive_members(archive):
    members = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ValueError(f"Недопустимый путь в архиве: {info.filename!r}")
        if info.is_dir():
            continue
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
            raise ValueError(f"Недопустимый тип файла: {info.filename!r}")
        if path.name not in EXPECTED_ROWS or path.name in members:
            raise ValueError(f"Лишний или повторный файл в архиве: {info.filename!r}")
        members[path.name] = info
    if set(members) != set(EXPECTED_ROWS):
        raise ValueError("В архиве должны быть все три исходных Parquet-файла")
    return members


def verify_row_counts(directory):
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        if error.name != "pyarrow":
            raise
        return {"checked": False, "reason": "pyarrow не установлен"}

    actual = {}
    for name, expected in EXPECTED_ROWS.items():
        rows = parquet.ParquetFile(directory / name).metadata.num_rows
        if rows != expected:
            raise ValueError(
                f"Неверное число строк в {name}: {rows}, ожидалось {expected}"
            )
        actual[name] = rows
    return {"checked": True, "rows": actual}


def extract_data(archive_path, data_dir):
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Сначала проверяем всю распаковку, затем заменяем готовые файлы по одному.
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=data_dir) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            for name, member in archive_members(archive).items():
                with (
                    archive.open(member) as source,
                    (staging / name).open("wb") as target,
                ):
                    shutil.copyfileobj(source, target, length=CHUNK_SIZE)
        verification = verify_row_counts(staging)
        for name in EXPECTED_ROWS:
            os.replace(staging / name, raw_dir / name)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="Готовый dataset.zip, без обращения к сети"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    archive = prepare_archive(args.data_dir, args.archive)
    verification = extract_data(archive, args.data_dir)
    print(
        json.dumps(
            {
                "archive": str(archive),
                "archive_sha256": ARCHIVE_SHA256,
                "raw_directory": str(args.data_dir / "raw"),
                "parquet": verification,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
