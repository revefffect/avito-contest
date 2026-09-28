"""Build or verify the portable submission without changing source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {"mini", "report", "evidence", "data_dump", "reference_solution", "scripts"}
ROOT_FILES = {"README.md", "LICENSE", "LICENSE.md", ".gitignore"}
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "qa",
    "build",
    "deliverables",
}
EXCLUDED_NAMES = {".DS_Store", "compile.json"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".ipynb",
    ".toml",
    ".yaml",
    ".yml",
    ".txt",
    ".tex",
    ".sty",
    ".log",
    ".csv",
    ".ini",
    ".cfg",
    ".sh",
}
SECRET_PATTERNS = {
    "private_key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "github_token": re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"
    ),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "openai_project_key": re.compile(rb"sk-proj-[A-Za-z0-9_-]{30,}"),
    "authorization_header": re.compile(
        rb"authorization\s*:\s*(?:bearer|basic)\s+[A-Za-z0-9+/=._-]{20,}", re.IGNORECASE
    ),
}
MANIFEST_NAMES = {"MANIFEST.sha256", "MANIFEST.json"}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and "\\" not in name
        and not any(part in {"", ".", ".."} for part in name.split("/"))
        and not any(char in name for char in "\r\n\x00")
        and ":" not in path.parts[0]
    )


def included(relative: Path) -> bool:
    parts = relative.parts
    if any(part in EXCLUDED_PARTS for part in parts) or relative.name in EXCLUDED_NAMES:
        return False
    if len(parts) == 1:
        return relative.name in ROOT_FILES
    if parts[0] not in TOP_LEVEL:
        return False
    if any(part.startswith(".") for part in parts):
        return False
    if relative.suffix in {".pyc", ".pyo", ".part", ".tmp"}:
        return False
    if relative.suffix == ".log" and parts[0] not in {
        "data_dump",
        "reference_solution",
    }:
        return False
    if parts[0] == "mini" and (relative.name == "answer.csv" or "outputs" in parts):
        return False
    return True


def scan_secrets(path: Path, relative: str) -> None:
    if path.name.lower() in {
        "credentials",
        "credentials.json",
        "id_rsa",
        "id_ed25519",
        ".env",
    }:
        raise ValueError(f"Potential credential file: {relative}")
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    with path.open("rb") as handle:
        tail = b""
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            data = tail + block
            for rule, pattern in SECRET_PATTERNS.items():
                if pattern.search(data):
                    raise ValueError(f"Potential secret ({rule}) in {relative}")
            tail = data[-512:]


def inventory(root: Path) -> dict[str, dict]:
    records = {}
    for base, directories, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        directories[:] = sorted(
            name
            for name in directories
            if name not in EXCLUDED_PARTS and not name.startswith(".")
        )
        for name in directories:
            path = base_path / name
            if path.is_symlink() and path.relative_to(root).parts[0] in TOP_LEVEL:
                raise ValueError(
                    f"Symlink directory is not allowed: {path.relative_to(root)}"
                )
        for name in sorted(files):
            path = base_path / name
            relative = path.relative_to(root)
            if not included(relative):
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Regular file required: {relative}")
            archive_name = relative.as_posix()
            if not safe_name(archive_name):
                raise ValueError(f"Unsafe archive name: {archive_name}")
            scan_secrets(path, archive_name)
            records[archive_name] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    required = {
        "README.md",
        "mini/solution.ipynb",
        "mini/config.json",
        "mini/requirements.txt",
        "report/avito_report.pdf",
        "scripts/package_mini.py",
    }
    missing = required - set(records)
    if missing:
        raise ValueError(f"Required package files missing: {sorted(missing)}")
    config = json.loads((root / "mini/config.json").read_text(encoding="utf-8"))
    for name, expected in config["files"].items():
        key = "mini/" + name
        if key not in records or records[key]["sha256"] != expected:
            raise ValueError(f"Frozen notebook input missing or changed: {key}")
    return dict(sorted(records.items()))


def zip_info(name: str) -> zipfile.ZipInfo:
    item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    item.create_system = 3
    item.external_attr = (stat.S_IFREG | 0o644) << 16
    item.compress_type = zipfile.ZIP_STORED
    return item


def verify(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        if any(not safe_name(name) for name in names):
            raise ValueError("Unsafe ZIP member name")
        if not MANIFEST_NAMES.issubset(names):
            raise ValueError("Both integrity manifests are required")
        for member in members:
            mode = member.external_attr >> 16
            if (
                member.is_dir()
                or stat.S_ISLNK(mode)
                or (mode and not stat.S_ISREG(mode))
            ):
                raise ValueError(f"Unexpected ZIP member type: {member.filename}")
        document = json.loads(bundle.read("MANIFEST.json"))
        if document.get("version") != 1:
            raise ValueError("Unsupported manifest version")
        records = document["files"]
        expected_lines = "".join(
            f"{record['sha256']}  {name}\n" for name, record in sorted(records.items())
        )
        if bundle.read("MANIFEST.sha256").decode("utf-8") != expected_lines:
            raise ValueError("SHA256 and size manifests disagree")
        if set(names) != set(records) | MANIFEST_NAMES:
            raise ValueError("Manifest does not exactly cover ZIP members")
        total_bytes = 0
        for name, record in sorted(records.items()):
            member = bundle.getinfo(name)
            if member.file_size != record["bytes"]:
                raise ValueError(f"ZIP header size mismatch: {name}")
            digest, actual_bytes = hashlib.sha256(), 0
            with bundle.open(member) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                    actual_bytes += len(block)
            if (
                actual_bytes != record["bytes"]
                or digest.hexdigest() != record["sha256"]
            ):
                raise ValueError(f"Content mismatch: {name}")
            total_bytes += actual_bytes
    return {
        "status": "passed",
        "archive": str(archive.resolve()),
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "payload_files": len(records),
        "zip_members": len(names),
        "payload_bytes": total_bytes,
        "all_member_hashes_and_sizes_passed": True,
        "exact_manifest_coverage": True,
        "data_dump_logs": sum(
            name.startswith("data_dump/") and name.endswith(".log") for name in records
        ),
        "reference_solution_logs": sum(
            name.startswith("reference_solution/") and name.endswith(".log")
            for name in records
        ),
        "ready_mini_answer_included": "mini/answer.csv" in records,
        "determinism": "Sorted members, fixed timestamps and permissions, ZIP_STORED payloads",
        "security_scope": "Names, file types and known credential patterns checked. Heuristic scanning is not proof that arbitrary private content is absent.",
    }


def build(root: Path, archive: Path) -> dict:
    records = inventory(root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".avito_mini_", suffix=".zip", dir=archive.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        manifests = {
            "MANIFEST.sha256": "".join(
                f"{record['sha256']}  {name}\n" for name, record in records.items()
            ).encode(),
            "MANIFEST.json": (
                json.dumps(
                    {"version": 1, "files": records},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
        }
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as bundle:
            for name in sorted(set(records) | MANIFEST_NAMES):
                if name in manifests:
                    bundle.writestr(zip_info(name), manifests[name])
                    continue
                digest, size = hashlib.sha256(), 0
                with (root / name).open("rb") as source, bundle.open(
                    zip_info(name), "w", force_zip64=True
                ) as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
                        digest.update(block)
                        size += len(block)
                if (
                    digest.hexdigest() != records[name]["sha256"]
                    or size != records[name]["bytes"]
                ):
                    raise ValueError(f"Source changed while packaging: {name}")
        report = verify(temporary)
        os.replace(temporary, archive)
        report["archive"] = str(archive.resolve())
        report["source_secret_scan_passed"] = True
        return report
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "verify"])
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Standalone submission project"
    )
    parser.add_argument(
        "--archive", type=Path, help="Default: ROOT/deliverables/avito_mini.zip"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Default: ROOT/deliverables/mini_package_verification.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    archive = (args.archive or root / "deliverables/avito_mini.zip").resolve()
    report_path = (
        args.report or root / "deliverables/mini_package_verification.json"
    ).resolve()
    if not report_path.is_relative_to(root / "deliverables"):
        parser.error("Verification reports must stay inside ROOT/deliverables")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = build(root, archive) if args.command == "build" else verify(archive)
    except Exception as error:
        failure = {
            "status": "failed",
            "archive": str(archive),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        report_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
