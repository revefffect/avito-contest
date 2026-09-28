"""Execute an isolated copy of the frozen mini notebook and compare the CSV SHA256.

Requires nbclient, nbformat and ipykernel plus mini/requirements.txt.
Local Jupyter uses loopback sockets. The notebook needs no external service.
Frozen report files are never overwritten by the default command.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-root", type=Path, default=ROOT / "mini")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "deliverables/notebook_verification.rerun.json",
    )
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=None,
        help="Optional directory for an isolated execution copy",
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="Timeout per notebook cell in seconds"
    )
    args = parser.parse_args()
    protected = [
        ROOT / name
        for name in ["report", "data_dump", "evidence", "reference_solution", "mini"]
    ]
    if any(args.output.resolve().is_relative_to(path) for path in protected):
        parser.error("Choose a new output outside frozen project inputs")
    if args.output.resolve().is_relative_to(args.mini_root.resolve()):
        parser.error("The notebook input directory must remain unchanged")
    if args.timeout <= 0:
        parser.error("Timeout must be positive")
    return args


def run(args):
    from datetime import datetime, timezone
    import ast
    import csv
    import hashlib
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    import platform
    import shutil
    import sys
    import tempfile
    import time

    import nbformat
    from nbclient import NotebookClient

    SOURCE = args.mini_root.resolve()
    REPORT_PATH = args.output.resolve()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORK = Path(tempfile.mkdtemp(prefix="avito_notebook_verify_", dir=args.temp_root))
    PACKAGE = WORK / "mini"
    IPYTHON = WORK / "ipython"
    JUPYTER = WORK / "jupyter"
    START = time.monotonic()

    def sha256(path):
        with Path(path).open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def save_report():
        temporary = REPORT_PATH.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, REPORT_PATH)

    report = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_notebook": str(SOURCE / "solution.ipynb"),
        "temporary_directory": str(WORK),
        "environment": {
            "python": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "catboost",
                    "nbclient",
                    "nbformat",
                    "ipykernel",
                ]
            },
        },
    }
    save_report()
    try:
        for path in SOURCE.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink in mini package: {path}")
        shutil.copytree(
            SOURCE,
            PACKAGE,
            ignore=shutil.ignore_patterns(
                "answer.csv", "outputs", ".ipynb_checkpoints", "__pycache__"
            ),
        )
        assert not (PACKAGE / "answer.csv").exists()
        assert not (WORK / "reference_solution").exists()
        source_hash = sha256(SOURCE / "solution.ipynb")
        config_hash = sha256(SOURCE / "config.json")
        config = json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))
        input_records = {}
        for relative, expected in config["files"].items():
            target = PACKAGE / relative
            assert target.resolve().is_relative_to(PACKAGE)
            actual = sha256(target)
            if actual != expected:
                raise ValueError(f"Input hash mismatch: {relative}")
            input_records[relative] = {"sha256": actual, "bytes": target.stat().st_size}
        notebook = nbformat.read(PACKAGE / "solution.ipynb", as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        source_code = "\n\n".join(cell.source for cell in code_cells)
        original_code_hash = hashlib.sha256(source_code.encode()).hexdigest()
        imported = set()
        for cell in code_cells:
            parsed = ast.parse(cell.source)
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.add(node.module or "")
        allowed = {"pathlib", "hashlib", "json", "re", "numpy", "pandas", "catboost"}
        if not imported.issubset(allowed):
            raise ValueError(
                f"Unexpected notebook imports: {sorted(imported - allowed)}"
            )
        for forbidden in [
            "avito_retrieval",
            "reference_solution",
            "avito contest",
            "sys.path",
            "subprocess",
            "requests",
            "http://",
            "https://",
        ]:
            if forbidden in source_code:
                raise ValueError(
                    f"Forbidden external dependency or source reference: {forbidden}"
                )
        for cell in code_cells:
            cell.outputs = []
            cell.execution_count = None
        nbformat.write(notebook, PACKAGE / "solution.ipynb")
        IPYTHON.mkdir()
        specification = JUPYTER / "kernels/avito_verify"
        specification.mkdir(parents=True)
        kernel_spec = {
            "argv": [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            "display_name": "Avito independent local verification",
            "language": "python",
        }
        (specification / "kernel.json").write_text(
            json.dumps(kernel_spec), encoding="utf-8"
        )
        os.environ["IPYTHONDIR"] = str(IPYTHON)
        os.environ["JUPYTER_PATH"] = str(JUPYTER)
        os.environ["PYTHONPATH"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        environment = os.environ.copy()
        report.update(
            {
                "source_notebook_sha256": source_hash,
                "source_config_sha256": config_hash,
                "code_cells_sha256": original_code_hash,
                "total_cells": len(notebook.cells),
                "code_cells": len(code_cells),
                "original_cell_outputs_cleared": True,
                "ready_answer_present_before_execution": False,
                "copied_input_hashes_passed": True,
                "copied_inputs": input_records,
                "static_imports": sorted(imported),
                "main_project_imports_or_paths_in_code": False,
                "kernel_spec": kernel_spec,
                "execution_environment": {
                    "PYTHONPATH": "",
                    "IPYTHONDIR": str(IPYTHON),
                    "JUPYTER_PATH": str(JUPYTER),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
                "permissions_scope": "Local Jupyter kernel requires loopback sockets. No external network or API is required by notebook code.",
            }
        )
        save_report()
        print("Executing clean standalone notebook in:", PACKAGE, flush=True)
        began = time.monotonic()
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            startup_timeout=60,
            kernel_name="avito_verify",
            resources={"metadata": {"path": str(PACKAGE)}},
            allow_errors=False,
        )
        executed = client.execute(cwd=str(PACKAGE), env=environment)
        execution_seconds = time.monotonic() - began
        nbformat.write(executed, WORK / "solution.executed.ipynb")
        executed_code = "\n\n".join(
            cell.source for cell in executed.cells if cell.cell_type == "code"
        )
        assert hashlib.sha256(executed_code.encode()).hexdigest() == original_code_hash
        executed_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
        assert len(executed_cells) == 5
        assert all(cell.execution_count is not None for cell in executed_cells)
        errors = [
            output
            for cell in executed_cells
            for output in cell.outputs
            if output.output_type == "error"
        ]
        assert not errors
        answer = PACKAGE / "answer.csv"
        generated_hash = sha256(answer)
        if generated_hash != config["answer_sha256"]:
            raise ValueError("Generated CSV does not match the frozen expected hash")
        with answer.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            records = list(reader)
        assert columns == ["query_id", "answer"]
        counts = [len(record["answer"].split()) for record in records]
        assert len(records) == 2452 and min(counts) == max(counts) == 50
        assert sha256(SOURCE / "solution.ipynb") == source_hash
        assert sha256(SOURCE / "config.json") == config_hash
        assert all(
            sha256(SOURCE / relative) == record["sha256"]
            for relative, record in input_records.items()
        )
        report.update(
            {
                "status": "passed",
                "execution_seconds": round(execution_seconds, 3),
                "total_seconds": round(time.monotonic() - START, 3),
                "expected_csv_sha256": config["answer_sha256"],
                "generated_csv_sha256": generated_hash,
                "exact_csv_hash_match": True,
                "generated_csv": str(answer),
                "query_rows": len(records),
                "candidates_min": min(counts),
                "candidates_max": max(counts),
                "executed_code_cells": len(executed_cells),
                "executed_notebook": str(WORK / "solution.executed.ipynb"),
                "source_notebook_unchanged": True,
                "source_config_and_inputs_unchanged": True,
                "ready_answer_used_as_input": False,
                "ready_answer_evidence": "No answer.csv was copied or present before execution. The only CSV read in inspected code is answer.csv.part immediately after generation.",
                "main_project_dependency_evidence": "Copied mini directory ran alone with an empty PYTHONPATH and a new IPython directory. Notebook imports only the recorded standard and installed numerical libraries.",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        save_report()
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "execution_seconds": report["execution_seconds"],
                    "csv_sha256": generated_hash,
                    "report": str(REPORT_PATH),
                    "temporary_directory": str(WORK),
                },
                indent=2,
            ),
            flush=True,
        )
    except BaseException as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "total_seconds": round(time.monotonic() - START, 3),
            }
        )
        save_report()
        raise


if __name__ == "__main__":
    run(parse_arguments())
