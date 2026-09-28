"""Выполнить все ячейки мини-ноутбука в новом ядре Python."""

from pathlib import Path
import json
import time

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "mini/solution.ipynb"
notebook = nbformat.read(path, as_version=4)
start = time.monotonic()
NotebookClient(
    notebook,
    timeout=120,
    kernel_name="python3",
    resources={"metadata": {"path": str(path.parent)}},
).execute()
nbformat.write(notebook, path)
print(json.dumps({"executed": True, "seconds": round(time.monotonic() - start, 3)}))
