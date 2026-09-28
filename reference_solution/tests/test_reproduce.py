import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import reproduce


def test_relative_user_paths_are_resolved_before_changing_to_project(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    caller = tmp_path / "caller"
    project.mkdir()
    caller.mkdir()
    expected = b"query_id,answer\n"
    config = {
        "benchmark_features": "artifacts/features.parquet",
        "models": [{"path": "artifacts/final.cbm"}],
        "weights": [1.0],
        "answer_sha256": hashlib.sha256(expected).hexdigest(),
    }
    (caller / "chosen.json").write_text(json.dumps(config))
    monkeypatch.setattr(reproduce, "ROOT", project)
    monkeypatch.chdir(caller)
    monkeypatch.setenv("PYTHONPATH", "src")
    monkeypatch.setattr(
        sys, "argv", ["reproduce", "--config", "chosen.json", "--output", "answer.csv"]
    )

    def fake_prediction(script, *arguments):
        assert script == "predict.py"
        path = Path(arguments[arguments.index("--output") + 1])
        assert path == caller / "answer.csv"
        path.write_bytes(expected)

    monkeypatch.setattr(reproduce, "run", fake_prediction)
    reproduce.main()
    assert (caller / "answer.csv").read_bytes() == expected
    assert not (project / "answer.csv").exists()


def test_quick_reproduction_rejects_changed_input_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce, "ROOT", tmp_path)
    original = b"frozen input"
    (tmp_path / "model.cbm").write_bytes(b"changed model")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        reproduce.verify_inputs(
            {
                "input_sha256": {
                    "model.cbm": hashlib.sha256(original).hexdigest(),
                }
            }
        )
