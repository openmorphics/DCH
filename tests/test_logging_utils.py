import csv
import json
import sys
import types
import importlib
from pathlib import Path
import pytest

from dch_pipeline.logging_utils import CSVLogger, JSONLLogger, TensorBoardLogger, ExperimentLogger


def test_csv_writes_header_once(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    fieldnames = ["epoch", "acc"]

    logger1 = CSVLogger(csv_path, fieldnames=fieldnames)
    logger1.write_row({"epoch": 1, "acc": 0.5})
    logger1.close()

    logger2 = CSVLogger(csv_path, fieldnames=fieldnames)
    logger2.write_row({"epoch": 2, "acc": 0.6})
    logger2.close()

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert header == fieldnames
    assert len(rows) == 2
    assert rows[0] == ["1", "0.5"]
    assert rows[1] == ["2", "0.6"]


def test_csv_rejects_extra_keys(tmp_path):
    csv_path = tmp_path / "metrics_extra.csv"
    fieldnames = ["epoch", "acc"]
    logger = CSVLogger(csv_path, fieldnames=fieldnames, allow_extra=False)
    logger.write_row({"epoch": 1, "acc": 0.5})  # ok
    with pytest.raises(ValueError):
        logger.write_row({"epoch": 2, "acc": 0.6, "extra": True})
    logger.close()


def test_jsonl_appends_lines(tmp_path):
    jsonl_path = tmp_path / "events.jsonl"
    logger = JSONLLogger(jsonl_path, auto_timestamp=True)
    logger.log({"a": 1})
    logger.log({"a": 2})
    logger.close()

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["a"] == 1 and "ts" in rec1
    assert rec2["a"] == 2 and "ts" in rec2


def test_tb_noop_when_missing(monkeypatch, tmp_path):
    # Ensure both backends appear missing
    # simulate missing backends via import_module patch only
    orig_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name in ("torch.utils.tensorboard", "tensorboardX"):
            raise ModuleNotFoundError("mocked missing")
        return orig_import_module(name, package)

    monkeypatch.setattr("dch_pipeline.logging_utils.import_module", fake_import_module)

    log_dir = tmp_path / "tb_missing"
    logger = TensorBoardLogger(log_dir)
    assert logger.is_active is False

    logger.log_scalar("x", 1.0, 1)
    logger.log_scalars("group", {"a": 0.1, "b": 0.2}, 2)
    logger.log_histogram("hist", [1, 2, 3], 3)
    logger.flush()
    logger.close()

    # No directory should be created
    assert not log_dir.exists()


def test_tb_uses_mocked_backend(monkeypatch, tmp_path):
    calls = {"init": 0, "scalars": [], "hist": [], "closed": False}

    class StubSummaryWriter:
        def __init__(self, log_dir=None):
            calls["init"] += 1
            self.log_dir = log_dir
            p = Path(log_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "MARKER").write_text("ok", encoding="utf-8")

        def add_scalar(self, tag, value, step):
            calls["scalars"].append((tag, value, step))

        def add_histogram(self, tag, values, step):
            calls["hist"].append((tag, list(values) if isinstance(values, (list, tuple)) else values, step))

        def flush(self):
            pass

        def close(self):
            calls["closed"] = True

    stub_module = types.SimpleNamespace(SummaryWriter=StubSummaryWriter)
    orig_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "torch.utils.tensorboard":
            return stub_module
        if name == "tensorboardX":
            return stub_module
        return orig_import_module(name, package)

    monkeypatch.setattr("dch_pipeline.logging_utils.import_module", fake_import_module)

    log_dir = tmp_path / "tb_ok"
    logger = TensorBoardLogger(log_dir)
    assert logger.is_active is True

    logger.log_scalar("loss", 0.1, 1)
    logger.log_scalars("metrics", {"acc": 0.9, "f1": 0.8}, 1)
    logger.log_histogram("weights", [1, 2, 3, 4], 1)
    logger.flush()
    logger.close()

    # Verify stub interactions and side-effects
    assert calls["init"] == 1
    assert (log_dir / "MARKER").exists()
    # Expect 1 + 2 scalar calls
    assert len(calls["scalars"]) == 3
    # One histogram call
    assert len(calls["hist"]) == 1
    assert calls["closed"] is True


def test_experiment_logger_forwards_calls(tmp_path):
    csv_path = tmp_path / "exp.csv"
    jsonl_path = tmp_path / "exp.jsonl"

    exp = ExperimentLogger(csv_path=csv_path, csv_fieldnames=["epoch", "acc"], jsonl_path=jsonl_path)
    exp.log_csv({"epoch": 1, "acc": 0.5})
    exp.log_csv_rows([{"epoch": 2, "acc": 0.6}])
    exp.log_jsonl({"event": "start"})
    exp.flush()
    exp.close()

    # CSV verification
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header == ["epoch", "acc"]
    assert rows == [["1", "0.5"], ["2", "0.6"]]

    # JSONL verification
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l) for l in lines] == [{"event": "start"}]