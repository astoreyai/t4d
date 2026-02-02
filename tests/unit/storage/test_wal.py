"""Tests for T4DX WAL."""

from pathlib import Path

import pytest

from t4dm.storage.t4dx.wal import OpType, WAL, WALWriter


class TestWAL:
    def test_append_and_replay(self, tmp_path: Path):
        wal = WAL(tmp_path / "test.wal")
        wal.open()
        wal.append(OpType.INSERT, {"item": {"id": "abc"}})
        wal.append(OpType.DELETE, {"item_id": "abc"})
        wal.close()

        entries = wal.replay()
        assert len(entries) == 2
        assert entries[0]["op"] == "INSERT"
        assert entries[1]["op"] == "DELETE"

    def test_truncate(self, tmp_path: Path):
        wal = WAL(tmp_path / "test.wal")
        wal.open()
        wal.append(OpType.INSERT, {"item": {"id": "abc"}})
        wal.truncate()

        entries = wal.replay()
        assert len(entries) == 0

    def test_replay_empty(self, tmp_path: Path):
        wal = WAL(tmp_path / "nonexistent.wal")
        assert wal.replay() == []

    def test_replay_skips_corrupt(self, tmp_path: Path):
        p = tmp_path / "corrupt.wal"
        p.write_text('{"op":"INSERT","x":1}\nNOT JSON\n{"op":"DELETE","y":2}\n')
        wal = WAL(p)
        entries = wal.replay()
        assert len(entries) == 2

    def test_size(self, tmp_path: Path):
        wal = WAL(tmp_path / "test.wal")
        assert wal.size == 0
        wal.open()
        wal.append(OpType.INSERT, {"item": {}})
        wal.close()
        assert wal.size > 0

    def test_auto_open_on_append(self, tmp_path: Path):
        wal = WAL(tmp_path / "test.wal")
        wal.append(OpType.INSERT, {"item": {}})
        wal.close()
        assert len(wal.replay()) == 1

    def test_batch_append(self, tmp_path: Path):
        wal = WAL(tmp_path / "test.wal")
        wal.open()
        entries = [
            {"op": "INSERT", "item": {"id": "a"}},
            {"op": "INSERT", "item": {"id": "b"}},
            {"op": "DELETE", "item_id": "a"},
        ]
        wal.batch_append(entries)
        wal.close()

        replayed = wal.replay()
        assert len(replayed) == 3
        assert replayed[0]["op"] == "INSERT"
        assert replayed[2]["op"] == "DELETE"

    def test_batch_append_empty(self, tmp_path: Path):
        wal = WAL(tmp_path / "test.wal")
        wal.open()
        wal.batch_append([])
        wal.close()
        assert len(wal.replay()) == 0


class TestWALWriter:
    def test_context_manager_writes_on_exit(self, tmp_path: Path):
        wal = WAL(tmp_path / "test.wal")
        wal.open()
        with WALWriter(wal) as w:
            w.append(OpType.INSERT, {"item": {"id": "x"}})
            w.append(OpType.DELETE, {"item_id": "x"})
        wal.close()

        entries = wal.replay()
        assert len(entries) == 2
        assert entries[0]["op"] == "INSERT"
        assert entries[1]["op"] == "DELETE"

    def test_context_manager_no_write_on_exception(self, tmp_path: Path):
        wal = WAL(tmp_path / "test.wal")
        wal.open()
        try:
            with WALWriter(wal) as w:
                w.append(OpType.INSERT, {"item": {"id": "x"}})
                raise ValueError("test error")
        except ValueError:
            pass
        wal.close()

        entries = wal.replay()
        assert len(entries) == 0

    def test_context_manager_empty(self, tmp_path: Path):
        wal = WAL(tmp_path / "test.wal")
        wal.open()
        with WALWriter(wal) as w:
            pass  # no entries
        wal.close()
        assert len(wal.replay()) == 0
