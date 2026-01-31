"""Shared fixtures for persistence tests."""

import pytest
from pathlib import Path
from t4dm.storage.t4dx.engine import T4DXEngine


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "t4dx_data"


@pytest.fixture
def engine(data_dir):
    eng = T4DXEngine(data_dir)
    eng.startup()
    yield eng
    if eng._started:
        eng.shutdown()
