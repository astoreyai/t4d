"""Tests for κ-based query policies."""

import pytest
from datetime import datetime, timedelta

from t4dm.core.query_policies import (
    EpisodicPolicy,
    ProceduralPolicy,
    QueryFilters,
    SemanticPolicy,
    select_policy,
)


class TestQueryFilters:
    def test_defaults_all_none(self):
        f = QueryFilters()
        assert f.kappa_min is None
        assert f.kappa_max is None
        assert f.time_min is None
        assert f.item_type is None


class TestEpisodicPolicy:
    def test_default_filters(self):
        now = datetime(2026, 1, 30, 12, 0, 0)
        f = EpisodicPolicy().filters(now=now)
        assert f.kappa_min == 0.0
        assert f.kappa_max == 0.3
        assert f.item_type == "episodic"
        assert f.time_min == now - timedelta(hours=24)

    def test_custom_window(self):
        f = EpisodicPolicy(hours=1.0, kappa_max=0.5).filters()
        assert f.kappa_max == 0.5


class TestSemanticPolicy:
    def test_default_filters(self):
        f = SemanticPolicy().filters()
        assert f.kappa_min == 0.7
        assert f.kappa_max == 1.0
        assert f.item_type == "semantic"
        assert f.time_min is None


class TestProceduralPolicy:
    def test_default_filters(self):
        f = ProceduralPolicy().filters()
        assert f.item_type == "procedural"


class TestSelectPolicy:
    def test_known_types(self):
        assert isinstance(select_policy("episodic"), EpisodicPolicy)
        assert isinstance(select_policy("semantic"), SemanticPolicy)
        assert isinstance(select_policy("procedural"), ProceduralPolicy)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            select_policy("unknown")
