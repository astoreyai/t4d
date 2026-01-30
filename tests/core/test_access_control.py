"""Tests for access control system."""

import pytest

from ww.core.access_control import (
    API_TOKEN,
    AccessDenied,
    CallerToken,
    CONSOLIDATION_TOKEN,
    DEBUG_TOKEN,
    HIPPOCAMPUS_TOKEN,
    VTA_TOKEN,
    require_capability,
)


class TestCallerToken:
    def test_has_capability(self):
        token = CallerToken("test", "internal", frozenset({"read", "write"}))
        assert token.has_capability("read")
        assert token.has_capability("write")
        assert not token.has_capability("admin")

    def test_token_immutable(self):
        token = CallerToken("test", "internal", frozenset({"read"}))
        with pytest.raises(AttributeError):
            token.module = "hacked"


class TestRequireCapability:
    def test_none_token_denied(self):
        with pytest.raises(AccessDenied):
            require_capability(None, "read")

    def test_missing_capability_denied(self):
        token = CallerToken("test", "external", frozenset({"read"}))
        with pytest.raises(AccessDenied) as exc_info:
            require_capability(token, "write_ca3")
        assert "write_ca3" in str(exc_info.value)

    def test_valid_capability_passes(self):
        token = CallerToken("test", "internal", frozenset({"write_ca3"}))
        require_capability(token, "write_ca3")  # Should not raise

    def test_invalid_token_type(self):
        with pytest.raises(AccessDenied):
            require_capability("not_a_token", "read")


class TestPrebuiltTokens:
    def test_hippocampus_can_write_ca3(self):
        require_capability(HIPPOCAMPUS_TOKEN, "write_ca3")

    def test_hippocampus_can_set_neuromod(self):
        require_capability(HIPPOCAMPUS_TOKEN, "set_neuromod")

    def test_api_cannot_write_ca3(self):
        with pytest.raises(AccessDenied):
            require_capability(API_TOKEN, "write_ca3")

    def test_api_cannot_set_neuromod(self):
        with pytest.raises(AccessDenied):
            require_capability(API_TOKEN, "set_neuromod")

    def test_api_can_read(self):
        require_capability(API_TOKEN, "read")

    def test_api_can_store_episodic(self):
        require_capability(API_TOKEN, "store_episodic")

    def test_consolidation_can_trigger_swr(self):
        require_capability(CONSOLIDATION_TOKEN, "trigger_swr")

    def test_consolidation_can_trigger_replay(self):
        require_capability(CONSOLIDATION_TOKEN, "trigger_replay")

    def test_vta_can_submit_reward(self):
        require_capability(VTA_TOKEN, "submit_reward")

    def test_debug_has_all_capabilities(self):
        for cap in ["write_ca3", "set_neuromod", "trigger_swr", "debug"]:
            require_capability(DEBUG_TOKEN, cap)

    def test_api_cannot_trigger_consolidation(self):
        with pytest.raises(AccessDenied):
            require_capability(API_TOKEN, "trigger_consolidation")

    def test_api_cannot_set_sleep_state(self):
        with pytest.raises(AccessDenied):
            require_capability(API_TOKEN, "set_sleep_state")
