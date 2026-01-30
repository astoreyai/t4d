"""Pytest configuration for embedding tests."""
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def patch_embedding_settings(patch_settings):
    """Auto-patch settings for all embedding tests."""
    # patch_settings already does the patching, we just make it autouse
    return patch_settings
