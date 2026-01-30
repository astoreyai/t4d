"""
Tests for World Weaver CLI.

Tests the typer-based CLI commands including store, recall, consolidate,
status, serve, version, and config.
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile

import pytest
from typer.testing import CliRunner

from ww.cli.main import app


runner = CliRunner()


class TestVersionCommand:
    """Tests for version command."""

    def test_version_shows_version(self):
        """Version command shows version number."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "World Weaver" in result.stdout
        assert "v" in result.stdout


class TestStatusCommand:
    """Tests for status command."""

    def test_status_shows_configuration(self):
        """Status command shows configuration table."""
        with patch("ww.core.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                environment="test",
                qdrant_host="localhost",
                neo4j_uri="bolt://localhost:7687",
                embedding_model="bge-m3",
            )
            with patch("ww.core.services.get_services", new=AsyncMock()):
                result = runner.invoke(app, ["status"])
                assert result.exit_code == 0
                assert "World Weaver Status" in result.stdout


class TestConfigCommand:
    """Tests for config command."""

    def test_config_init_creates_file(self):
        """Config init creates default configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            result = runner.invoke(app, ["config", "--init", "--path", str(config_path)])
            assert result.exit_code == 0
            assert config_path.exists()
            content = config_path.read_text()
            assert "session_id" in content
            assert "qdrant_host" in content

    def test_config_show_displays_settings(self):
        """Config show displays current settings."""
        with patch("ww.core.config.get_settings") as mock_settings:
            mock_obj = MagicMock()
            mock_obj.__fields__ = {"environment": None, "qdrant_host": None}
            mock_obj.environment = "test"
            mock_obj.qdrant_host = "localhost"
            mock_settings.return_value = mock_obj

            result = runner.invoke(app, ["config", "--show"])
            assert result.exit_code == 0
            assert "Configuration" in result.stdout


class TestStoreCommand:
    """Tests for store command."""

    @patch("ww.core.services.get_services")
    def test_store_episodic(self, mock_services):
        """Store command creates episodic memory."""
        mock_episodic = MagicMock()
        mock_episodic.add_episode = AsyncMock(
            return_value=MagicMock(id="test-uuid")
        )
        mock_services.return_value = (mock_episodic, MagicMock(), MagicMock())

        result = runner.invoke(
            app,
            ["store", "Test content", "--type", "episodic", "--importance", "0.8"],
        )
        assert result.exit_code == 0
        assert "Stored episode" in result.stdout

    @patch("ww.core.services.get_services")
    def test_store_semantic(self, mock_services):
        """Store command creates semantic entity."""
        mock_semantic = MagicMock()
        mock_semantic.add_entity = AsyncMock(
            return_value=MagicMock(id="test-uuid")
        )
        mock_services.return_value = (MagicMock(), mock_semantic, MagicMock())

        result = runner.invoke(
            app,
            ["store", "Test concept", "--type", "semantic"],
        )
        assert result.exit_code == 0
        assert "Stored entity" in result.stdout

    @patch("ww.core.services.get_services")
    def test_store_procedural(self, mock_services):
        """Store command creates procedural skill."""
        mock_procedural = MagicMock()
        mock_procedural.add_skill = AsyncMock(
            return_value=MagicMock(id="test-uuid")
        )
        mock_services.return_value = (MagicMock(), MagicMock(), mock_procedural)

        result = runner.invoke(
            app,
            ["store", "Test skill", "--type", "procedural", "--tags", "test-skill"],
        )
        assert result.exit_code == 0
        assert "Stored skill" in result.stdout

    def test_store_with_tags(self):
        """Store command parses comma-separated tags."""
        with patch("ww.core.services.get_services") as mock_services:
            mock_episodic = MagicMock()
            mock_episodic.add_episode = AsyncMock(
                return_value=MagicMock(id="test-uuid")
            )
            mock_services.return_value = (mock_episodic, MagicMock(), MagicMock())

            result = runner.invoke(
                app,
                ["store", "Tagged content", "--tags", "tag1,tag2,tag3"],
            )
            assert result.exit_code == 0

    def test_store_with_metadata(self):
        """Store command parses JSON metadata."""
        with patch("ww.core.services.get_services") as mock_services:
            mock_episodic = MagicMock()
            mock_episodic.add_episode = AsyncMock(
                return_value=MagicMock(id="test-uuid")
            )
            mock_services.return_value = (mock_episodic, MagicMock(), MagicMock())

            result = runner.invoke(
                app,
                ["store", "Content with metadata", "--metadata", '{"key": "value"}'],
            )
            assert result.exit_code == 0

    def test_store_invalid_type(self):
        """Store command rejects invalid memory type."""
        with patch("ww.core.services.get_services") as mock_services:
            mock_services.return_value = (MagicMock(), MagicMock(), MagicMock())

            result = runner.invoke(
                app,
                ["store", "Content", "--type", "invalid"],
            )
            assert result.exit_code == 1


class TestRecallCommand:
    """Tests for recall command."""

    @patch("ww.core.services.get_services")
    def test_recall_table_format(self, mock_services):
        """Recall command shows results in table format."""
        mock_episodic = MagicMock()
        mock_result = MagicMock()
        mock_result.item = MagicMock(
            id="test-uuid",
            content="Test content",
            timestamp="2024-01-01T00:00:00",
        )
        mock_result.score = 0.95
        mock_episodic.recall_similar = AsyncMock(return_value=[mock_result])

        mock_semantic = MagicMock()
        mock_semantic.search_similar = AsyncMock(return_value=[])

        mock_procedural = MagicMock()
        mock_procedural.find_relevant_skills = AsyncMock(return_value=[])

        mock_services.return_value = (mock_episodic, mock_semantic, mock_procedural)

        result = runner.invoke(app, ["recall", "test query"])
        assert result.exit_code == 0
        assert "Recall Results" in result.stdout

    @patch("ww.core.services.get_services")
    def test_recall_json_format(self, mock_services):
        """Recall command shows results in JSON format."""
        mock_episodic = MagicMock()
        mock_result = MagicMock()
        mock_result.item = MagicMock(
            id="test-uuid",
            content="Test content",
            timestamp="2024-01-01T00:00:00",
        )
        mock_result.score = 0.95
        mock_episodic.recall_similar = AsyncMock(return_value=[mock_result])

        mock_semantic = MagicMock()
        mock_semantic.search_similar = AsyncMock(return_value=[])

        mock_procedural = MagicMock()
        mock_procedural.find_relevant_skills = AsyncMock(return_value=[])

        mock_services.return_value = (mock_episodic, mock_semantic, mock_procedural)

        result = runner.invoke(app, ["recall", "test query", "--format", "json"])
        assert result.exit_code == 0
        # JSON output should be parseable
        assert "episodic" in result.stdout

    @patch("ww.core.services.get_services")
    def test_recall_specific_type(self, mock_services):
        """Recall command filters by memory type."""
        mock_episodic = MagicMock()
        mock_episodic.recall_similar = AsyncMock(return_value=[])

        mock_services.return_value = (mock_episodic, MagicMock(), MagicMock())

        result = runner.invoke(
            app,
            ["recall", "test query", "--type", "episodic"],
        )
        assert result.exit_code == 0
        mock_episodic.recall_similar.assert_called_once()

    @patch("ww.core.services.get_services")
    def test_recall_with_limit(self, mock_services):
        """Recall command respects limit parameter."""
        mock_episodic = MagicMock()
        mock_episodic.recall_similar = AsyncMock(return_value=[])

        mock_semantic = MagicMock()
        mock_semantic.search_similar = AsyncMock(return_value=[])

        mock_procedural = MagicMock()
        mock_procedural.find_relevant_skills = AsyncMock(return_value=[])

        mock_services.return_value = (mock_episodic, mock_semantic, mock_procedural)

        result = runner.invoke(app, ["recall", "test query", "-k", "10"])
        assert result.exit_code == 0
        mock_episodic.recall_similar.assert_called_with("test query", limit=10)


class TestConsolidateCommand:
    """Tests for consolidate command."""

    def test_consolidate_dry_run(self):
        """Consolidate dry run shows pending stats."""
        with patch("ww.consolidation.service.get_consolidation_service") as mock_service:
            mock_svc = MagicMock()
            mock_svc.get_scheduler_stats = MagicMock(return_value={"pending_count": 5})
            mock_service.return_value = mock_svc

            result = runner.invoke(app, ["consolidate", "--dry-run"])
            assert result.exit_code == 0
            assert "Pending" in result.stdout or "pending" in result.stdout.lower()

    def test_consolidate_incremental(self):
        """Consolidate runs light mode by default."""
        with patch("ww.consolidation.service.get_consolidation_service") as mock_service:
            mock_svc = MagicMock()
            mock_svc.consolidate = AsyncMock(
                return_value={
                    "episodes_processed": 10,
                    "entities_created": 2,
                    "procedures_consolidated": 1,
                }
            )
            mock_service.return_value = mock_svc

            result = runner.invoke(app, ["consolidate"])
            assert result.exit_code == 0
            assert "complete" in result.stdout.lower()
            mock_svc.consolidate.assert_called_once_with(mode="light")

    def test_consolidate_full(self):
        """Consolidate --full runs deep mode."""
        with patch("ww.consolidation.service.get_consolidation_service") as mock_service:
            mock_svc = MagicMock()
            mock_svc.consolidate = AsyncMock(
                return_value={
                    "episodes_processed": 100,
                    "entities_created": 20,
                    "procedures_consolidated": 5,
                }
            )
            mock_service.return_value = mock_svc

            result = runner.invoke(app, ["consolidate", "--full"])
            assert result.exit_code == 0
            mock_svc.consolidate.assert_called_once_with(mode="deep")


class TestEpisodicSubcommands:
    """Tests for episodic sub-commands."""

    def test_episodic_add_calls_store(self):
        """Episodic add delegates to store command."""
        # The episodic add command delegates to store() internally.
        # We test that the command accepts the correct arguments.
        result = runner.invoke(
            app,
            ["episodic", "add", "--help"],
        )
        assert result.exit_code == 0
        assert "valence" in result.stdout.lower()
        assert "tags" in result.stdout.lower()

    @patch("ww.core.services.get_services")
    def test_episodic_search(self, mock_services):
        """Episodic search filters to episodic type."""
        mock_episodic = MagicMock()
        mock_episodic.recall_similar = AsyncMock(return_value=[])
        mock_services.return_value = (mock_episodic, MagicMock(), MagicMock())

        result = runner.invoke(app, ["episodic", "search", "test query"])
        assert result.exit_code == 0

    @patch("ww.core.services.get_services")
    def test_episodic_recent(self, mock_services):
        """Episodic recent shows recent episodes."""
        mock_episodic = MagicMock()
        mock_episode = MagicMock()
        mock_episode.id = "test-uuid"
        mock_episode.content = "Test content"
        mock_episode.timestamp = "2024-01-01T00:00:00"
        mock_episode.valence = 0.5
        mock_episodic.get_recent_episodes = AsyncMock(return_value=[mock_episode])

        mock_services.return_value = (mock_episodic, MagicMock(), MagicMock())

        result = runner.invoke(app, ["episodic", "recent"])
        assert result.exit_code == 0
        assert "Recent Episodes" in result.stdout


class TestSemanticSubcommands:
    """Tests for semantic sub-commands."""

    def test_semantic_add_accepts_args(self):
        """Semantic add accepts name and description."""
        result = runner.invoke(
            app,
            ["semantic", "add", "--help"],
        )
        assert result.exit_code == 0
        assert "desc" in result.stdout.lower() or "description" in result.stdout.lower()

    @patch("ww.core.services.get_services")
    def test_semantic_search(self, mock_services):
        """Semantic search filters to semantic type."""
        mock_semantic = MagicMock()
        mock_semantic.search_similar = AsyncMock(return_value=[])
        mock_services.return_value = (MagicMock(), mock_semantic, MagicMock())

        result = runner.invoke(app, ["semantic", "search", "test query"])
        assert result.exit_code == 0


class TestProceduralSubcommands:
    """Tests for procedural sub-commands."""

    def test_procedural_add_accepts_args(self):
        """Procedural add accepts name and description."""
        result = runner.invoke(
            app,
            ["procedural", "add", "--help"],
        )
        assert result.exit_code == 0
        assert "desc" in result.stdout.lower() or "description" in result.stdout.lower()

    @patch("ww.core.services.get_services")
    def test_procedural_search(self, mock_services):
        """Procedural search filters to procedural type."""
        mock_procedural = MagicMock()
        mock_procedural.find_relevant_skills = AsyncMock(return_value=[])
        mock_services.return_value = (MagicMock(), MagicMock(), mock_procedural)

        result = runner.invoke(app, ["procedural", "search", "test query"])
        assert result.exit_code == 0


class TestServeCommand:
    """Tests for serve command."""

    @patch("uvicorn.run")
    def test_serve_starts_server(self, mock_uvicorn_run):
        """Serve command starts uvicorn server."""
        runner.invoke(app, ["serve", "--port", "8888"])
        mock_uvicorn_run.assert_called_once()
        call_kwargs = mock_uvicorn_run.call_args
        assert call_kwargs[1]["port"] == 8888

    @patch("uvicorn.run")
    def test_serve_with_reload(self, mock_uvicorn_run):
        """Serve command passes reload flag."""
        runner.invoke(app, ["serve", "--reload"])
        call_kwargs = mock_uvicorn_run.call_args
        assert call_kwargs[1]["reload"] is True


class TestSessionManagement:
    """Tests for session ID handling."""

    def test_session_from_environment(self):
        """CLI uses WW_SESSION_ID from environment."""
        os.environ["WW_SESSION_ID"] = "test-session-123"
        try:
            from ww.cli.main import get_session_id
            assert get_session_id() == "test-session-123"
        finally:
            del os.environ["WW_SESSION_ID"]

    def test_default_session_id(self):
        """CLI uses default session when not set."""
        if "WW_SESSION_ID" in os.environ:
            del os.environ["WW_SESSION_ID"]

        from ww.cli.main import get_session_id
        assert get_session_id() == "cli-session"


class TestErrorHandling:
    """Tests for error handling in CLI."""

    @patch("ww.core.services.get_services")
    def test_store_handles_errors(self, mock_services):
        """Store command handles service errors gracefully."""
        mock_services.side_effect = Exception("Connection failed")

        result = runner.invoke(app, ["store", "Test content"])
        assert result.exit_code == 1
        assert "Error" in result.stdout

    @patch("ww.core.services.get_services")
    def test_recall_handles_errors(self, mock_services):
        """Recall command handles service errors gracefully."""
        mock_services.side_effect = Exception("Connection failed")

        result = runner.invoke(app, ["recall", "test query"])
        assert result.exit_code == 1
        assert "Error" in result.stdout


class TestHelpOutput:
    """Tests for help text."""

    def test_main_help(self):
        """Main app shows help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "World Weaver" in result.stdout
        assert "store" in result.stdout
        assert "recall" in result.stdout

    def test_store_help(self):
        """Store command shows help."""
        result = runner.invoke(app, ["store", "--help"])
        assert result.exit_code == 0
        assert "content" in result.stdout.lower()

    def test_recall_help(self):
        """Recall command shows help."""
        result = runner.invoke(app, ["recall", "--help"])
        assert result.exit_code == 0
        assert "query" in result.stdout.lower()

    def test_episodic_help(self):
        """Episodic subcommands show help."""
        result = runner.invoke(app, ["episodic", "--help"])
        assert result.exit_code == 0
        assert "add" in result.stdout
        assert "search" in result.stdout
