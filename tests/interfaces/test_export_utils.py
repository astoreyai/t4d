"""Tests for export utilities module."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestExportUtility:
    """Tests for ExportUtility class."""

    @pytest.fixture
    def mock_stores(self):
        """Create mock storage backends."""
        with patch("t4dm.interfaces.export_utils.EpisodicMemory") as mock_ep, \
             patch("t4dm.interfaces.export_utils.SemanticMemory") as mock_sem, \
             patch("t4dm.interfaces.export_utils.ProceduralMemory") as mock_proc, \
             patch("t4dm.interfaces.export_utils.get_vector_store") as mock_qdrant:

            # Configure mocks
            mock_ep_instance = MagicMock()
            mock_ep_instance.initialize = AsyncMock()
            mock_ep.return_value = mock_ep_instance

            mock_sem_instance = MagicMock()
            mock_sem_instance.initialize = AsyncMock()
            mock_sem.return_value = mock_sem_instance

            mock_proc_instance = MagicMock()
            mock_proc_instance.initialize = AsyncMock()
            mock_proc.return_value = mock_proc_instance

            # Configure vector store
            mock_vs = MagicMock()
            mock_vs.episodes_collection = "episodes"
            mock_vs.entities_collection = "entities"
            mock_vs.procedures_collection = "procedures"
            mock_vs.scroll = AsyncMock(return_value=([], None))
            mock_qdrant.return_value = mock_vs

            yield {
                "episodic": mock_ep_instance,
                "semantic": mock_sem_instance,
                "procedural": mock_proc_instance,
                "vector_store": mock_vs,
            }

    @pytest.mark.asyncio
    async def test_init_without_rich(self, mock_stores):
        """Test initialization without rich library."""
        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            from t4dm.interfaces.export_utils import ExportUtility

            exporter = ExportUtility(session_id="test")
            assert exporter.session_id == "test"
            assert exporter.console is None

    @pytest.mark.asyncio
    async def test_initialize(self, mock_stores):
        """Test initialization of storage backends."""
        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")
            await exporter.initialize()

            assert exporter._initialized is True
            mock_stores["episodic"].initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_episodes_json(self, mock_stores):
        """Test exporting episodes to JSON."""
        # Setup mock data
        mock_stores["vector_store"].scroll = AsyncMock(return_value=([
            ("id1", {"session_id": "test", "content": "Episode 1", "timestamp": "2024-01-01T12:00:00"}, None),
            ("id2", {"session_id": "test", "content": "Episode 2", "timestamp": "2024-01-02T12:00:00"}, None),
        ], None))

        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "episodes.json"
                count = await exporter.export_episodes_json(str(output_path))

                assert count == 2
                assert output_path.exists()

                with output_path.open() as f:
                    data = json.load(f)
                    assert data["count"] == 2
                    assert len(data["episodes"]) == 2

    @pytest.mark.asyncio
    async def test_export_episodes_csv(self, mock_stores):
        """Test exporting episodes to CSV."""
        mock_stores["vector_store"].scroll = AsyncMock(return_value=([
            ("id1", {"session_id": "test", "content": "Episode 1", "timestamp": "2024-01-01T12:00:00",
                     "outcome": "success", "emotional_valence": 0.8, "access_count": 5,
                     "stability": 2.0, "last_accessed": "2024-01-02T12:00:00"}, None),
        ], None))

        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "episodes.csv"
                count = await exporter.export_episodes_csv(str(output_path))

                assert count == 1
                assert output_path.exists()

                content = output_path.read_text()
                assert "content" in content
                assert "Episode 1" in content

    @pytest.mark.asyncio
    async def test_export_entities_json(self, mock_stores):
        """Test exporting entities to JSON."""
        mock_stores["vector_store"].scroll = AsyncMock(return_value=([
            ("id1", {"name": "Entity 1", "entity_type": "CONCEPT", "summary": "Test entity"}, None),
        ], None))

        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "entities.json"
                count = await exporter.export_entities_json(str(output_path))

                assert count == 1
                assert output_path.exists()

    @pytest.mark.asyncio
    async def test_export_with_limit(self, mock_stores):
        """Test exporting with limit."""
        mock_stores["vector_store"].scroll = AsyncMock(return_value=([
            ("id1", {"content": "Episode 1", "timestamp": "2024-01-01T12:00:00"}, None),
            ("id2", {"content": "Episode 2", "timestamp": "2024-01-01T12:00:00"}, None),
            ("id3", {"content": "Episode 3", "timestamp": "2024-01-01T12:00:00"}, None),
        ], None))

        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "episodes.json"
                count = await exporter.export_episodes_json(str(output_path), limit=2)

                assert count == 2

    @pytest.mark.asyncio
    async def test_escape_xml(self, mock_stores):
        """Test XML escaping function."""
        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility()

            escaped = exporter._escape_xml("<test>&'\"value>")
            assert "&lt;" in escaped
            assert "&gt;" in escaped
            assert "&amp;" in escaped
            assert "&apos;" in escaped
            assert "&quot;" in escaped

    @pytest.mark.asyncio
    async def test_export_skills_json(self, mock_stores):
        """Test exporting skills to JSON."""
        mock_stores["vector_store"].scroll = AsyncMock(return_value=([
            ("id1", {"name": "Skill 1", "domain": "coding", "steps": [], "success_rate": 0.9}, None),
        ], None))

        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "skills.json"
                count = await exporter.export_skills_json(str(output_path))

                assert count == 1
                assert output_path.exists()

    @pytest.mark.asyncio
    async def test_backup_session(self, mock_stores):
        """Test full session backup."""
        mock_stores["vector_store"].scroll = AsyncMock(return_value=([], None))
        mock_stores["semantic"].graph_store = MagicMock()
        mock_stores["semantic"].graph_store.get_relationships = AsyncMock(return_value=[])

        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")
            exporter.semantic = mock_stores["semantic"]

            with tempfile.TemporaryDirectory() as tmpdir:
                results = await exporter.backup_session(tmpdir)

                assert "episodes" in results
                assert "entities" in results
                assert "skills" in results
                assert (Path(tmpdir) / "metadata.json").exists()


class TestExportGraphML:
    """Tests for GraphML export."""

    @pytest.fixture
    def mock_stores_with_graph(self):
        """Create mock stores with graph support."""
        with patch("t4dm.interfaces.export_utils.EpisodicMemory") as mock_ep, \
             patch("t4dm.interfaces.export_utils.SemanticMemory") as mock_sem, \
             patch("t4dm.interfaces.export_utils.ProceduralMemory") as mock_proc, \
             patch("t4dm.interfaces.export_utils.get_vector_store") as mock_qdrant:

            mock_ep_instance = MagicMock()
            mock_ep_instance.initialize = AsyncMock()
            mock_ep.return_value = mock_ep_instance

            mock_sem_instance = MagicMock()
            mock_sem_instance.initialize = AsyncMock()
            mock_sem_instance.graph_store = MagicMock()
            mock_sem_instance.graph_store.get_relationships = AsyncMock(return_value=[
                {"other_id": "id2", "type": "RELATED", "properties": {"weight": 0.8}}
            ])
            mock_sem.return_value = mock_sem_instance

            mock_proc_instance = MagicMock()
            mock_proc_instance.initialize = AsyncMock()
            mock_proc.return_value = mock_proc_instance

            mock_vs = MagicMock()
            mock_vs.episodes_collection = "episodes"
            mock_vs.entities_collection = "entities"
            mock_vs.procedures_collection = "procedures"
            mock_vs.scroll = AsyncMock(return_value=([
                ("id1", {"name": "Entity 1", "entity_type": "CONCEPT", "summary": "Test"}, None),
            ], None))
            mock_qdrant.return_value = mock_vs

            yield {
                "semantic": mock_sem_instance,
                "vector_store": mock_vs,
            }

    @pytest.mark.asyncio
    async def test_export_graph_graphml(self, mock_stores_with_graph):
        """Test exporting to GraphML format."""
        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")
            exporter.semantic = mock_stores_with_graph["semantic"]

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "graph.graphml"
                count = await exporter.export_graph_graphml(str(output_path))

                assert count == 1
                assert output_path.exists()

                content = output_path.read_text()
                assert "graphml" in content
                assert "Entity 1" in content


class TestPathTraversalProtection:
    """SEC-003: Tests for path traversal attack prevention."""

    def test_validate_export_path_traversal_detected(self):
        """Test that path traversal patterns are rejected."""
        from t4dm.interfaces.export_utils import _validate_export_path

        # Various traversal attempts
        with pytest.raises(ValueError, match="Path traversal detected"):
            _validate_export_path("../../../etc/passwd")

        with pytest.raises(ValueError, match="Path traversal detected"):
            _validate_export_path("/home/user/../../../etc/passwd")

        with pytest.raises(ValueError, match="Path traversal detected"):
            _validate_export_path("/etc/shadow")

        with pytest.raises(ValueError, match="Path traversal detected"):
            _validate_export_path("/usr/local/bin/evil")

    def test_validate_export_path_outside_allowed_dirs(self):
        """Test that paths outside allowed directories are rejected."""
        from t4dm.interfaces.export_utils import _validate_export_path

        # Path outside any allowed directory
        with pytest.raises(ValueError, match="not within allowed directories"):
            _validate_export_path("/var/log/syslog")

        with pytest.raises(ValueError, match="not within allowed directories"):
            _validate_export_path("/root/sensitive.txt")

    def test_validate_export_path_valid_tmp(self):
        """Test that valid /tmp paths are accepted."""
        from t4dm.interfaces.export_utils import _validate_export_path

        result = _validate_export_path("/tmp/export.json")
        assert result == Path("/tmp/export.json")

    def test_validate_export_path_valid_home_dirs(self):
        """Test that valid home directory paths are accepted."""
        from t4dm.interfaces.export_utils import _validate_export_path

        home = Path.home()

        # Documents directory - use resolve() to handle symlinks
        result = _validate_export_path(str(home / "Documents" / "export.json"))
        assert result.parent == (home / "Documents").resolve()

        # Downloads directory
        result = _validate_export_path(str(home / "Downloads" / "backup.csv"))
        assert result.parent == (home / "Downloads").resolve()

        # ww_exports directory
        result = _validate_export_path(str(home / "ww_exports" / "session.json"))
        assert result.parent == (home / "ww_exports").resolve()

    def test_validate_export_path_custom_allowed_dirs(self):
        """Test custom allowed directories."""
        from t4dm.interfaces.export_utils import _validate_export_path

        custom_dirs = [Path("/custom/exports")]

        # Valid with custom dirs
        result = _validate_export_path(
            "/custom/exports/file.json",
            allowed_dirs=custom_dirs
        )
        assert result == Path("/custom/exports/file.json")

        # Previously valid default dir now invalid
        with pytest.raises(ValueError, match="not within allowed directories"):
            _validate_export_path("/tmp/file.json", allowed_dirs=custom_dirs)

    @pytest.fixture
    def mock_stores(self):
        """Create mock storage backends for export tests."""
        with patch("t4dm.interfaces.export_utils.EpisodicMemory") as mock_ep, \
             patch("t4dm.interfaces.export_utils.SemanticMemory") as mock_sem, \
             patch("t4dm.interfaces.export_utils.ProceduralMemory") as mock_proc, \
             patch("t4dm.interfaces.export_utils.get_vector_store") as mock_qdrant:

            mock_ep_instance = MagicMock()
            mock_ep_instance.initialize = AsyncMock()
            mock_ep.return_value = mock_ep_instance

            mock_sem_instance = MagicMock()
            mock_sem_instance.initialize = AsyncMock()
            mock_sem_instance.graph_store = MagicMock()
            mock_sem_instance.graph_store.get_relationships = AsyncMock(return_value=[])
            mock_sem.return_value = mock_sem_instance

            mock_proc_instance = MagicMock()
            mock_proc_instance.initialize = AsyncMock()
            mock_proc.return_value = mock_proc_instance

            mock_vs = MagicMock()
            mock_vs.episodes_collection = "episodes"
            mock_vs.entities_collection = "entities"
            mock_vs.procedures_collection = "procedures"
            mock_vs.scroll = AsyncMock(return_value=([], None))
            mock_qdrant.return_value = mock_vs

            yield {
                "semantic": mock_sem_instance,
                "vector_store": mock_vs,
            }

    @pytest.mark.asyncio
    async def test_export_episodes_json_path_traversal(self, mock_stores):
        """Test export_episodes_json rejects path traversal."""
        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")

            with pytest.raises(ValueError, match="Path traversal detected"):
                await exporter.export_episodes_json("../../../etc/passwd")

    @pytest.mark.asyncio
    async def test_export_episodes_csv_path_traversal(self, mock_stores):
        """Test export_episodes_csv rejects path traversal."""
        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")

            with pytest.raises(ValueError, match="Path traversal detected"):
                await exporter.export_episodes_csv("../../../etc/passwd")

    @pytest.mark.asyncio
    async def test_export_entities_json_path_traversal(self, mock_stores):
        """Test export_entities_json rejects path traversal."""
        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")

            with pytest.raises(ValueError, match="Path traversal detected"):
                await exporter.export_entities_json("../../../etc/passwd")

    @pytest.mark.asyncio
    async def test_export_entities_csv_path_traversal(self, mock_stores):
        """Test export_entities_csv rejects path traversal."""
        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")

            with pytest.raises(ValueError, match="Path traversal detected"):
                await exporter.export_entities_csv("../../../etc/passwd")

    @pytest.mark.asyncio
    async def test_export_skills_json_path_traversal(self, mock_stores):
        """Test export_skills_json rejects path traversal."""
        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")

            with pytest.raises(ValueError, match="Path traversal detected"):
                await exporter.export_skills_json("../../../etc/passwd")

    @pytest.mark.asyncio
    async def test_export_graph_graphml_path_traversal(self, mock_stores):
        """Test export_graph_graphml rejects path traversal."""
        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")
            exporter.semantic = mock_stores["semantic"]

            with pytest.raises(ValueError, match="Path traversal detected"):
                await exporter.export_graph_graphml("../../../etc/passwd")

    @pytest.mark.asyncio
    async def test_backup_session_path_traversal(self, mock_stores):
        """Test backup_session rejects path traversal."""
        from t4dm.interfaces.export_utils import ExportUtility

        with patch("t4dm.interfaces.export_utils.RICH_AVAILABLE", False):
            exporter = ExportUtility(session_id="test")
            exporter.semantic = mock_stores["semantic"]

            with pytest.raises(ValueError, match="Path traversal detected"):
                await exporter.backup_session("../../../etc")
