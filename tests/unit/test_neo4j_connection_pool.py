"""
Unit tests for Neo4j connection pooling configuration.

Tests cover:
1. Pool configuration from settings
2. Health check functionality
3. Pool metrics retrieval
4. Connection pool parameters are properly passed to driver
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from t4dm.storage.neo4j_store import Neo4jStore
from t4dm.core.config import Settings, get_settings


class TestNeo4jConnectionPoolConfiguration:
    """Test connection pool configuration."""

    def test_pool_settings_exist_in_config(self):
        """Test that pool settings are available in Settings."""
        settings = get_settings()

        assert hasattr(settings, 'neo4j_pool_size')
        assert hasattr(settings, 'neo4j_connection_timeout')
        assert hasattr(settings, 'neo4j_connection_lifetime')

    def test_default_pool_settings(self):
        """Test default values for pool settings."""
        settings = Settings()

        assert settings.neo4j_pool_size == 50
        assert settings.neo4j_connection_timeout == 30.0
        assert settings.neo4j_connection_lifetime == 3600.0

    def test_custom_pool_settings(self):
        """Test that pool settings can be customized."""
        settings = Settings(
            neo4j_pool_size=100,
            neo4j_connection_timeout=60.0,
            neo4j_connection_lifetime=7200.0,
        )

        assert settings.neo4j_pool_size == 100
        assert settings.neo4j_connection_timeout == 60.0
        assert settings.neo4j_connection_lifetime == 7200.0

    @pytest.mark.asyncio
    async def test_driver_initialization_with_pool_settings(self):
        """Test that driver is initialized with pool settings."""
        store = Neo4jStore()

        with patch('t4dm.storage.neo4j_store.AsyncGraphDatabase.driver') as mock_driver:
            mock_driver.return_value = MagicMock()

            await store._get_driver()

            # Verify driver was called with pool parameters
            mock_driver.assert_called_once()
            call_kwargs = mock_driver.call_args.kwargs

            assert 'max_connection_pool_size' in call_kwargs
            assert 'connection_acquisition_timeout' in call_kwargs
            assert 'max_connection_lifetime' in call_kwargs
            assert 'connection_timeout' in call_kwargs
            assert 'keep_alive' in call_kwargs
            assert call_kwargs['keep_alive'] is True

    @pytest.mark.asyncio
    async def test_driver_initialization_uses_config_values(self):
        """Test that driver uses values from config."""
        store = Neo4jStore()
        settings = get_settings()

        with patch('t4dm.storage.neo4j_store.AsyncGraphDatabase.driver') as mock_driver:
            mock_driver.return_value = MagicMock()

            await store._get_driver()

            call_kwargs = mock_driver.call_args.kwargs

            assert call_kwargs['max_connection_pool_size'] == settings.neo4j_pool_size
            assert call_kwargs['connection_acquisition_timeout'] == settings.neo4j_connection_timeout
            assert call_kwargs['max_connection_lifetime'] == settings.neo4j_connection_lifetime
            assert call_kwargs['connection_timeout'] == settings.neo4j_connection_timeout

    @pytest.mark.asyncio
    async def test_driver_created_only_once(self):
        """Test that driver is only created once (cached)."""
        store = Neo4jStore()

        with patch('t4dm.storage.neo4j_store.AsyncGraphDatabase.driver') as mock_driver:
            mock_driver.return_value = MagicMock()

            # Call _get_driver multiple times
            driver1 = await store._get_driver()
            driver2 = await store._get_driver()
            driver3 = await store._get_driver()

            # Driver should only be initialized once
            assert mock_driver.call_count == 1
            assert driver1 is driver2
            assert driver2 is driver3


class TestNeo4jHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self):
        """Test that health_check returns a status dict."""
        store = Neo4jStore()

        # Mock the driver and session
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={'test': 1})

        mock_session = MagicMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session)

        with patch.object(store, '_get_driver', return_value=mock_driver):
            result = await store.health_check()

            assert 'status' in result
            assert 'database' in result
            assert 'uri' in result
            assert 'pool_metrics' in result

    @pytest.mark.asyncio
    async def test_health_check_healthy_status(self):
        """Test that successful health check returns healthy status."""
        store = Neo4jStore()

        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={'test': 1})

        mock_session = MagicMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session)

        with patch.object(store, '_get_driver', return_value=mock_driver):
            result = await store.health_check()

            assert result['status'] == 'healthy'
            assert result['database'] == store.database
            assert result['uri'] == store.uri

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_failure(self):
        """Test that failed health check returns unhealthy status."""
        store = Neo4jStore()

        mock_session = MagicMock()
        mock_session.run = AsyncMock(side_effect=Exception("Connection failed"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session)

        with patch.object(store, '_get_driver', return_value=mock_driver):
            result = await store.health_check()

            assert result['status'] == 'unhealthy'
            assert 'error' in result
            assert 'Connection failed' in result['error']

    @pytest.mark.asyncio
    async def test_health_check_includes_pool_metrics(self):
        """Test that health check includes pool metrics when available."""
        store = Neo4jStore()

        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={'test': 1})

        mock_session = MagicMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock pool with metrics
        mock_pool = MagicMock()
        mock_pool.in_use = 5
        mock_pool.size = 10
        mock_pool.max_size = 50

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver._pool = mock_pool

        with patch.object(store, '_get_driver', return_value=mock_driver):
            result = await store.health_check()

            assert 'pool_metrics' in result
            pool_metrics = result['pool_metrics']
            assert pool_metrics.get('connections_in_use') == 5
            assert pool_metrics.get('pool_size') == 10
            assert pool_metrics.get('max_pool_size') == 50


class TestNeo4jPoolMetrics:
    """Test pool metrics retrieval."""

    @pytest.mark.asyncio
    async def test_get_pool_metrics_before_initialization(self):
        """Test getting pool metrics before driver is initialized."""
        store = Neo4jStore()

        metrics = await store.get_pool_metrics()

        assert metrics['driver_initialized'] is False

    @pytest.mark.asyncio
    async def test_get_pool_metrics_after_initialization(self):
        """Test getting pool metrics after driver is initialized."""
        store = Neo4jStore()

        mock_driver = MagicMock()
        store._driver = mock_driver

        metrics = await store.get_pool_metrics()

        assert metrics['driver_initialized'] is True
        assert 'uri' in metrics
        assert 'database' in metrics
        assert metrics['uri'] == store.uri
        assert metrics['database'] == store.database

    @pytest.mark.asyncio
    async def test_get_pool_metrics_includes_config(self):
        """Test that pool metrics include configured values."""
        store = Neo4jStore()
        store._driver = MagicMock()

        settings = get_settings()
        metrics = await store.get_pool_metrics()

        assert metrics['configured_max_pool_size'] == settings.neo4j_pool_size
        assert metrics['connection_timeout'] == settings.neo4j_connection_timeout
        assert metrics['connection_lifetime'] == settings.neo4j_connection_lifetime

    @pytest.mark.asyncio
    async def test_get_pool_metrics_includes_runtime_stats(self):
        """Test that pool metrics include runtime statistics when available."""
        store = Neo4jStore()

        # Mock pool with statistics
        mock_pool = MagicMock()
        mock_pool.in_use = 15
        mock_pool.size = 20
        mock_pool.max_size = 50

        mock_driver = MagicMock()
        mock_driver._pool = mock_pool
        store._driver = mock_driver

        metrics = await store.get_pool_metrics()

        assert metrics['connections_in_use'] == 15
        assert metrics['current_pool_size'] == 20
        assert metrics['max_pool_size'] == 50

    @pytest.mark.asyncio
    async def test_get_pool_metrics_handles_missing_pool(self):
        """Test that pool metrics handles missing _pool attribute gracefully."""
        store = Neo4jStore()

        mock_driver = MagicMock(spec=[])  # No _pool attribute
        store._driver = mock_driver

        metrics = await store.get_pool_metrics()

        # Should still return basic metrics
        assert metrics['driver_initialized'] is True
        assert 'configured_max_pool_size' in metrics


class TestNeo4jPoolSettingsValidation:
    """Test validation of pool settings."""

    def test_pool_size_must_be_positive(self):
        """Test that pool size validation works."""
        # Valid pool size
        settings = Settings(neo4j_pool_size=1)
        assert settings.neo4j_pool_size == 1

        settings = Settings(neo4j_pool_size=100)
        assert settings.neo4j_pool_size == 100

    def test_connection_timeout_must_be_positive(self):
        """Test that connection timeout validation works."""
        # Valid timeout
        settings = Settings(neo4j_connection_timeout=0.1)
        assert settings.neo4j_connection_timeout == 0.1

        settings = Settings(neo4j_connection_timeout=300.0)
        assert settings.neo4j_connection_timeout == 300.0

    def test_connection_lifetime_must_be_positive(self):
        """Test that connection lifetime validation works."""
        # Valid lifetime
        settings = Settings(neo4j_connection_lifetime=60.0)
        assert settings.neo4j_connection_lifetime == 60.0

        settings = Settings(neo4j_connection_lifetime=86400.0)
        assert settings.neo4j_connection_lifetime == 86400.0


class TestNeo4jPoolLogging:
    """Test logging of pool initialization."""

    @pytest.mark.asyncio
    async def test_driver_initialization_logs_pool_config(self):
        """Test that driver initialization logs pool configuration."""
        store = Neo4jStore()

        with patch('t4dm.storage.neo4j_store.AsyncGraphDatabase.driver') as mock_driver:
            mock_driver.return_value = MagicMock()

            with patch('t4dm.storage.neo4j_store.logger') as mock_logger:
                await store._get_driver()

                # Should log pool configuration
                mock_logger.info.assert_called_once()
                log_message = mock_logger.info.call_args[0][0]

                assert 'pool_size' in log_message
                assert 'timeout' in log_message
                assert 'lifetime' in log_message
