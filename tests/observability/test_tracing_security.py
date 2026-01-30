"""Tests for OpenTelemetry tracing security configuration."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# Use a valid test password that meets complexity requirements
TEST_PASSWORD = "TestPassword123!"


class TestOTLPSecurityConfig:
    """Tests for OTLP security configuration."""

    def test_insecure_mode_default(self):
        """Test that insecure mode is default."""
        from ww.core.config import Settings

        settings = Settings(neo4j_password=TEST_PASSWORD)
        assert settings.otel_insecure is True

    def test_secure_mode_configurable(self):
        """Test that secure mode can be enabled."""
        from ww.core.config import Settings

        settings = Settings(neo4j_password=TEST_PASSWORD, otel_insecure=False)
        assert settings.otel_insecure is False

    def test_cert_file_configurable(self):
        """Test that certificate file can be configured."""
        from ww.core.config import Settings

        settings = Settings(neo4j_password=TEST_PASSWORD, otel_cert_file="/path/to/cert.pem")
        assert settings.otel_cert_file == "/path/to/cert.pem"

    def test_headers_configurable(self):
        """Test that headers can be configured."""
        from ww.core.config import Settings

        settings = Settings(neo4j_password=TEST_PASSWORD, otel_headers={"Authorization": "Bearer token"})
        assert settings.otel_headers == {"Authorization": "Bearer token"}


class TestTracingConfiguration:
    """Tests for tracing configuration."""

    def test_tracing_disabled_by_default(self):
        """Test that tracing is disabled by default."""
        from ww.core.config import Settings

        settings = Settings(neo4j_password=TEST_PASSWORD)
        assert settings.otel_enabled is False

    def test_configure_tracing_disabled(self):
        """Test that configure_tracing returns None when disabled."""
        from ww.core.config import Settings
        from ww.observability.tracing import configure_tracing, shutdown_tracing

        with patch("ww.core.config.get_settings") as mock_settings:
            mock_settings.return_value = Settings(neo4j_password=TEST_PASSWORD, otel_enabled=False)

            try:
                result = configure_tracing()
                assert result is None
            finally:
                shutdown_tracing()

    def test_configure_tracing_enabled_insecure(self):
        """Test tracing configuration in insecure mode."""
        from ww.core.config import Settings
        from ww.observability.tracing import configure_tracing, shutdown_tracing

        with patch("ww.core.config.get_settings") as mock_settings, \
             patch("ww.observability.tracing.OTLPSpanExporter") as mock_exporter:
            mock_settings.return_value = Settings(
                neo4j_password=TEST_PASSWORD,
                otel_enabled=True,
                otel_insecure=True,
            )
            mock_exporter.return_value = MagicMock()

            try:
                provider = configure_tracing()

                assert provider is not None
                mock_exporter.assert_called_once()
                call_kwargs = mock_exporter.call_args.kwargs
                assert call_kwargs.get("insecure") is True
            finally:
                shutdown_tracing()

    def test_configure_tracing_with_headers(self):
        """Test tracing configuration with auth headers."""
        from ww.core.config import Settings
        from ww.observability.tracing import configure_tracing, shutdown_tracing

        with patch("ww.core.config.get_settings") as mock_settings, \
             patch("ww.observability.tracing.OTLPSpanExporter") as mock_exporter:
            mock_settings.return_value = Settings(
                neo4j_password=TEST_PASSWORD,
                otel_enabled=True,
                otel_headers={"Authorization": "Bearer token"},
            )
            mock_exporter.return_value = MagicMock()

            try:
                configure_tracing()

                call_kwargs = mock_exporter.call_args.kwargs
                assert call_kwargs.get("headers") == (("Authorization", "Bearer token"),)
            finally:
                shutdown_tracing()

    def test_configure_tracing_with_cert_file(self):
        """Test tracing configuration with custom certificate."""
        from ww.core.config import Settings
        from ww.observability.tracing import configure_tracing, shutdown_tracing

        # Create a temporary certificate file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write("-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----\n")
            cert_path = f.name

        try:
            with patch("ww.core.config.get_settings") as mock_settings, \
                 patch("ww.observability.tracing.OTLPSpanExporter") as mock_exporter, \
                 patch("grpc.ssl_channel_credentials") as mock_creds:
                mock_settings.return_value = Settings(
                    neo4j_password=TEST_PASSWORD,
                    otel_enabled=True,
                    otel_insecure=False,
                    otel_cert_file=cert_path,
                )
                mock_exporter.return_value = MagicMock()
                mock_creds.return_value = MagicMock()

                configure_tracing()

                # Verify credentials were created
                mock_creds.assert_called_once()
        finally:
            os.unlink(cert_path)
            shutdown_tracing()

    def test_configure_tracing_secure_no_cert(self):
        """Test tracing configuration with system certificates."""
        from ww.core.config import Settings
        from ww.observability.tracing import configure_tracing, shutdown_tracing

        with patch("ww.core.config.get_settings") as mock_settings, \
             patch("ww.observability.tracing.OTLPSpanExporter") as mock_exporter:
            mock_settings.return_value = Settings(
                neo4j_password=TEST_PASSWORD,
                otel_enabled=True,
                otel_insecure=False,
                otel_cert_file=None,
            )
            mock_exporter.return_value = MagicMock()

            try:
                configure_tracing()

                call_kwargs = mock_exporter.call_args.kwargs
                assert call_kwargs.get("insecure") is False
            finally:
                shutdown_tracing()


class TestTracedDecorator:
    """Tests for traced decorator."""

    def test_traced_sync_function(self):
        """Test traced decorator on sync function."""
        from ww.core.config import Settings
        from ww.observability.tracing import shutdown_tracing, traced_sync

        @traced_sync()
        def my_function():
            return "result"

        with patch("ww.core.config.get_settings") as mock_settings:
            mock_settings.return_value = Settings(neo4j_password=TEST_PASSWORD, otel_enabled=False)

            try:
                # Should not error even without tracing configured
                result = my_function()
                assert result == "result"
            finally:
                shutdown_tracing()

    @pytest.mark.asyncio
    async def test_traced_async_function(self):
        """Test traced decorator on async function."""
        from ww.core.config import Settings
        from ww.observability.tracing import shutdown_tracing, traced

        @traced()
        async def my_async_function():
            return "async_result"

        with patch("ww.core.config.get_settings") as mock_settings:
            mock_settings.return_value = Settings(neo4j_password=TEST_PASSWORD, otel_enabled=False)

            try:
                result = await my_async_function()
                assert result == "async_result"
            finally:
                shutdown_tracing()

    def test_traced_with_custom_name(self):
        """Test traced decorator with custom span name."""
        from ww.core.config import Settings
        from ww.observability.tracing import shutdown_tracing, traced_sync

        @traced_sync(name="custom_span")
        def my_function():
            return "result"

        with patch("ww.core.config.get_settings") as mock_settings:
            mock_settings.return_value = Settings(neo4j_password=TEST_PASSWORD, otel_enabled=False)

            try:
                result = my_function()
                assert result == "result"
            finally:
                shutdown_tracing()

    def test_traced_with_attributes(self):
        """Test traced decorator with attributes."""
        from ww.core.config import Settings
        from ww.observability.tracing import shutdown_tracing, traced_sync

        @traced_sync(attributes={"key": "value"})
        def my_function():
            return "result"

        with patch("ww.core.config.get_settings") as mock_settings:
            mock_settings.return_value = Settings(neo4j_password=TEST_PASSWORD, otel_enabled=False)

            try:
                result = my_function()
                assert result == "result"
            finally:
                shutdown_tracing()

    def test_traced_exception_handling(self):
        """Test traced decorator records exceptions."""
        from ww.core.config import Settings
        from ww.observability.tracing import shutdown_tracing, traced_sync

        @traced_sync()
        def failing_function():
            raise ValueError("Test error")

        with patch("ww.core.config.get_settings") as mock_settings:
            mock_settings.return_value = Settings(neo4j_password=TEST_PASSWORD, otel_enabled=False)

            try:
                with pytest.raises(ValueError):
                    failing_function()
            finally:
                shutdown_tracing()


class TestBatchExportConfig:
    """Tests for batch export configuration."""

    def test_batch_delay_configurable(self):
        """Test batch delay is configurable."""
        from ww.core.config import Settings

        settings = Settings(neo4j_password=TEST_PASSWORD, otel_batch_delay_ms=10000)
        assert settings.otel_batch_delay_ms == 10000

    def test_batch_size_configurable(self):
        """Test batch size is configurable."""
        from ww.core.config import Settings

        settings = Settings(neo4j_password=TEST_PASSWORD, otel_max_export_batch_size=1000)
        assert settings.otel_max_export_batch_size == 1000

    def test_batch_delay_validation_too_small(self):
        """Test batch delay validation for too small values."""
        from pydantic import ValidationError

        from ww.core.config import Settings

        with pytest.raises(ValidationError):
            Settings(otel_batch_delay_ms=50)  # Too small

    def test_batch_delay_validation_too_large(self):
        """Test batch delay validation for too large values."""
        from pydantic import ValidationError

        from ww.core.config import Settings

        with pytest.raises(ValidationError):
            Settings(otel_batch_delay_ms=100000)  # Too large

    def test_batch_size_validation_zero(self):
        """Test batch size validation for zero."""
        from pydantic import ValidationError

        from ww.core.config import Settings

        with pytest.raises(ValidationError):
            Settings(otel_max_export_batch_size=0)

    def test_batch_size_validation_too_large(self):
        """Test batch size validation for too large values."""
        from pydantic import ValidationError

        from ww.core.config import Settings

        with pytest.raises(ValidationError):
            Settings(otel_max_export_batch_size=20000)


class TestTracerProviderCaching:
    """Tests for tracer provider caching."""

    def test_provider_cached_on_reconfig(self):
        """Test that provider is cached and not recreated."""
        from ww.core.config import Settings
        from ww.observability.tracing import configure_tracing, shutdown_tracing

        with patch("ww.core.config.get_settings") as mock_settings, \
             patch("ww.observability.tracing.OTLPSpanExporter") as mock_exporter:
            mock_settings.return_value = Settings(
                neo4j_password=TEST_PASSWORD,
                otel_enabled=True,
                otel_insecure=True,
            )
            mock_exporter.return_value = MagicMock()

            try:
                provider1 = configure_tracing()
                provider2 = configure_tracing()

                # Should return the same provider
                assert provider1 is provider2

                # Exporter should only be created once
                assert mock_exporter.call_count == 1
            finally:
                shutdown_tracing()

    def test_provider_recreated_after_shutdown(self):
        """Test that provider can be recreated after shutdown."""
        from ww.core.config import Settings
        from ww.observability.tracing import configure_tracing, shutdown_tracing

        with patch("ww.core.config.get_settings") as mock_settings, \
             patch("ww.observability.tracing.OTLPSpanExporter") as mock_exporter:
            mock_settings.return_value = Settings(
                neo4j_password=TEST_PASSWORD,
                otel_enabled=True,
                otel_insecure=True,
            )
            mock_exporter.return_value = MagicMock()

            provider1 = configure_tracing()
            shutdown_tracing()

            provider2 = configure_tracing()

            # Should create new provider after shutdown
            assert provider1 is not provider2

            # Exporter should be created twice
            assert mock_exporter.call_count == 2

            shutdown_tracing()


class TestBatchProcessorConfig:
    """Tests for batch processor configuration."""

    def test_batch_processor_uses_config_values(self):
        """Test that batch processor uses config values."""
        from ww.core.config import Settings
        from ww.observability.tracing import configure_tracing, shutdown_tracing

        with patch("ww.core.config.get_settings") as mock_settings, \
             patch("ww.observability.tracing.OTLPSpanExporter") as mock_exporter, \
             patch("ww.observability.tracing.BatchSpanProcessor") as mock_processor:
            mock_settings.return_value = Settings(
                neo4j_password=TEST_PASSWORD,
                otel_enabled=True,
                otel_batch_delay_ms=1000,
                otel_max_export_batch_size=256,
            )
            mock_exporter.return_value = MagicMock()
            mock_processor.return_value = MagicMock()

            try:
                configure_tracing()

                # Verify processor was created with correct config
                call_kwargs = mock_processor.call_args.kwargs
                assert call_kwargs.get("schedule_delay_millis") == 1000
                assert call_kwargs.get("max_export_batch_size") == 256
            finally:
                shutdown_tracing()


class TestGetTracer:
    """Tests for get_tracer function."""

    def test_get_tracer_auto_initializes(self):
        """Test that get_tracer auto-initializes if not configured."""
        from ww.core.config import Settings
        from ww.observability.tracing import get_tracer, shutdown_tracing

        with patch("ww.core.config.get_settings") as mock_settings:
            mock_settings.return_value = Settings(neo4j_password=TEST_PASSWORD, otel_enabled=False)

            try:
                tracer = get_tracer()
                assert tracer is not None
            finally:
                shutdown_tracing()

    def test_get_tracer_with_name(self):
        """Test that get_tracer accepts custom name."""
        from ww.core.config import Settings
        from ww.observability.tracing import get_tracer, shutdown_tracing

        with patch("ww.core.config.get_settings") as mock_settings:
            mock_settings.return_value = Settings(neo4j_password=TEST_PASSWORD, otel_enabled=False)

            try:
                tracer = get_tracer(name="test-service")
                assert tracer is not None
            finally:
                shutdown_tracing()


class TestInitTracingBackwardCompat:
    """Tests for init_tracing backward compatibility."""

    def test_init_tracing_calls_configure(self):
        """Test that init_tracing calls configure_tracing."""
        from ww.core.config import Settings
        from ww.observability.tracing import init_tracing, shutdown_tracing

        with patch("ww.core.config.get_settings") as mock_settings, \
             patch("ww.observability.tracing.OTLPSpanExporter") as mock_exporter:
            mock_settings.return_value = Settings(
                neo4j_password=TEST_PASSWORD,
                otel_enabled=True,
                otel_insecure=True,
            )
            mock_exporter.return_value = MagicMock()

            try:
                tracer = init_tracing()
                assert tracer is not None
            finally:
                shutdown_tracing()

    def test_init_tracing_disabled_returns_noop(self):
        """Test that init_tracing returns no-op tracer when disabled."""
        from ww.core.config import Settings
        from ww.observability.tracing import init_tracing, shutdown_tracing

        with patch("ww.core.config.get_settings") as mock_settings:
            mock_settings.return_value = Settings(neo4j_password=TEST_PASSWORD, otel_enabled=False)

            try:
                tracer = init_tracing()
                assert tracer is not None
            finally:
                shutdown_tracing()
