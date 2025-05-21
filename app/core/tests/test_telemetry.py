import unittest
from unittest import mock
import os

# Module to be tested
from app.core import telemetry
# Ensure opentelemetry.trace is available for patching
import opentelemetry.trace


class TestTelemetrySetup(unittest.TestCase):
    @mock.patch('app.core.telemetry.trace.set_tracer_provider')
    @mock.patch('app.core.telemetry.BatchSpanProcessor')
    @mock.patch('app.core.telemetry.OTLPSpanExporter')
    @mock.patch('app.core.telemetry.TracerProvider')
    def test_telemetry_disabled_via_env_false(
        self, mock_tracer_provider_cls, mock_otlp_exporter_cls, mock_batch_processor_cls, mock_set_tracer_provider
    ):
        with mock.patch.dict(os.environ, {"OPENTELEMETRY_ENABLED": "false", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://dummy:1234"}):
            telemetry.setup_telemetry()
        mock_tracer_provider_cls.assert_not_called()
        mock_otlp_exporter_cls.assert_not_called()
        mock_batch_processor_cls.assert_not_called()
        mock_set_tracer_provider.assert_not_called() # Since it should return early

    @mock.patch('app.core.telemetry.trace.set_tracer_provider')
    @mock.patch('app.core.telemetry.BatchSpanProcessor')
    @mock.patch('app.core.telemetry.OTLPSpanExporter')
    @mock.patch('app.core.telemetry.TracerProvider')
    def test_telemetry_disabled_via_env_False(
        self, mock_tracer_provider_cls, mock_otlp_exporter_cls, mock_batch_processor_cls, mock_set_tracer_provider
    ):
        with mock.patch.dict(os.environ, {"OPENTELEMETRY_ENABLED": "False", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://dummy:1234"}):
            telemetry.setup_telemetry()
        mock_tracer_provider_cls.assert_not_called()
        mock_otlp_exporter_cls.assert_not_called()
        mock_batch_processor_cls.assert_not_called()
        mock_set_tracer_provider.assert_not_called()

    @mock.patch('app.core.telemetry.trace.set_tracer_provider')
    @mock.patch('app.core.telemetry.BatchSpanProcessor') 
    @mock.patch('app.core.telemetry.OTLPSpanExporter')
    @mock.patch('app.core.telemetry.TracerProvider')
    def test_telemetry_enabled_no_endpoint(
        self, mock_tracer_provider_cls, mock_otlp_exporter_cls, mock_batch_processor_cls, mock_set_tracer_provider
    ):
        # Ensure relevant env vars are not set, or OPENTELEMETRY_ENABLED is explicitly true
        # Clear OTEL_EXPORTER_OTLP_ENDPOINT specifically for this test.
        # Keep OPENTELEMETRY_ENABLED as "true" or ensure it's unset to test default enabled behavior.
        env_vars = {"OPENTELEMETRY_ENABLED": "true"}
        if "OTEL_EXPORTER_OTLP_ENDPOINT" in os.environ:
            # If it's there, we need to ensure it's not for this test.
            # A simple way is to patch.dict with it explicitly removed or not present.
            # Or, more robustly, save and restore. For this, we assume it's not critical to restore.
             original_endpoint = os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)

        with mock.patch.dict(os.environ, env_vars, clear=True):
             # The 'clear=True' in mock.patch.dict might remove OPENTELEMETRY_ENABLED if not careful.
             # Let's ensure OPENTELEMETRY_ENABLED is "true" and OTEL_EXPORTER_OTLP_ENDPOINT is absent.
             current_env = {}
             if "OPENTELEMETRY_ENABLED" in os.environ: # capture what mock.patch.dict will see
                 current_env["OPENTELEMETRY_ENABLED"] = os.environ["OPENTELEMETRY_ENABLED"]
             
             # Override for the test:
             current_env["OPENTELEMETRY_ENABLED"] = "true" 
             if "OTEL_EXPORTER_OTLP_ENDPOINT" in current_env: # Ensure it's not present for this test
                 del current_env["OTEL_EXPORTER_OTLP_ENDPOINT"]

             with mock.patch.dict(os.environ, current_env, clear=True): # clear=True ensures only current_env is visible
                telemetry.setup_telemetry()
        
        if "original_endpoint" in locals() and original_endpoint is not None: # Restore if we removed it
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = original_endpoint


        mock_tracer_provider_cls.assert_called_once() 
        mock_otlp_exporter_cls.assert_not_called()
        mock_batch_processor_cls.assert_not_called() 
        mock_set_tracer_provider.assert_called_once_with(mock_tracer_provider_cls.return_value)

    @mock.patch('app.core.telemetry.trace.set_tracer_provider')
    @mock.patch('app.core.telemetry.BatchSpanProcessor')
    @mock.patch('app.core.telemetry.OTLPSpanExporter')
    @mock.patch('app.core.telemetry.TracerProvider')
    def test_telemetry_enabled_with_endpoint(
        self, mock_tracer_provider_cls, mock_otlp_exporter_cls, mock_batch_processor_cls, mock_set_tracer_provider
    ):
        dummy_endpoint = "http://test-exporter:1234"
        with mock.patch.dict(os.environ, {"OPENTELEMETRY_ENABLED": "true", "OTEL_EXPORTER_OTLP_ENDPOINT": dummy_endpoint}):
            telemetry.setup_telemetry()

        mock_tracer_provider_cls.assert_called_once()
        mock_otlp_exporter_cls.assert_called_once_with(endpoint=dummy_endpoint)
        mock_batch_processor_cls.assert_called_once_with(mock_otlp_exporter_cls.return_value)
        
        mock_tracer_provider_instance = mock_tracer_provider_cls.return_value
        mock_tracer_provider_instance.add_span_processor.assert_called_once_with(mock_batch_processor_cls.return_value)
        
        mock_set_tracer_provider.assert_called_once_with(mock_tracer_provider_instance)

    @mock.patch('app.core.telemetry.trace.get_tracer')
    def test_get_tracer_calls_otel_get_tracer(self, mock_otel_get_tracer):
        tracer_name = "my.custom.tracer"
        returned_tracer = telemetry.get_tracer(tracer_name)
        mock_otel_get_tracer.assert_called_once_with(tracer_name)
        self.assertEqual(returned_tracer, mock_otel_get_tracer.return_value)

if __name__ == '__main__':
    unittest.main()
