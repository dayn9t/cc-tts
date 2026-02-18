"""Tests for the wakeword backend factory."""

import pytest
from unittest.mock import patch, MagicMock

from cc_stt.wakeword.factory import create_wakeword_backend
from cc_stt.wakeword.openwakeword import OpenWakeWordBackend
from cc_stt.wakeword.wekws import WeKWSBackend


class TestCreateWakewordBackend:
    """Tests for create_wakeword_backend factory function."""

    def test_factory_creates_openwakeword(self):
        """Verify OpenWakeWordBackend creation."""
        with patch(
            "cc_stt.wakeword.factory.OpenWakeWordBackend"
        ) as mock_backend_class:
            mock_instance = MagicMock(spec=OpenWakeWordBackend)
            mock_backend_class.return_value = mock_instance

            result = create_wakeword_backend(
                backend="openwakeword",
                name="alexa",
                threshold=0.6,
            )

            mock_backend_class.assert_called_once_with(
                wakeword="alexa",
                threshold=0.6,
            )
            assert result == mock_instance

    def test_factory_creates_wekws(self):
        """Verify WeKWSBackend creation."""
        with patch("cc_stt.wakeword.factory.WeKWSBackend") as mock_backend_class:
            mock_instance = MagicMock(spec=WeKWSBackend)
            mock_backend_class.return_value = mock_instance

            result = create_wakeword_backend(
                backend="wekws",
                name="my_wakeword",
                threshold=0.7,
                model_path="/path/to/model.onnx",
                window_size=50,
            )

            mock_backend_class.assert_called_once_with(
                model_path="/path/to/model.onnx",
                threshold=0.7,
                window_size=50,
            )
            assert result == mock_instance

    def test_factory_invalid_backend(self):
        """Verify error for invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend: invalid"):
            create_wakeword_backend(
                backend="invalid",  # type: ignore
                name="test",
            )

    def test_factory_wekws_without_model_path(self):
        """Verify error when model_path missing for wekws backend."""
        with pytest.raises(
            ValueError, match="model_path is required for 'wekws' backend"
        ):
            create_wakeword_backend(
                backend="wekws",
                name="test",
            )
