"""Tests for the wakeword backend factory."""

import pytest
from unittest.mock import patch, MagicMock

from cc_stt.wakeword.factory import create_wakeword_backend
from cc_stt.wakeword.openwakeword import OpenWakeWordBackend
from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend
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

    def test_factory_creates_sherpa_onnx(self):
        """Verify SherpaONNXBackend creation."""
        with patch(
            "cc_stt.wakeword.factory.SherpaONNXBackend"
        ) as mock_backend_class:
            mock_instance = MagicMock(spec=SherpaONNXBackend)
            mock_backend_class.return_value = mock_instance

            result = create_wakeword_backend(
                backend="sherpa-onnx",
                name="my_wakeword",
                model_dir="/path/to/model",
                keywords=["hello", "world"],
                num_threads=8,
                provider="cuda",
            )

            mock_backend_class.assert_called_once_with(
                model_dir="/path/to/model",
                keywords=["hello", "world"],
                keywords_file=None,
                num_threads=8,
                provider="cuda",
            )
            assert result == mock_instance

    def test_factory_sherpa_onnx_without_model_dir(self):
        """Verify error when model_dir missing for sherpa-onnx backend."""
        with pytest.raises(
            ValueError, match="model_dir is required for 'sherpa-onnx' backend"
        ):
            create_wakeword_backend(
                backend="sherpa-onnx",
                name="test",
                keywords=["hello"],
            )

    def test_factory_sherpa_onnx_without_keywords(self):
        """Verify error when neither keywords nor keywords_file provided for sherpa-onnx backend."""
        with pytest.raises(
            ValueError,
            match="Either keywords or keywords_file is required for 'sherpa-onnx' backend",
        ):
            create_wakeword_backend(
                backend="sherpa-onnx",
                name="test",
                model_dir="/path/to/model",
            )
