import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from cc_stt.wakeword import OpenWakeWordBackend


class TestOpenWakeWordBackend:
    """Tests for OpenWakeWordBackend (formerly WakewordDetector)."""

    @patch("cc_stt.wakeword.openwakeword.openwakeword.get_pretrained_model_paths")
    @patch("cc_stt.wakeword.openwakeword.Model")
    def test_openwakeword_backend_init(self, mock_model_class, mock_get_paths):
        """Test OpenWakeWordBackend initialization."""
        mock_get_paths.return_value = ["/path/to/alexa_v0.1.onnx"]
        mock_model = MagicMock()
        mock_model.models = {"alexa_v0.1": MagicMock()}
        mock_model_class.return_value = mock_model

        backend = OpenWakeWordBackend(
            wakeword="alexa",
            threshold=0.5,
        )

        assert backend.wakeword == "alexa"
        assert backend.threshold == 0.5
        mock_model_class.assert_called_once()

    @patch("cc_stt.wakeword.openwakeword.openwakeword.get_pretrained_model_paths")
    @patch("cc_stt.wakeword.openwakeword.Model")
    def test_openwakeword_backend_process_audio(self, mock_model_class, mock_get_paths):
        """Test audio processing with OpenWakeWordBackend."""
        mock_get_paths.return_value = ["/path/to/alexa_v0.1.onnx"]
        mock_model = MagicMock()
        mock_model.models = {"alexa_v0.1": MagicMock()}
        mock_model.predict.return_value = {"alexa_v0.1": 0.0}  # No detection
        mock_model_class.return_value = mock_model

        backend = OpenWakeWordBackend(
            wakeword="alexa",
        )

        # Silence audio, should not trigger
        silence = np.zeros(1280, dtype=np.float32)
        result = backend.process_audio(silence)

        assert result is False
        mock_model.predict.assert_called_once()

    @patch("cc_stt.wakeword.openwakeword.openwakeword.get_pretrained_model_paths")
    @patch("cc_stt.wakeword.openwakeword.Model")
    def test_openwakeword_backend_detection(self, mock_model_class, mock_get_paths):
        """Test wakeword detection."""
        mock_get_paths.return_value = ["/path/to/alexa_v0.1.onnx"]
        mock_model = MagicMock()
        mock_model.models = {"alexa_v0.1": MagicMock()}
        mock_model.predict.return_value = {"alexa_v0.1": 0.8}  # Detection triggered
        mock_model_class.return_value = mock_model

        backend = OpenWakeWordBackend(
            wakeword="alexa",
            threshold=0.5,
        )

        audio = np.random.randn(1280).astype(np.float32) * 0.1
        result = backend.process_audio(audio)

        assert result is True

    @patch("cc_stt.wakeword.openwakeword.openwakeword.get_pretrained_model_paths")
    @patch("cc_stt.wakeword.openwakeword.Model")
    def test_openwakeword_backend_reset(self, mock_model_class, mock_get_paths):
        """Test reset method."""
        mock_get_paths.return_value = ["/path/to/alexa_v0.1.onnx"]
        mock_model = MagicMock()
        mock_model.models = {"alexa_v0.1": MagicMock()}
        mock_model_class.return_value = mock_model

        backend = OpenWakeWordBackend(
            wakeword="alexa",
        )

        backend.reset()

        mock_model.reset.assert_called_once()
