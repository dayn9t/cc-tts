"""Tests for SherpaONNX backend implementation."""

from pathlib import Path

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestSherpaONNXBackend:
    """Tests for SherpaONNXBackend class."""

    @patch("cc_stt.wakeword.sherpa_onnx.OnlineTransducerModelConfig")
    @patch("cc_stt.wakeword.sherpa_onnx.HAS_SHERPA_ONNX", True)
    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    def test_sherpa_onnx_backend_init(self, mock_spotter_class, mock_config_class):
        """Test SherpaONNXBackend initialization creates KeywordSpotter."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        # Mock the KeywordSpotter
        mock_spotter = MagicMock()
        mock_spotter_class.return_value = mock_spotter

        # Mock the config class
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        # Mock file existence check
        with patch.object(Path, "exists", return_value=True):
            backend = SherpaONNXBackend(
                model_dir="/path/to/model",
                keywords=["你好"],
                num_threads=4,
                provider="cpu",
            )

        mock_spotter_class.assert_called_once()
        assert str(backend.model_dir) == "/path/to/model"
        assert backend.num_threads == 4
        assert backend.provider == "cpu"

    @patch("cc_stt.wakeword.sherpa_onnx.OnlineTransducerModelConfig")
    @patch("cc_stt.wakeword.sherpa_onnx.HAS_SHERPA_ONNX", True)
    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    def test_sherpa_onnx_backend_init_with_keywords_file(self, mock_spotter_class, mock_config_class):
        """Test SherpaONNXBackend initialization with keywords file."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        mock_spotter = MagicMock()
        mock_spotter_class.return_value = mock_spotter

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        with patch.object(Path, "exists", return_value=True):
            backend = SherpaONNXBackend(
                model_dir="/path/to/model",
                keywords_file="/path/to/keywords.txt",
                num_threads=2,
                provider="cpu",
            )

        mock_spotter_class.assert_called_once()
        assert backend.keywords_file == "/path/to/keywords.txt"

    @patch("cc_stt.wakeword.sherpa_onnx.OnlineTransducerModelConfig")
    @patch("cc_stt.wakeword.sherpa_onnx.HAS_SHERPA_ONNX", True)
    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    def test_sherpa_onnx_process_audio(self, mock_spotter_class, mock_config_class):
        """Test audio processing with no detection."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        # Mock the KeywordSpotter and stream
        mock_spotter = MagicMock()
        mock_spotter.get_result.return_value = ""  # No keyword detected
        mock_spotter_class.return_value = mock_spotter

        mock_stream = MagicMock()
        mock_spotter.create_stream.return_value = mock_stream

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        with patch.object(Path, "exists", return_value=True):
            backend = SherpaONNXBackend(
                model_dir="/path/to/model",
                keywords=["你好"],
            )

        # Process audio
        audio = np.zeros(16000, dtype=np.float32)
        result = backend.process_audio(audio)

        # Verify stream accepted waveform
        mock_stream.accept_waveform.assert_called_once()
        assert result is False  # No detection

    @patch("cc_stt.wakeword.sherpa_onnx.OnlineTransducerModelConfig")
    @patch("cc_stt.wakeword.sherpa_onnx.HAS_SHERPA_ONNX", True)
    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    def test_sherpa_onnx_process_audio_detection(self, mock_spotter_class, mock_config_class):
        """Test audio processing with detection."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        # Mock the KeywordSpotter and stream
        mock_spotter = MagicMock()
        mock_spotter.get_result.return_value = "你好"  # Keyword detected
        mock_spotter_class.return_value = mock_spotter

        mock_stream = MagicMock()
        mock_spotter.create_stream.return_value = mock_stream

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        with patch.object(Path, "exists", return_value=True):
            backend = SherpaONNXBackend(
                model_dir="/path/to/model",
                keywords=["你好"],
            )

        # Process audio
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = backend.process_audio(audio)

        assert result is True  # Detection occurred

    @patch("cc_stt.wakeword.sherpa_onnx.OnlineTransducerModelConfig")
    @patch("cc_stt.wakeword.sherpa_onnx.HAS_SHERPA_ONNX", True)
    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    def test_sherpa_onnx_reset(self, mock_spotter_class, mock_config_class):
        """Test reset method creates new stream."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        mock_spotter = MagicMock()
        mock_spotter_class.return_value = mock_spotter

        mock_stream1 = MagicMock()
        mock_stream2 = MagicMock()
        mock_spotter.create_stream.side_effect = [mock_stream1, mock_stream2]

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        with patch.object(Path, "exists", return_value=True):
            backend = SherpaONNXBackend(
                model_dir="/path/to/model",
                keywords=["你好"],
            )

        # Initial stream created
        assert backend._stream == mock_stream1

        # Reset creates new stream
        backend.reset()

        assert backend._stream == mock_stream2
        mock_spotter.create_stream.assert_called()

    @patch("cc_stt.wakeword.sherpa_onnx.HAS_SHERPA_ONNX", True)
    def test_check_model_files_missing(self):
        """Test that missing model files raise error."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                SherpaONNXBackend(
                    model_dir="/path/to/model",
                    keywords=["你好"],
                )

    @patch("cc_stt.wakeword.sherpa_onnx.OnlineTransducerModelConfig")
    @patch("cc_stt.wakeword.sherpa_onnx.HAS_SHERPA_ONNX", True)
    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    def test_create_keywords_file(self, mock_spotter_class, mock_config_class):
        """Test that keywords list is converted to file."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        mock_spotter = MagicMock()
        mock_spotter_class.return_value = mock_spotter

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        with patch.object(Path, "exists", return_value=True):
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_file = MagicMock()
                mock_file.name = "/tmp/keywords.txt"
                mock_temp.return_value.__enter__.return_value = mock_file

                backend = SherpaONNXBackend(
                    model_dir="/path/to/model",
                    keywords=["你好", "小助手"],
                )

                # Verify keywords were written to temp file (once per keyword)
                assert mock_file.write.call_count == 2
