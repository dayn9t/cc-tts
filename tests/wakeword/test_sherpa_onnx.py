"""Tests for SherpaONNX backend implementation."""

import os
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestSherpaONNXBackend:
    """Tests for SherpaONNXBackend class."""

    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    @patch("cc_stt.wakeword.sherpa_onnx.os.path.exists")
    def test_sherpa_onnx_backend_init(self, mock_exists, mock_spotter_class):
        """Test SherpaONNXBackend initialization with mock."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        # Mock file existence checks
        mock_exists.return_value = True

        # Mock the KeywordSpotter
        mock_spotter = MagicMock()
        mock_spotter_class.return_value = mock_spotter

        backend = SherpaONNXBackend(
            model_dir="/path/to/model",
            keywords=["hello", "world"],
            num_threads=4,
            provider="cpu",
        )

        # Verify KeywordSpotter was created
        mock_spotter_class.assert_called_once()
        call_kwargs = mock_spotter_class.call_args.kwargs
        assert call_kwargs["num_threads"] == 4
        assert call_kwargs["provider"] == "cpu"

    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    @patch("cc_stt.wakeword.sherpa_onnx.os.path.exists")
    def test_sherpa_onnx_process_audio(self, mock_exists, mock_spotter_class):
        """Test audio processing (no detection)."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        # Mock file existence checks
        mock_exists.return_value = True

        # Mock the KeywordSpotter with no detection
        mock_spotter = MagicMock()
        mock_spotter_class.return_value = mock_spotter

        # Mock stream and methods
        mock_stream = MagicMock()
        mock_spotter.create_stream.return_value = mock_stream
        mock_spotter.is_ready.return_value = False  # Prevent infinite loop

        # Mock empty result (no detection)
        mock_result = MagicMock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        mock_spotter.get_result.return_value = mock_result

        backend = SherpaONNXBackend(
            model_dir="/path/to/model",
            keywords=["hello"],
        )

        # Process audio
        audio = np.zeros(16000, dtype=np.float32)
        result = backend.process_audio(audio)

        assert result is False
        mock_spotter.create_stream.assert_called_once()

    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    @patch("cc_stt.wakeword.sherpa_onnx.os.path.exists")
    def test_sherpa_onnx_process_audio_detection(self, mock_exists, mock_spotter_class):
        """Test audio processing with detection."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        # Mock file existence checks
        mock_exists.return_value = True

        # Mock the KeywordSpotter with detection
        mock_spotter = MagicMock()
        mock_spotter_class.return_value = mock_spotter

        # Mock stream and methods
        mock_stream = MagicMock()
        mock_spotter.create_stream.return_value = mock_stream
        mock_spotter.is_ready.return_value = False  # Process once then stop

        # Mock result with detection
        mock_detection = MagicMock()
        mock_detection.keyword = "hello"
        mock_detection.score = 0.95
        mock_result = MagicMock()
        mock_result.__iter__ = Mock(return_value=iter([mock_detection]))
        mock_spotter.get_result.return_value = mock_result

        backend = SherpaONNXBackend(
            model_dir="/path/to/model",
            keywords=["hello"],
        )

        # Process audio
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = backend.process_audio(audio)

        assert result is True

    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    @patch("cc_stt.wakeword.sherpa_onnx.os.path.exists")
    def test_sherpa_onnx_reset(self, mock_exists, mock_spotter_class):
        """Test reset method."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        # Mock file existence checks
        mock_exists.return_value = True

        # Mock the KeywordSpotter
        mock_spotter = MagicMock()
        mock_spotter_class.return_value = mock_spotter

        backend = SherpaONNXBackend(
            model_dir="/path/to/model",
            keywords=["hello"],
        )

        # Create a stream first
        mock_stream = MagicMock()
        backend._stream = mock_stream

        # Reset should call the spotter's reset method
        backend.reset()

        mock_spotter.reset.assert_called_once_with(mock_stream)

    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    @patch("cc_stt.wakeword.sherpa_onnx.os.path.exists")
    def test_sherpa_onnx_missing_model_files(self, mock_exists, mock_spotter_class):
        """Test that missing model files raises error."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        # Mock file existence to return False
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            SherpaONNXBackend(
                model_dir="/path/to/model",
                keywords=["hello"],
            )

    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    @patch("cc_stt.wakeword.sherpa_onnx.os.path.exists")
    def test_sherpa_onnx_keywords_file_creation(self, mock_exists, mock_spotter_class):
        """Test that keywords file is created from list."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        # Mock file existence checks
        mock_exists.return_value = True

        # Mock the KeywordSpotter
        mock_spotter = MagicMock()
        mock_spotter_class.return_value = mock_spotter

        backend = SherpaONNXBackend(
            model_dir="/path/to/model",
            keywords=["hello", "world"],
        )

        # Verify keywords file was created (stored in _temp_keywords_file)
        assert backend._temp_keywords_file is not None
        assert os.path.exists(backend._temp_keywords_file)

        # Verify file content
        with open(backend._temp_keywords_file) as f:
            content = f.read()
            assert "hello : boost=1.0" in content
            assert "world : boost=1.0" in content

    @patch("cc_stt.wakeword.sherpa_onnx.KeywordSpotter")
    @patch("cc_stt.wakeword.sherpa_onnx.os.path.exists")
    def test_sherpa_onnx_with_keywords_file(self, mock_exists, mock_spotter_class):
        """Test initialization with existing keywords file."""
        from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

        # Mock file existence checks
        mock_exists.return_value = True

        # Mock the KeywordSpotter
        mock_spotter = MagicMock()
        mock_spotter_class.return_value = mock_spotter

        backend = SherpaONNXBackend(
            model_dir="/path/to/model",
            keywords_file="/path/to/keywords.txt",
        )

        # Verify the keywords file path was used
        mock_spotter_class.assert_called_once()
        call_kwargs = mock_spotter_class.call_args.kwargs
        assert call_kwargs["keywords_file"] == "/path/to/keywords.txt"
