"""Tests for WeKWS backend implementation."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestRingBuffer:
    """Tests for RingBuffer class."""

    def test_ring_buffer_basic(self):
        """Test basic append and overflow behavior."""
        from cc_stt.wakeword.wekws import RingBuffer

        buffer = RingBuffer(maxlen=5)

        # Append items
        buffer.append(np.array([1.0, 2.0]))
        buffer.append(np.array([3.0, 4.0]))
        buffer.append(np.array([5.0, 6.0]))

        assert len(buffer) == 3

        # Append more to cause overflow
        buffer.append(np.array([7.0, 8.0]))
        buffer.append(np.array([9.0, 10.0]))
        buffer.append(np.array([11.0, 12.0]))  # This should overflow

        assert len(buffer) == 5  # Should not exceed maxlen

    def test_ring_buffer_to_array(self):
        """Test to_array returns correct shape."""
        from cc_stt.wakeword.wekws import RingBuffer

        buffer = RingBuffer(maxlen=4)

        # Append 3 frames with 5 features each
        for i in range(3):
            buffer.append(np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + i)

        arr = buffer.to_array()

        assert arr.shape == (3, 5)  # (n_frames, n_features)
        assert arr.dtype == np.float32

    def test_ring_buffer_clear(self):
        """Test clear empties the buffer."""
        from cc_stt.wakeword.wekws import RingBuffer

        buffer = RingBuffer(maxlen=5)
        buffer.append(np.array([1.0, 2.0]))
        buffer.append(np.array([3.0, 4.0]))

        assert len(buffer) == 2

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.to_array().shape == (0,)


class TestWeKWSBackend:
    """Tests for WeKWSBackend class."""

    @patch("cc_stt.wakeword.wekws.rt.InferenceSession")
    def test_wekws_backend_initialization(self, mock_session_class):
        """Test WeKWSBackend initialization creates ONNX session."""
        from cc_stt.wakeword.wekws import WeKWSBackend

        # Mock the ONNX session
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        backend = WeKWSBackend(
            model_path="/path/to/model.onnx",
            threshold=0.6,
            window_size=40,
        )

        mock_session_class.assert_called_once_with("/path/to/model.onnx")
        assert backend.threshold == 0.6
        assert backend.window_size == 40

    @patch("cc_stt.wakeword.wekws.rt.InferenceSession")
    def test_wekws_process_audio_silence(self, mock_session_class):
        """Test that silent audio returns False."""
        from cc_stt.wakeword.wekws import WeKWSBackend

        # Mock the ONNX session with low score output
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.1, 0.2]])]  # Low scores
        mock_session_class.return_value = mock_session

        # Mock feature extractor to return controlled number of frames
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = np.array([
            [1.0] * 80,  # 1 frame, 80 features
        ], dtype=np.float32)

        backend = WeKWSBackend(
            model_path="/path/to/model.onnx",
            threshold=0.5,
            window_size=4,
            feature_extractor=mock_extractor,
        )

        # Process audio - need 4 frames to fill buffer
        audio = np.zeros(160, dtype=np.float32)

        result = backend.process_audio(audio)  # 1 frame, buffer not full
        assert result is False

        result = backend.process_audio(audio)  # 2 frames, buffer not full
        assert result is False

        result = backend.process_audio(audio)  # 3 frames, buffer not full
        assert result is False

        result = backend.process_audio(audio)  # 4 frames, inference runs
        assert result is False  # Low score, no detection

    @patch("cc_stt.wakeword.wekws.rt.InferenceSession")
    def test_wekws_process_audio_detection(self, mock_session_class):
        """Test that high score returns True."""
        from cc_stt.wakeword.wekws import WeKWSBackend

        # Mock the ONNX session with high score output
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.95, 0.05]])]  # High score
        mock_session_class.return_value = mock_session

        # Mock feature extractor to return controlled number of frames
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = np.array([
            [1.0] * 80,  # 1 frame, 80 features
        ], dtype=np.float32)

        backend = WeKWSBackend(
            model_path="/path/to/model.onnx",
            threshold=0.5,
            window_size=2,
            feature_extractor=mock_extractor,
        )

        # Process audio - need 2 frames to fill buffer
        audio = np.random.randn(160).astype(np.float32) * 0.1

        result = backend.process_audio(audio)  # 1 frame, buffer not full
        assert result is False

        result = backend.process_audio(audio)  # 2 frames, inference runs
        assert result is True  # High score, detection triggered

    @patch("cc_stt.wakeword.wekws.rt.InferenceSession")
    def test_wekws_reset(self, mock_session_class):
        """Test that reset clears the feature buffer."""
        from cc_stt.wakeword.wekws import WeKWSBackend

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock feature extractor to return controlled number of frames
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = np.array([
            [1.0] * 80,  # 1 frame, 80 features
        ], dtype=np.float32)

        backend = WeKWSBackend(
            model_path="/path/to/model.onnx",
            threshold=0.5,
            window_size=10,
            feature_extractor=mock_extractor,
        )

        # Add 2 frames
        audio = np.zeros(160, dtype=np.float32)
        backend.process_audio(audio)
        backend.process_audio(audio)

        assert len(backend._feature_buffer) == 2

        backend.reset()

        assert len(backend._feature_buffer) == 0
