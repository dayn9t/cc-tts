"""WeKWS backend implementation for wakeword detection using ONNX Runtime."""

from collections import deque

import numpy as np
import onnxruntime as rt

from cc_stt.features import FeatureExtractor


class RingBuffer:
    """Fixed-size ring buffer for storing feature frames.

    This buffer maintains a sliding window of feature frames for inference.
    When full, new items overwrite the oldest ones.
    """

    def __init__(self, maxlen: int):
        """Initialize the ring buffer.

        Args:
            maxlen: Maximum number of items to store in the buffer.
        """
        self.maxlen = maxlen
        self._buffer: deque[np.ndarray] = deque(maxlen=maxlen)

    def append(self, item: np.ndarray) -> None:
        """Add an item to the buffer.

        Args:
            item: Numpy array to append to the buffer.
        """
        self._buffer.append(item)

    def __len__(self) -> int:
        """Return the current number of items in the buffer."""
        return len(self._buffer)

    def clear(self) -> None:
        """Clear all items from the buffer."""
        self._buffer.clear()

    def to_array(self) -> np.ndarray:
        """Convert buffer contents to a numpy array.

        Returns:
            Numpy array of shape (n_frames, n_features) as float32.
            Returns empty array with shape (0,) if buffer is empty.
        """
        if len(self._buffer) == 0:
            return np.array([], dtype=np.float32)

        return np.stack(self._buffer, axis=0).astype(np.float32)


class WeKWSBackend:
    """Wakeword detection backend using WeKWS models with ONNX Runtime.

    This backend uses a sliding window approach where audio features are
    accumulated in a ring buffer and inference is run when the buffer is full.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        window_size: int = 40,
        feature_extractor: FeatureExtractor | None = None,
        sample_rate: int = 16000,
    ):
        """Initialize the WeKWS backend.

        Args:
            model_path: Path to the ONNX model file.
            threshold: Detection threshold (0.0 to 1.0). Higher values require
                stronger confidence for detection.
            window_size: Number of feature frames to accumulate before running
                inference.
            feature_extractor: FeatureExtractor instance for extracting FBank
                features. If None, a default extractor is created.
            sample_rate: Audio sample rate in Hz.

        Raises:
            RuntimeError: If the ONNX model cannot be loaded.
        """
        self.model_path = model_path
        self.threshold = threshold
        self.window_size = window_size
        self.sample_rate = sample_rate

        # Create or use provided feature extractor
        self.feature_extractor = feature_extractor or FeatureExtractor(
            sample_rate=sample_rate
        )

        # Initialize ONNX runtime session
        self._session = rt.InferenceSession(model_path)

        # Get input/output names
        self._input_name = self._session.get_inputs()[0].name

        # Initialize feature buffer
        self._feature_buffer = RingBuffer(maxlen=window_size)

    def process_audio(self, audio: np.ndarray) -> bool:
        """Process an audio frame and return detection result.

        Args:
            audio: Audio frame as numpy array. Expected to be float32
                with values typically in range [-1.0, 1.0].

        Returns:
            True if wakeword was detected in this frame, False otherwise.
        """
        # Extract FBank features
        features = self.feature_extractor.extract(audio)

        # Add frames to ring buffer
        for frame in features:
            self._feature_buffer.append(frame)

        # Only run inference when buffer is full
        if len(self._feature_buffer) < self.window_size:
            return False

        # Prepare input for ONNX model
        # Shape: (1, window_size, n_features)
        input_data = self._feature_buffer.to_array()
        input_data = np.expand_dims(input_data, axis=0)

        # Run inference
        outputs = self._session.run(None, {self._input_name: input_data})

        # Get the score (assuming first output, first batch, first class)
        # WeKWS models typically output logits or probabilities
        scores = outputs[0]
        # Take the maximum score across all classes (wakeword detection)
        max_score = float(np.max(scores))

        return max_score > self.threshold

    def reset(self) -> None:
        """Reset the internal state of the wakeword detector.

        Clears the feature buffer to start fresh detection.
        """
        self._feature_buffer.clear()
