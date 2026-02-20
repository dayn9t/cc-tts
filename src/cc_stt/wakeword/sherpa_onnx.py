"""Sherpa-ONNX backend implementation for wakeword detection."""

import os
import tempfile
from typing import List

import numpy as np

# Import sherpa_onnx - may not be installed in test environment
try:
    from sherpa_onnx import KeywordSpotter
except ImportError:
    KeywordSpotter = None  # type: ignore


class SherpaONNXBackend:
    """Wakeword detection backend using Sherpa-ONNX KeywordSpotter.

    This class provides wakeword detection functionality using Sherpa-ONNX
    streaming transducer models with keyword spotting capability.

    Required model files in model_dir:
        - encoder.onnx: Encoder model
        - decoder.onnx: Decoder model
        - joiner.onnx: Joiner model
        - tokens.txt: Token list
    """

    REQUIRED_PATTERNS = {
        "encoder": "encoder*.onnx",
        "decoder": "decoder*.onnx",
        "joiner": "joiner*.onnx",
        "tokens": "tokens.txt",
    }

    def __init__(
        self,
        model_dir: str,
        keywords: List[str] | None = None,
        keywords_file: str | None = None,
        num_threads: int = 4,
        provider: str = "cpu",
        name: str = "sherpa-onnx-kws",
    ):
        """Initialize the Sherpa-ONNX backend.

        Args:
            model_dir: Directory containing the model files.
            keywords: List of keywords to detect (e.g., ["hello", "world"]).
                Either keywords or keywords_file must be provided.
            keywords_file: Path to a keywords file. Either keywords or
                keywords_file must be provided.
            num_threads: Number of threads for ONNX Runtime.
            provider: ONNX Runtime provider ("cpu" or "cuda").
            name: Display name for the wake word (e.g., "小狗小狗").

        Raises:
            FileNotFoundError: If required model files are missing.
            ValueError: If neither keywords nor keywords_file is provided.
            RuntimeError: If sherpa_onnx is not installed.
        """
        self.wakeword = name  # Display name for daemon
        self.model_dir = model_dir
        self.num_threads = num_threads
        self.provider = provider
        self._temp_keywords_file: str | None = None
        self._model_files: dict[str, str] = {}

        # Check if sherpa_onnx is available
        if KeywordSpotter is None:
            raise RuntimeError(
                "sherpa_onnx is not installed. "
                "Install it with: pip install sherpa-onnx"
            )

        # Validate model files exist and find paths
        self._model_files = self._find_model_files()

        # Determine keywords file path
        if keywords_file:
            self.keywords_file = keywords_file
        elif keywords:
            self.keywords_file = self._create_keywords_file(keywords)
            self._temp_keywords_file = self.keywords_file
        else:
            raise ValueError("Either keywords or keywords_file must be provided")

        # Initialize the KeywordSpotter
        self._spotter = self._create_spotter()
        self._stream = None

    def _find_model_files(self) -> dict[str, str]:
        """Find required model files in model_dir.

        Supports pattern matching for models with version numbers
        (e.g., encoder-epoch-12-avg-2-chunk-16-left-64.onnx).

        Returns:
            Dictionary mapping file type to full path.

        Raises:
            FileNotFoundError: If any required file is missing.
        """
        import fnmatch

        model_files = {}
        dir_files = os.listdir(self.model_dir)

        for file_type, pattern in self.REQUIRED_PATTERNS.items():
            matched = None
            for filename in dir_files:
                if fnmatch.fnmatch(filename, pattern):
                    # Prefer non-int8 version if available
                    if ".int8." not in filename:
                        matched = filename
                        break
                    elif matched is None:
                        matched = filename

            if matched is None:
                raise FileNotFoundError(
                    f"Model file not found for pattern: {pattern}. "
                    f"Required patterns: {self.REQUIRED_PATTERNS}"
                )

            model_files[file_type] = os.path.join(self.model_dir, matched)

        return model_files

    def _create_keywords_file(self, keywords: List[str]) -> str:
        """Create a temporary keywords file from a list of keywords.

        Sherpa-ONNX keywords file format:
        keyword1 : boost=1.0
        keyword2 : boost=1.0

        Args:
            keywords: List of keywords to write to the file.

        Returns:
            Path to the created keywords file.
        """
        # Create a temporary file
        fd, path = tempfile.mkstemp(suffix=".txt", prefix="keywords_")
        try:
            with os.fdopen(fd, "w") as f:
                for keyword in keywords:
                    # Default boost value of 1.0
                    f.write(f"{keyword} : boost=1.0\n")
        except:
            os.unlink(path)
            raise
        return path

    def _create_spotter(self) -> "KeywordSpotter":
        """Create and configure the KeywordSpotter instance.

        Returns:
            Configured KeywordSpotter instance.
        """
        spotter = KeywordSpotter(
            tokens=self._model_files["tokens"],
            encoder=self._model_files["encoder"],
            decoder=self._model_files["decoder"],
            joiner=self._model_files["joiner"],
            num_threads=self.num_threads,
            provider=self.provider,
            keywords_file=self.keywords_file,
        )
        return spotter

    def process_audio(self, audio: np.ndarray) -> bool:
        """Process an audio frame and return detection result.

        Args:
            audio: Audio frame as numpy array. Expected to be float32
                with values typically in range [-1.0, 1.0].

        Returns:
            True if wakeword was detected in this frame, False otherwise.
        """
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Create stream if not exists
        if self._stream is None:
            self._stream = self._spotter.create_stream()

        # Accept waveform
        self._stream.accept_waveform(16000, audio)

        # Process and check for detections (limit iterations to prevent CPU spike)
        max_iterations = 10
        iterations = 0
        while self._spotter.is_ready(self._stream) and iterations < max_iterations:
            self._spotter.decode_stream(self._stream)
            iterations += 1

        # Check for keyword detections
        result = self._spotter.get_result(self._stream)
        for detection in result:
            if hasattr(detection, "keyword") and detection.keyword:
                # Reset after detection
                self.reset()
                return True

        return False

    def reset(self) -> None:
        """Reset the internal state of the wakeword detector.

        This should be called when starting a new detection session or
        after a wakeword has been detected and processed.
        """
        if self._stream is not None:
            self._spotter.reset_stream(self._stream)
            self._stream = None

    def __del__(self):
        """Cleanup temporary files on destruction."""
        if self._temp_keywords_file and os.path.exists(self._temp_keywords_file):
            try:
                os.unlink(self._temp_keywords_file)
            except OSError:
                pass  # Ignore cleanup errors
