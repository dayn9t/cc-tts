"""Sherpa-ONNX KWS backend implementation for wakeword detection."""

import os
import tempfile
from pathlib import Path

import numpy as np

# Handle sherpa_onnx import gracefully (optional dependency)
try:
    from sherpa_onnx import KeywordSpotter, OnlineTransducerModelConfig

    HAS_SHERPA_ONNX = True
except ImportError:
    HAS_SHERPA_ONNX = False
    KeywordSpotter = None
    OnlineTransducerModelConfig = None


class SherpaONNXBackend:
    """Wakeword detection backend using Sherpa-ONNX (Chinese/English bilingual).

    Supports pre-trained models from k2-fsa project.
    Can define custom keywords without retraining the model.

    Args:
        model_dir: Directory containing encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt
        keywords: List of keywords to detect (e.g., ["小爱同学", "小度小度"])
        keywords_file: Path to keywords.txt file (alternative to keywords list)
        num_threads: Number of threads for ONNX inference (default: 4)
        provider: ONNX execution provider ("cpu" or "cuda", default: "cpu")

    Raises:
        FileNotFoundError: If required model files are missing.
        ValueError: If neither keywords nor keywords_file is provided.
        RuntimeError: If sherpa_onnx package is not installed.
    """

    def __init__(
        self,
        model_dir: str,
        keywords: list[str] | None = None,
        keywords_file: str | None = None,
        num_threads: int = 4,
        provider: str = "cpu",
    ) -> None:
        """Initialize the SherpaONNX backend.

        Args:
            model_dir: Directory containing model files.
            keywords: List of keywords to detect.
            keywords_file: Path to keywords file.
            num_threads: Number of threads for ONNX inference.
            provider: ONNX execution provider.

        Raises:
            RuntimeError: If sherpa_onnx is not installed.
            FileNotFoundError: If model files are missing.
            ValueError: If keywords configuration is invalid.
        """
        if not HAS_SHERPA_ONNX:
            raise RuntimeError(
                "sherpa_onnx package is required for SherpaONNXBackend. "
                "Install with: pip install sherpa-onnx"
            )

        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.provider = provider
        self.keywords_file = keywords_file

        # Validate model directory
        if not self._check_model_files():
            raise FileNotFoundError(
                f"Sherpa-ONNX model files not found in {model_dir}. "
                f"Expected: encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt"
            )

        # Create keywords file if list provided
        if keywords and not keywords_file:
            keywords_file = self._create_keywords_file(keywords)

        if not keywords_file:
            raise ValueError("Either keywords or keywords_file must be provided")

        self.keywords_file = keywords_file

        # Initialize KeywordSpotter
        self.spotter = KeywordSpotter(
            encoder_config=OnlineTransducerModelConfig(
                encoder=str(self.model_dir / "encoder.onnx"),
                decoder=str(self.model_dir / "decoder.onnx"),
                joiner=str(self.model_dir / "joiner.onnx"),
            ),
            tokens=str(self.model_dir / "tokens.txt"),
            keywords_file=keywords_file,
            num_threads=num_threads,
            provider=provider,
        )

        # Create initial stream
        self._stream = self.spotter.create_stream()

    def _check_model_files(self) -> bool:
        """Check if all required model files exist.

        Returns:
            True if all required files exist, False otherwise.
        """
        required_files = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
        return all((self.model_dir / f).exists() for f in required_files)

    def _create_keywords_file(self, keywords: list[str]) -> str:
        """Create a keywords.txt file from keyword list.

        Sherpa-ONNX expects keywords in format:
        keyword_name phoneme_sequence

        For Chinese, we use the characters themselves as a simple approximation.
        In production, proper pinyin conversion would be better.

        Args:
            keywords: List of keywords to convert.

        Returns:
            Path to the created keywords file.
        """
        # Create a temporary file for keywords
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            for keyword in keywords:
                # Simple approach: use space-separated characters as phonemes
                # In production, proper pinyin conversion would be better
                phonemes = " ".join(keyword)
                f.write(f"{keyword} {phonemes}\n")
            return f.name

    def process_audio(self, audio: np.ndarray) -> bool:
        """Process an audio frame and return detection result.

        Args:
            audio: Audio frame as numpy array. Expected to be float32
                with values typically in range [-1.0, 1.0].

        Returns:
            True if wakeword was detected in this frame, False otherwise.
        """
        self._stream.accept_waveform(16000, audio)
        result = self.spotter.get_result(self._stream)
        return result is not None and result != ""

    def reset(self) -> None:
        """Reset the internal state of the wakeword detector.

        Creates a new stream for fresh detection.
        """
        self._stream = self.spotter.create_stream()
