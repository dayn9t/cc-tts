"""Factory for creating wakeword backends."""

from typing import List, Literal

from cc_stt.wakeword.base import WakewordBackend
from cc_stt.wakeword.openwakeword import OpenWakeWordBackend
from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend
from cc_stt.wakeword.wekws import WeKWSBackend


def create_wakeword_backend(
    backend: Literal["openwakeword", "wekws", "sherpa-onnx"],
    name: str,
    threshold: float = 0.5,
    model_path: str | None = None,
    window_size: int = 40,
    model_dir: str | None = None,
    keywords: List[str] | None = None,
    keywords_file: str | None = None,
    num_threads: int = 4,
    provider: str = "cpu",
) -> WakewordBackend:
    """Create a wakeword backend instance.

    Args:
        backend: Type of backend to create ("openwakeword", "wekws", or "sherpa-onnx").
        name: Name of the wakeword model to use.
        threshold: Detection threshold (0.0 to 1.0). Higher values require
            stronger confidence for detection.
        model_path: Path to the ONNX model file. Required for "wekws" backend.
        window_size: Number of feature frames to accumulate before running
            inference. Used only by "wekws" backend.
        model_dir: Directory containing model files. Required for "sherpa-onnx" backend.
        keywords: List of keywords to detect. Required for "sherpa-onnx" backend
            if keywords_file is not provided.
        keywords_file: Path to keywords file. Alternative to keywords for
            "sherpa-onnx" backend.
        num_threads: Number of threads for ONNX Runtime. Used by "sherpa-onnx" backend.
        provider: ONNX Runtime provider ("cpu" or "cuda"). Used by "sherpa-onnx" backend.

    Returns:
        An instance of the requested wakeword backend.

    Raises:
        ValueError: If backend is unknown or if required parameters are not provided.
    """
    if backend == "openwakeword":
        return OpenWakeWordBackend(wakeword=name, threshold=threshold)
    elif backend == "wekws":
        if model_path is None:
            raise ValueError("model_path is required for 'wekws' backend")
        return WeKWSBackend(
            model_path=model_path,
            threshold=threshold,
            window_size=window_size,
        )
    elif backend == "sherpa-onnx":
        if model_dir is None:
            raise ValueError("model_dir is required for 'sherpa-onnx' backend")
        if keywords is None and keywords_file is None:
            raise ValueError(
                "Either keywords or keywords_file is required for 'sherpa-onnx' backend"
            )
        return SherpaONNXBackend(
            model_dir=model_dir,
            keywords=keywords,
            keywords_file=keywords_file,
            num_threads=num_threads,
            provider=provider,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
