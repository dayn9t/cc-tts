"""Factory for creating wakeword backends."""

from typing import Literal

from cc_stt.wakeword.base import WakewordBackend
from cc_stt.wakeword.openwakeword import OpenWakeWordBackend
from cc_stt.wakeword.wekws import WeKWSBackend


def create_wakeword_backend(
    backend: Literal["openwakeword", "wekws"],
    name: str,
    threshold: float = 0.5,
    model_path: str | None = None,
    window_size: int = 40,
) -> WakewordBackend:
    """Create a wakeword backend instance.

    Args:
        backend: Type of backend to create ("openwakeword" or "wekws").
        name: Name of the wakeword model to use.
        threshold: Detection threshold (0.0 to 1.0). Higher values require
            stronger confidence for detection.
        model_path: Path to the ONNX model file. Required for "wekws" backend.
        window_size: Number of feature frames to accumulate before running
            inference. Used only by "wekws" backend.

    Returns:
        An instance of the requested wakeword backend.

    Raises:
        ValueError: If backend is unknown or if model_path is required but not provided.
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
    else:
        raise ValueError(f"Unknown backend: {backend}")
