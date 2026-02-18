"""Base Protocol for wakeword detection backends."""

from typing import Protocol

import numpy as np


class WakewordBackend(Protocol):
    """Protocol defining the interface for wakeword detection backends.

    Implementations of this protocol provide wakeword detection capabilities
    using different underlying engines (e.g., wekws, openwakeword, etc.).
    """

    def process_audio(self, audio: np.ndarray) -> bool:
        """Process an audio frame and return detection result.

        Args:
            audio: Audio frame as numpy array. Expected to be float32
                with values typically in range [-1.0, 1.0].

        Returns:
            True if wakeword was detected in this frame, False otherwise.
        """
        ...

    def reset(self) -> None:
        """Reset the internal state of the wakeword detector.

        This should be called when starting a new detection session or
        after a wakeword has been detected and processed.
        """
        ...
