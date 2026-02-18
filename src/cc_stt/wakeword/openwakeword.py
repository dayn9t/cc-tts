"""OpenWakeWord backend implementation for wakeword detection."""

import numpy as np
import openwakeword
from openwakeword.model import Model


class OpenWakeWordBackend:
    """Wakeword detection backend using OpenWakeWord library.

    This class provides wakeword detection functionality using pretrained
    models from the OpenWakeWord library.
    """

    def __init__(self, wakeword: str = "alexa", threshold: float = 0.5):
        """Initialize the OpenWakeWord backend.

        Args:
            wakeword: Name of the wakeword model to use (e.g., "alexa").
            threshold: Detection threshold (0.0 to 1.0). Higher values require
                stronger confidence for detection.

        Raises:
            ValueError: If the specified wakeword model is not found.
        """
        self.wakeword = wakeword
        self.threshold = threshold

        # Get the full path to the pretrained model
        model_paths = openwakeword.get_pretrained_model_paths()
        matching_model = [p for p in model_paths if wakeword in p]

        if not matching_model:
            raise ValueError(f"Model '{wakeword}' not found in pretrained models")

        self.model = Model(wakeword_model_paths=[matching_model[0]])
        # Store the actual model key (e.g., 'alexa_v0.1')
        self.model_key = list(self.model.models.keys())[0]

    def process_audio(self, audio: np.ndarray) -> bool:
        """Process audio frame and detect wakeword.

        Args:
            audio: Audio frame as numpy array.

        Returns:
            True if wakeword is detected, False otherwise.
        """
        prediction = self.model.predict(audio)
        score = prediction.get(self.model_key, 0)
        if score > self.threshold:
            self.model.reset()
            return True
        return False

    def reset(self) -> None:
        """Reset detection state."""
        self.model.reset()
