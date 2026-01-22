import numpy as np
from funasr import AutoModel
from typing import Optional

class SpeechTranscriber:
    def __init__(self, model_name: str = "paraformer-zh", hotwords: Optional[list[str]] = None):
        self.model_name = model_name
        self.hotwords = hotwords or []
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load FunASR model"""
        try:
            self.model = AutoModel(model=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load ASR model: {e}")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if len(audio) == 0:
            return ""

        try:
            # FunASR expects audio as numpy array
            result = self.model.generate(
                input=audio,
                hotword=" ".join(self.hotwords) if self.hotwords else None
            )

            # Extract text from result
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'text' in result[0]:
                    return result[0]['text']
                elif isinstance(result[0], str):
                    return result[0]

            return ""
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    def update_hotwords(self, hotwords: list[str]):
        """Update hotwords list"""
        self.hotwords = hotwords
