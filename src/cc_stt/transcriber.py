import sys
import numpy as np
from funasr import AutoModel

class SpeechTranscriber:
    def __init__(self, model_name: str = "paraformer-zh"):
        print("正在加载 FunASR 模型...", file=sys.stderr, flush=True)
        self.model = AutoModel(model=model_name, disable_update=True)
        print("FunASR 模型加载完成", file=sys.stderr, flush=True)
        self.hotwords: list[str] = []

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if len(audio) == 0:
            return ""

        result = self.model.generate(
            input=audio,
            hotword=" ".join(self.hotwords) if self.hotwords else None
        )

        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'text' in result[0]:
                return result[0]['text']
            elif isinstance(result[0], str):
                return result[0]

        return ""

    def update_hotwords(self, hotwords: list[str]):
        self.hotwords = hotwords
