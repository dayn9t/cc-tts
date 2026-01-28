import numpy as np
import openwakeword
from openwakeword.model import Model


class WakewordDetector:
    def __init__(self, wakeword: str = "alexa", threshold: float = 0.5):
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
        """处理音频帧，返回是否检测到唤醒词"""
        prediction = self.model.predict(audio)
        score = prediction.get(self.model_key, 0)
        if score > self.threshold:
            self.model.reset()
            return True
        return False

    def reset(self):
        """重置检测状态"""
        self.model.reset()
