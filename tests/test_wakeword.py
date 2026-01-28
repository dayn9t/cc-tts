import numpy as np
from cc_stt.wakeword import WakewordDetector


def test_wakeword_detector_init():
    detector = WakewordDetector(wakeword="alexa")
    assert detector.wakeword == "alexa"
    assert detector.model is not None


def test_wakeword_detector_process_audio():
    detector = WakewordDetector(wakeword="alexa")
    # 静音音频，不应触发
    silence = np.zeros(1280, dtype=np.float32)
    result = detector.process_audio(silence)
    assert result is False
