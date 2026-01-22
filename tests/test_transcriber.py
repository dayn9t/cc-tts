import pytest
import numpy as np
from cc_stt.transcriber import SpeechTranscriber

def test_transcriber_init():
    # This will download model on first run
    transcriber = SpeechTranscriber(model_name="paraformer-zh")
    assert transcriber.model is not None
    assert transcriber.hotwords == []

def test_transcriber_empty_audio():
    transcriber = SpeechTranscriber()
    result = transcriber.transcribe(np.array([]))
    assert result == ""
