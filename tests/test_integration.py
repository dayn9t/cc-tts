import pytest
from cc_stt.config import Config
from cc_stt.hotwords import HotwordsManager
from cc_stt.recorder import AudioRecorder
from cc_stt.transcriber import SpeechTranscriber

def test_full_pipeline_components(tmp_path):
    """Test all components can be initialized together"""
    # Config
    config_file = tmp_path / "config.json"
    config = Config.load(str(config_file))
    assert config.audio.sample_rate == 16000

    # Hotwords
    hotwords_file = tmp_path / "hotwords.txt"
    hotwords_mgr = HotwordsManager(str(hotwords_file))
    assert len(hotwords_mgr.get_hotwords()) > 0

    # Recorder
    recorder = AudioRecorder(
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels
    )
    assert recorder.sample_rate == 16000

    # Transcriber
    transcriber = SpeechTranscriber()
    transcriber.update_hotwords(hotwords_mgr.get_hotwords())
    assert len(transcriber.hotwords) > 0
