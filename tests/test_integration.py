import pytest
from cc_stt.config import Config
from cc_stt.hotwords import HotwordsManager
from cc_stt.recorder import AudioRecorder
from cc_stt.transcriber import SpeechTranscriber
from cc_stt.wakeword import WakewordDetector

tkinter = pytest.importorskip("tkinter", reason="tkinter not available (headless environment)")

from cc_stt.voice_edit import VoiceEditor
from cc_stt.sender import Sender
from cc_stt.daemon import Daemon

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


def test_daemon_initialization(tmp_path):
    """Test Daemon can be initialized with all components"""
    config_file = tmp_path / "config.json"
    hotwords_file = tmp_path / "hotwords.txt"

    # Create temporary config
    import json
    config_data = {
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "max_duration": 30,
            "silence_threshold": 0.01,
            "silence_duration": 1.0
        },
        "model": {
            "name": "paraformer-zh"
        }
    }
    config_file.write_text(json.dumps(config_data))

    # Initialize daemon components
    daemon = Daemon(wakeword="alexa")

    # Verify all components are initialized
    assert daemon.config is not None
    assert daemon.wakeword is not None
    assert daemon.recorder is not None
    assert daemon.transcriber is not None
    assert daemon.voice_editor is not None
    assert daemon.sender is not None
    assert daemon.running == False

    # Test stop method
    daemon.stop()
    assert daemon.running == False


def test_voice_editor_initialization():
    """Test VoiceEditor can be initialized"""
    editor = VoiceEditor()
    assert editor is not None


def test_sender_initialization():
    """Test Sender can be initialized"""
    sender = Sender()
    assert sender is not None


def test_wakeword_detector_initialization():
    """Test WakewordDetector can be initialized"""
    detector = WakewordDetector(wakeword="alexa")
    assert detector.wakeword == "alexa"
    assert detector.models is not None


