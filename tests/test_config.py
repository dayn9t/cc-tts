from cc_stt.config import Config, AudioConfig, ModelConfig, WakewordConfig


def test_config_defaults():
    config = Config(
        audio=AudioConfig(),
        model=ModelConfig(),
        wakeword=WakewordConfig(),
        hotwords_file="~/.config/cc-stt/hotwords.txt"
    )
    assert config.audio.sample_rate == 16000
    assert config.audio.max_duration == 30
    assert config.model.name == "paraformer-zh"


def test_config_load_creates_default(tmp_path):
    config_file = tmp_path / "config.json"
    config = Config.load(str(config_file))
    assert config_file.exists()
    assert config.audio.sample_rate == 16000


def test_wakeword_config_defaults():
    """Verify default backend is 'openwakeword'"""
    config = WakewordConfig()
    assert config.backend == "openwakeword"
    assert config.name == "alexa"
    assert config.threshold == 0.3
    assert config.gain == 2.0
    assert config.model_path is None
    assert config.window_size == 40


def test_wakeword_config_wekws_values():
    """Test WeKWS config with custom values"""
    config = WakewordConfig(
        backend="wekws",
        name="custom_wakeword",
        threshold=0.5,
        gain=1.5,
        model_path="/path/to/model.onnx",
        window_size=50
    )
    assert config.backend == "wekws"
    assert config.name == "custom_wakeword"
    assert config.threshold == 0.5
    assert config.gain == 1.5
    assert config.model_path == "/path/to/model.onnx"
    assert config.window_size == 50


def test_config_load_default(tmp_path):
    """Verify Config.load() creates correct defaults"""
    config_file = tmp_path / "config.json"
    config = Config.load(str(config_file))

    # Check wakeword defaults
    assert config.wakeword.backend == "openwakeword"
    assert config.wakeword.name == "alexa"
    assert config.wakeword.threshold == 0.3
    assert config.wakeword.gain == 2.0
    assert config.wakeword.model_path is None
    assert config.wakeword.window_size == 40


def test_config_load_with_wakeword_values(tmp_path):
    """Verify Config.load() parses wakeword values from file"""
    import json

    config_file = tmp_path / "config.json"
    config_data = {
        "audio": {"sample_rate": 16000},
        "model": {"name": "paraformer-zh"},
        "wakeword": {
            "backend": "wekws",
            "name": "hey_jarvis",
            "threshold": 0.6,
            "gain": 3.0,
            "model_path": "/models/wekws.onnx",
            "window_size": 45
        },
        "hotwords": {"file": "~/.config/cc-stt/hotwords.txt"}
    }
    config_file.write_text(json.dumps(config_data))

    config = Config.load(str(config_file))
    assert config.wakeword.backend == "wekws"
    assert config.wakeword.name == "hey_jarvis"
    assert config.wakeword.threshold == 0.6
    assert config.wakeword.gain == 3.0
    assert config.wakeword.model_path == "/models/wekws.onnx"
    assert config.wakeword.window_size == 45


def test_sherpa_onnx_config_defaults():
    """Verify Sherpa-ONNX config defaults"""
    config = WakewordConfig()
    assert config.sherpa_model_dir is None
    assert config.sherpa_keywords is None
    assert config.sherpa_keywords_file is None
    assert config.sherpa_num_threads == 4


def test_sherpa_onnx_config_custom_values():
    """Test Sherpa-ONNX config with custom values"""
    config = WakewordConfig(
        backend="sherpa-onnx",
        sherpa_model_dir="/models/sherpa",
        sherpa_keywords=["hey computer", "ok device"],
        sherpa_keywords_file="/config/keywords.txt",
        sherpa_num_threads=8
    )
    assert config.backend == "sherpa-onnx"
    assert config.sherpa_model_dir == "/models/sherpa"
    assert config.sherpa_keywords == ["hey computer", "ok device"]
    assert config.sherpa_keywords_file == "/config/keywords.txt"
    assert config.sherpa_num_threads == 8
