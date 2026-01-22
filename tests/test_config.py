from cc_stt.config import Config, AudioConfig, ModelConfig

def test_config_defaults():
    config = Config(
        audio=AudioConfig(),
        model=ModelConfig(),
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
