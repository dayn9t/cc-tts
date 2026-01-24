from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    max_duration: int = 30
    silence_threshold: float = 0.01
    silence_duration: float = 2.0

@dataclass
class ModelConfig:
    name: str = "paraformer-zh"

@dataclass
class Config:
    audio: AudioConfig
    model: ModelConfig
    hotwords_file: str

    @classmethod
    def load(cls, path: str = "~/.config/cc-stt/config.json") -> "Config":
        config_path = Path(path).expanduser()

        if not config_path.exists():
            config = cls(
                audio=AudioConfig(),
                model=ModelConfig(),
                hotwords_file="~/.config/cc-stt/hotwords.txt"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps({
                "audio": {
                    "sample_rate": config.audio.sample_rate,
                    "channels": config.audio.channels,
                    "max_duration": config.audio.max_duration,
                    "silence_threshold": config.audio.silence_threshold,
                    "silence_duration": config.audio.silence_duration,
                },
                "model": {"name": config.model.name},
                "hotwords": {"file": config.hotwords_file}
            }, indent=2))
            return config

        data = json.loads(config_path.read_text())
        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            model=ModelConfig(**data.get("model", {})),
            hotwords_file=data.get("hotwords", {}).get("file", "~/.config/cc-stt/hotwords.txt")
        )
