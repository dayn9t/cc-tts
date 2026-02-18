from dataclasses import dataclass
from pathlib import Path
from typing import Literal
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
class WakewordConfig:
    # Backend selection
    backend: Literal["openwakeword", "wekws", "sherpa-onnx"] = "openwakeword"

    # Common settings
    name: str = "alexa"
    threshold: float = 0.3
    gain: float = 2.0

    # WeKWS-specific settings
    model_path: str | None = None
    window_size: int = 40

    # Sherpa-ONNX-specific settings
    sherpa_model_dir: str | None = None
    sherpa_keywords: list[str] | None = None
    sherpa_keywords_file: str | None = None
    sherpa_num_threads: int = 4

@dataclass
class Config:
    audio: AudioConfig
    model: ModelConfig
    wakeword: WakewordConfig
    hotwords_file: str

    @classmethod
    def load(cls, path: str = "~/.config/cc-stt/config.json") -> "Config":
        config_path = Path(path).expanduser()

        if not config_path.exists():
            config = cls(
                audio=AudioConfig(),
                model=ModelConfig(),
                wakeword=WakewordConfig(),
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
                "wakeword": {
                    "backend": config.wakeword.backend,
                    "name": config.wakeword.name,
                    "threshold": config.wakeword.threshold,
                    "gain": config.wakeword.gain,
                    "model_path": config.wakeword.model_path,
                    "window_size": config.wakeword.window_size,
                    "sherpa_model_dir": config.wakeword.sherpa_model_dir,
                    "sherpa_keywords": config.wakeword.sherpa_keywords,
                    "sherpa_keywords_file": config.wakeword.sherpa_keywords_file,
                    "sherpa_num_threads": config.wakeword.sherpa_num_threads,
                },
                "hotwords": {"file": config.hotwords_file}
            }, indent=2))
            return config

        data = json.loads(config_path.read_text())
        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            model=ModelConfig(**data.get("model", {})),
            wakeword=WakewordConfig(**data.get("wakeword", {})),
            hotwords_file=data.get("hotwords", {}).get("file", "~/.config/cc-stt/hotwords.txt")
        )
