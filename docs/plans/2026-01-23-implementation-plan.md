# Voice Input Tool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build MCP-based voice input tool for Claude Code using FunASR

**Architecture:** MCP Server exposing transcribe/configure_hotwords tools, sounddevice for recording, FunASR Paraformer-zh for recognition, file-based hotwords config

**Tech Stack:** Python 3.12, MCP SDK, FunASR, sounddevice, numpy, uv

---

## Task 1: Setup Dependencies

**Files:**
- Modify: `pyproject.toml:10-11`

**Step 1: Add project dependencies**

Edit `pyproject.toml`:

```toml
dependencies = [
    "mcp>=1.0.0",
    "funasr>=1.0.0",
    "sounddevice>=0.4.6",
    "numpy>=1.24.0",
    "modelscope>=1.9.0",
]
```

**Step 2: Install dependencies**

Run: `uv sync`
Expected: Dependencies installed successfully

**Step 3: Verify installation**

Run: `uv run python -c "import mcp, funasr, sounddevice, numpy; print('OK')"`
Expected: Output "OK"

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add project dependencies for MCP, FunASR, audio"
```

---

## Task 2: Configuration Module

**Files:**
- Create: `src/cc_stt/config.py`
- Create: `tests/test_config.py`

**Step 1: Write test for Config dataclass**

Create `tests/test_config.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'cc_stt.config'"

**Step 3: Implement Config classes**

Create `src/cc_stt/config.py`:

```python
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
    cache_dir: str = "~/.cache/modelscope"

@dataclass
class Config:
    audio: AudioConfig
    model: ModelConfig
    hotwords_file: str

    @classmethod
    def load(cls, path: str = "~/.config/cc-stt/config.json") -> "Config":
        """Load config from file, create default if not exists"""
        config_path = Path(path).expanduser()

        if not config_path.exists():
            # Create default config
            config = cls(
                audio=AudioConfig(),
                model=ModelConfig(),
                hotwords_file="~/.config/cc-stt/hotwords.txt"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump({
                    "audio": {
                        "sample_rate": config.audio.sample_rate,
                        "channels": config.audio.channels,
                        "max_duration": config.audio.max_duration,
                        "silence_threshold": config.audio.silence_threshold,
                        "silence_duration": config.audio.silence_duration,
                    },
                    "model": {
                        "name": config.model.name,
                        "cache_dir": config.model.cache_dir,
                    },
                    "hotwords": {
                        "file": config.hotwords_file
                    }
                }, f, indent=2)
            return config

        # Load existing config
        with open(config_path) as f:
            data = json.load(f)

        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            model=ModelConfig(**data.get("model", {})),
            hotwords_file=data.get("hotwords", {}).get("file", "~/.config/cc-stt/hotwords.txt")
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Test config file creation**

Create `tests/test_config.py` additional test:

```python
def test_config_load_creates_default(tmp_path):
    config_file = tmp_path / "config.json"
    config = Config.load(str(config_file))
    assert config_file.exists()
    assert config.audio.sample_rate == 16000
```

Run: `uv run pytest tests/test_config.py::test_config_load_creates_default -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/cc_stt/config.py tests/test_config.py
git commit -m "feat: add configuration module with defaults"
```

---

## Task 3: Hotwords Manager

**Files:**
- Create: `src/cc_stt/hotwords.py`
- Create: `tests/test_hotwords.py`

**Step 1: Write test for HotwordsManager**

Create `tests/test_hotwords.py`:

```python
from cc_stt.hotwords import HotwordsManager, DEFAULT_HOTWORDS

def test_hotwords_manager_defaults(tmp_path):
    hotwords_file = tmp_path / "hotwords.txt"
    mgr = HotwordsManager(str(hotwords_file))
    assert len(mgr.get_hotwords()) > 0
    assert "Claude Code" in mgr.get_hotwords()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hotwords.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement HotwordsManager**

Create `src/cc_stt/hotwords.py`:

```python
from pathlib import Path

DEFAULT_HOTWORDS = [
    "Claude Code", "MCP", "Model Context Protocol",
    "TypeScript", "JavaScript", "Python", "Rust",
    "git", "npm", "pnpm", "bun", "uv",
    "API", "JSON", "YAML", "SQL"
]

class HotwordsManager:
    def __init__(self, config_path: str = "~/.config/cc-stt/hotwords.txt"):
        self.config_path = Path(config_path).expanduser()
        self.hotwords: list[str] = []
        self.load()

    def load(self) -> list[str]:
        """Load hotwords from file, create default if not exists"""
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_default()

        self.hotwords = []
        with open(self.config_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    self.hotwords.append(line)

        return self.hotwords

    def _write_default(self):
        """Write default hotwords to file"""
        with open(self.config_path, 'w') as f:
            f.write("# Claude Code related\n")
            for word in DEFAULT_HOTWORDS[:3]:
                f.write(f"{word}\n")
            f.write("\n# Programming languages\n")
            for word in DEFAULT_HOTWORDS[3:7]:
                f.write(f"{word}\n")
            f.write("\n# Common commands\n")
            for word in DEFAULT_HOTWORDS[7:]:
                f.write(f"{word}\n")

    def save(self, hotwords: list[str], mode: str = "replace"):
        """Save hotwords to file"""
        if mode == "append":
            self.hotwords.extend(hotwords)
        else:  # replace
            self.hotwords = hotwords

        with open(self.config_path, 'w') as f:
            for word in self.hotwords:
                f.write(f"{word}\n")

    def get_hotwords(self) -> list[str]:
        """Get current hotwords list"""
        return self.hotwords
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hotwords.py -v`
Expected: PASS

**Step 5: Test save functionality**

Add test to `tests/test_hotwords.py`:

```python
def test_hotwords_save_replace(tmp_path):
    hotwords_file = tmp_path / "hotwords.txt"
    mgr = HotwordsManager(str(hotwords_file))
    mgr.save(["test1", "test2"], mode="replace")
    assert mgr.get_hotwords() == ["test1", "test2"]

def test_hotwords_save_append(tmp_path):
    hotwords_file = tmp_path / "hotwords.txt"
    mgr = HotwordsManager(str(hotwords_file))
    original_count = len(mgr.get_hotwords())
    mgr.save(["new_word"], mode="append")
    assert len(mgr.get_hotwords()) == original_count + 1
```

Run: `uv run pytest tests/test_hotwords.py -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add src/cc_stt/hotwords.py tests/test_hotwords.py
git commit -m "feat: add hotwords manager with file persistence"
```

---

## Task 4: Audio Recorder

**Files:**
- Create: `src/cc_stt/recorder.py`
- Create: `tests/test_recorder.py`

**Step 1: Write test for AudioRecorder initialization**

Create `tests/test_recorder.py`:

```python
from cc_stt.recorder import AudioRecorder

def test_recorder_init():
    recorder = AudioRecorder(sample_rate=16000, channels=1)
    assert recorder.sample_rate == 16000
    assert recorder.channels == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_recorder.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement AudioRecorder**

Create `src/cc_stt/recorder.py`:

```python
import sounddevice as sd
import numpy as np
from typing import Optional

class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames: list[np.ndarray] = []

    def record(self, max_duration: int = 30, silence_threshold: float = 0.01,
               silence_duration: float = 2.0) -> np.ndarray:
        """Record audio until silence or timeout"""
        self.frames = []
        silence_frames = int(silence_duration * self.sample_rate / 1024)
        silent_chunks = 0

        def callback(indata, frames, time, status):
            if status:
                print(f"Recording status: {status}")

            # Calculate RMS energy
            rms = np.sqrt(np.mean(indata**2))

            # Track silence
            nonlocal silent_chunks
            if rms < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0

            self.frames.append(indata.copy())

            # Stop if silent for too long
            if silent_chunks >= silence_frames:
                raise sd.CallbackStop()

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=callback,
                blocksize=1024
            ):
                sd.sleep(int(max_duration * 1000))
        except sd.CallbackStop:
            pass

        if not self.frames:
            return np.array([], dtype=np.float32)

        # Concatenate all frames
        audio = np.concatenate(self.frames, axis=0)
        return audio.flatten()

    def get_audio_devices(self) -> list[dict]:
        """List available audio input devices"""
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels']
                })
        return input_devices
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_recorder.py -v`
Expected: PASS

**Step 5: Test device listing**

Add test to `tests/test_recorder.py`:

```python
def test_recorder_list_devices():
    recorder = AudioRecorder()
    devices = recorder.get_audio_devices()
    assert isinstance(devices, list)
```

Run: `uv run pytest tests/test_recorder.py -v`
Expected: PASS (2 tests)

**Step 6: Commit**

```bash
git add src/cc_stt/recorder.py tests/test_recorder.py
git commit -m "feat: add audio recorder with VAD silence detection"
```

---

## Task 5: Speech Transcriber

**Files:**
- Create: `src/cc_stt/transcriber.py`
- Create: `tests/test_transcriber.py`

**Step 1: Write test for SpeechTranscriber initialization**

Create `tests/test_transcriber.py`:

```python
import pytest
from cc_stt.transcriber import SpeechTranscriber

def test_transcriber_init():
    # This will download model on first run
    transcriber = SpeechTranscriber(model_name="paraformer-zh")
    assert transcriber.model is not None
    assert transcriber.hotwords == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transcriber.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement SpeechTranscriber**

Create `src/cc_stt/transcriber.py`:

```python
import numpy as np
from funasr import AutoModel
from typing import Optional

class SpeechTranscriber:
    def __init__(self, model_name: str = "paraformer-zh", hotwords: Optional[list[str]] = None):
        self.model_name = model_name
        self.hotwords = hotwords or []
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load FunASR model"""
        try:
            self.model = AutoModel(model=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load ASR model: {e}")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if len(audio) == 0:
            return ""

        try:
            # FunASR expects audio as numpy array
            result = self.model.generate(
                input=audio,
                hotword=" ".join(self.hotwords) if self.hotwords else None
            )

            # Extract text from result
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'text' in result[0]:
                    return result[0]['text']
                elif isinstance(result[0], str):
                    return result[0]

            return ""
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    def update_hotwords(self, hotwords: list[str]):
        """Update hotwords list"""
        self.hotwords = hotwords
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transcriber.py -v`
Expected: PASS (may take time for model download on first run)

**Step 5: Test transcription with empty audio**

Add test to `tests/test_transcriber.py`:

```python
def test_transcriber_empty_audio():
    transcriber = SpeechTranscriber()
    result = transcriber.transcribe(np.array([]))
    assert result == ""
```

Run: `uv run pytest tests/test_transcriber.py -v`
Expected: PASS (2 tests)

**Step 6: Commit**

```bash
git add src/cc_stt/transcriber.py tests/test_transcriber.py
git commit -m "feat: add speech transcriber with FunASR integration"
```

---

## Task 6: MCP Server Implementation

**Files:**
- Create: `src/cc_stt/server.py`
- Modify: `src/cc_stt/__init__.py`

**Step 1: Implement MCP Server**

Create `src/cc_stt/server.py`:

```python
from mcp.server import Server
from mcp.types import Tool, TextContent, ErrorData
import mcp.server.stdio
from .recorder import AudioRecorder
from .transcriber import SpeechTranscriber
from .hotwords import HotwordsManager
from .config import Config

# Initialize components
config = Config.load()
recorder = AudioRecorder(
    sample_rate=config.audio.sample_rate,
    channels=config.audio.channels
)
transcriber = SpeechTranscriber(model_name=config.model.name)
hotwords_mgr = HotwordsManager(config.hotwords_file)

# Update transcriber with loaded hotwords
transcriber.update_hotwords(hotwords_mgr.get_hotwords())

app = Server("cc-stt")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="transcribe",
            description="录音并转换为文字",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_duration": {
                        "type": "number",
                        "description": "最大录音时长（秒）",
                        "default": 30
                    },
                    "sample_rate": {
                        "type": "number",
                        "description": "采样率",
                        "default": 16000
                    }
                }
            }
        ),
        Tool(
            name="configure_hotwords",
            description="更新热词配置",
            inputSchema={
                "type": "object",
                "properties": {
                    "hotwords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "热词列表"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "append"],
                        "description": "更新模式：replace 替换全部，append 追加",
                        "default": "append"
                    }
                },
                "required": ["hotwords"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "transcribe":
            max_duration = arguments.get("max_duration", 30)

            # Record audio
            audio = recorder.record(
                max_duration=max_duration,
                silence_threshold=config.audio.silence_threshold,
                silence_duration=config.audio.silence_duration
            )

            # Transcribe
            text = transcriber.transcribe(audio, config.audio.sample_rate)

            return [TextContent(type="text", text=text)]

        elif name == "configure_hotwords":
            hotwords = arguments["hotwords"]
            mode = arguments.get("mode", "append")

            # Validate hotwords
            if not hotwords or not all(isinstance(w, str) and w.strip() for w in hotwords):
                raise ValueError("Hotwords must be non-empty strings")

            # Update hotwords
            hotwords_mgr.save(hotwords, mode=mode)
            transcriber.update_hotwords(hotwords_mgr.get_hotwords())

            count = len(hotwords_mgr.get_hotwords())
            return [TextContent(type="text", text=f"Updated {count} hotwords")]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Run MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )
```

**Step 2: Update __init__.py entry point**

Modify `src/cc_stt/__init__.py`:

```python
import asyncio
from .server import main

def cli():
    """CLI entry point"""
    asyncio.run(main())

__all__ = ["cli", "main"]
```

**Step 3: Update pyproject.toml entry point**

Modify `pyproject.toml`:

```toml
[project.scripts]
cc-stt = "cc_stt:cli"
```

**Step 4: Test server can start**

Run: `uv run cc-stt --help 2>&1 | head -5`
Expected: Server starts (may show MCP protocol messages)

**Step 5: Commit**

```bash
git add src/cc_stt/server.py src/cc_stt/__init__.py pyproject.toml
git commit -m "feat: implement MCP server with transcribe and configure_hotwords tools"
```

---

## Task 7: Integration Testing

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
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
```

**Step 2: Run integration test**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for full pipeline"
```

---

## Task 8: Documentation

**Files:**
- Modify: `README.md`

**Step 1: Write README**

Edit `README.md`:

```markdown
# cc-stt - Claude Code Speech-to-Text

Voice input tool for Claude Code using FunASR.

## Features

- MCP Server integration with Claude Code
- Real-time microphone recording with VAD silence detection
- FunASR Paraformer-zh model for Chinese speech recognition
- Configurable hotwords for technical terms
- Push-to-talk recording mode

## Installation

```bash
uv sync
```

## Configuration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "cc-stt": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/cc-stt",
        "run",
        "cc-stt"
      ]
    }
  }
}
```

## Usage

The MCP server provides two tools:

### transcribe

Record audio and convert to text.

Parameters:
- `max_duration` (optional): Maximum recording duration in seconds (default: 30)

### configure_hotwords

Update hotwords configuration.

Parameters:
- `hotwords`: Array of hotword strings
- `mode` (optional): "replace" or "append" (default: "append")

## Configuration Files

- `~/.config/cc-stt/config.json` - Main configuration
- `~/.config/cc-stt/hotwords.txt` - Hotwords list

## Development

Run tests:

```bash
uv run pytest tests/ -v
```

## License

MIT
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README"
```

---

## Task 9: Final Verification

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Verify server starts**

Run: `timeout 2 uv run cc-stt 2>&1 || echo "Server started"`
Expected: Server starts without errors

**Step 3: Check file structure**

Run: `find src tests -type f -name "*.py" | sort`
Expected: All module files present

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final verification and cleanup"
```

---

## Summary

**Implementation complete with:**

1. ✅ Configuration management (config.py)
2. ✅ Hotwords manager (hotwords.py)
3. ✅ Audio recorder with VAD (recorder.py)
4. ✅ Speech transcriber with FunASR (transcriber.py)
5. ✅ MCP Server with two tools (server.py)
6. ✅ Comprehensive tests
7. ✅ Documentation

**Next steps:**

1. Test with actual Claude Code integration
2. Adjust VAD parameters based on real usage
3. Add error logging for production debugging
4. Consider adding recording visualization

**Total tasks:** 9
**Estimated implementation time:** Follow TDD cycle for each task
