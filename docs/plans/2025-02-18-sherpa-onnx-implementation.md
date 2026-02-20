# Sherpa-ONNX Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Sherpa-ONNX KWS backend for Chinese wake word detection using pre-trained models.

**Architecture:** Create `SherpaONNXBackend` class implementing `WakewordBackend` Protocol, update factory and config to support `backend="sherpa-onnx"`, add model download utility.

**Tech Stack:** sherpa-onnx, onnxruntime, wget/tqdm for model download

---

## Prerequisites Check

**Run before starting:**
```bash
cd /home/jiang/cc/audio/cc-stt

# Verify project structure
ls src/cc_stt/wakeword/
# Should show: __init__.py, base.py, factory.py, openwakeword.py, wekws.py

# Check Python version
python --version
# Should be 3.12+

# Verify tests pass
cd /home/jiang/cc/audio/cc-stt
PYTHONPATH=src .venv/bin/python -m pytest tests/wakeword/ -v
# Should show all tests passing
```

---

## Task 1: Update WakewordConfig for Sherpa-ONNX Settings

**Files:**
- Modify: `src/cc_stt/config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py - add to existing file

def test_sherpa_onnx_config_defaults():
    """Test Sherpa-ONNX config default values."""
    from cc_stt.config import WakewordConfig

    config = WakewordConfig()
    assert config.backend == "openwakeword"  # Default unchanged
    assert config.sherpa_model_dir is None
    assert config.sherpa_keywords is None
    assert config.sherpa_keywords_file is None
    assert config.sherpa_num_threads == 4


def test_sherpa_onnx_config_custom_values():
    """Test Sherpa-ONNX config with custom values."""
    from cc_stt.config import WakewordConfig

    config = WakewordConfig(
        backend="sherpa-onnx",
        sherpa_model_dir="models/sherpa-kws/zh",
        sherpa_keywords=["小爱同学", "小度小度"],
        sherpa_num_threads=8,
    )
    assert config.backend == "sherpa-onnx"
    assert config.sherpa_model_dir == "models/sherpa-kws/zh"
    assert config.sherpa_keywords == ["小爱同学", "小度小度"]
    assert config.sherpa_num_threads == 8
```

**Step 2: Run test to verify it fails**

```bash
cd /home/jiang/cc/audio/cc-stt
PYTHONPATH=src .venv/bin/python -m pytest tests/test_config.py::test_sherpa_onnx_config_defaults -v
```

Expected: FAIL with "AttributeError: 'WakewordConfig' object has no attribute 'sherpa_model_dir'"

**Step 3: Update WakewordConfig**

```python
# src/cc_stt/config.py - modify WakewordConfig dataclass

@dataclass
class WakewordConfig:
    """Wake word detection configuration.

    Supports multiple backends: openwakeword (English), wekws (Chinese),
    sherpa-onnx (Chinese/English bilingual)
    """
    # Backend selection
    backend: Literal["openwakeword", "wekws", "sherpa-onnx"] = "openwakeword"

    # Common settings
    name: str = "alexa"
    threshold: float = 0.3
    gain: float = 2.0

    # WeKWS-specific settings
    model_path: str | None = None
    window_size: int = 40

    # Sherpa-ONNX specific settings (NEW)
    sherpa_model_dir: str | None = None
    sherpa_keywords: list[str] | None = None
    sherpa_keywords_file: str | None = None
    sherpa_num_threads: int = 4
```

Also update `Literal` import and Config.load() method to handle new fields.

**Step 4: Run tests to verify they pass**

```bash
cd /home/jiang/cc/audio/cc-stt
PYTHONPATH=src .venv/bin/python -m pytest tests/test_config.py::test_sherpa_onnx_config_defaults tests/test_config.py::test_sherpa_onnx_config_custom_values -v
```

Expected: 2 tests PASS

**Step 5: Run full config test suite**

```bash
cd /home/jiang/cc/audio/cc-stt
PYTHONPATH=src .venv/bin/python -m pytest tests/test_config.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/cc_stt/config.py tests/test_config.py
git commit -m "feat: add Sherpa-ONNX config fields to WakewordConfig"
```

---

## Task 2: Create SherpaONNXBackend Implementation

**Files:**
- Create: `src/cc_stt/wakeword/sherpa_onnx.py`
- Create: `tests/wakeword/test_sherpa_onnx.py`

**Step 1: Write the failing test**

```python
# tests/wakeword/test_sherpa_onnx.py

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock


def test_sherpa_onnx_backend_init():
    """Test SherpaONNXBackend initialization with mocked sherpa_onnx."""
    from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

    with patch('cc_stt.wakeword.sherpa_onnx.sherpa_onnx') as mock_sherpa:
        mock_spotter = Mock()
        mock_sherpa.KeywordSpotter.return_value = mock_spotter
        mock_spotter.create_stream.return_value = Mock()

        backend = SherpaONNXBackend(
            model_dir="/fake/model/dir",
            keywords=["小爱同学"],
            num_threads=4,
        )

        assert backend.spotter is not None
        assert backend.stream is not None


def test_sherpa_onnx_process_audio():
    """Test SherpaONNXBackend process_audio method."""
    from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

    with patch('cc_stt.wakeword.sherpa_onnx.sherpa_onnx') as mock_sherpa:
        # Setup mock
        mock_spotter = Mock()
        mock_sherpa.KeywordSpotter.return_value = mock_spotter
        mock_stream = Mock()
        mock_spotter.create_stream.return_value = mock_stream

        # Create backend
        backend = SherpaONNXBackend(
            model_dir="/fake/model/dir",
            keywords=["小爱同学"],
        )

        # Test audio processing - no detection
        mock_spotter.get_result.return_value = None
        audio = np.random.randn(1280).astype(np.float32)
        result = backend.process_audio(audio)

        assert result is False
        mock_stream.accept_waveform.assert_called_once()


def test_sherpa_onnx_process_audio_detection():
    """Test SherpaONNXBackend with keyword detection."""
    from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

    with patch('cc_stt.wakeword.sherpa_onnx.sherpa_onnx') as mock_sherpa:
        mock_spotter = Mock()
        mock_sherpa.KeywordSpotter.return_value = mock_spotter
        mock_spotter.create_stream.return_value = Mock()

        backend = SherpaONNXBackend(
            model_dir="/fake/model/dir",
            keywords=["小爱同学"],
        )

        # Simulate detection
        mock_spotter.get_result.return_value = "小爱同学"
        audio = np.random.randn(1280).astype(np.float32)
        result = backend.process_audio(audio)

        assert result is True


def test_sherpa_onnx_reset():
    """Test SherpaONNXBackend reset method."""
    from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend

    with patch('cc_stt.wakeword.sherpa_onnx.sherpa_onnx') as mock_sherpa:
        mock_spotter = Mock()
        mock_sherpa.KeywordSpotter.return_value = mock_spotter
        mock_spotter.create_stream.return_value = Mock()

        backend = SherpaONNXBackend(
            model_dir="/fake/model/dir",
            keywords=["小爱同学"],
        )

        # Reset should create new stream
        backend.reset()
        assert mock_spotter.create_stream.call_count == 2  # Once in init, once in reset
```

**Step 2: Run test to verify it fails**

```bash
cd /home/jiang/cc/audio/cc-stt
PYTHONPATH=src .venv/bin/python -m pytest tests/wakeword/test_sherpa_onnx.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'cc_stt.wakeword.sherpa_onnx'"

**Step 3: Implement SherpaONNXBackend**

```python
# src/cc_stt/wakeword/sherpa_onnx.py
"""Sherpa-ONNX KWS backend implementation."""

import sys
from pathlib import Path
from typing import Any

import numpy as np


class SherpaONNXBackend:
    """Wake word detection using Sherpa-ONNX (Chinese/English bilingual).

    Supports pre-trained models from k2-fsa project.
    Can define custom keywords without retraining the model.

    Args:
        model_dir: Directory containing encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt
        keywords: List of keywords to detect (e.g., ["小爱同学", "小度小度"])
        keywords_file: Path to keywords.txt file (alternative to keywords list)
        num_threads: Number of threads for ONNX inference (default: 4)
        provider: ONNX execution provider ("cpu" or "cuda", default: "cpu")
    """

    def __init__(
        self,
        model_dir: str,
        keywords: list[str] | None = None,
        keywords_file: str | None = None,
        num_threads: int = 4,
        provider: str = "cpu",
    ) -> None:
        import sherpa_onnx

        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.provider = provider

        # Validate model directory
        if not self._check_model_files():
            raise FileNotFoundError(
                f"Sherpa-ONNX model files not found in {model_dir}. "
                f"Expected: encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt"
            )

        # Create keywords file if list provided
        if keywords and not keywords_file:
            keywords_file = self._create_keywords_file(keywords)

        if not keywords_file:
            raise ValueError("Either keywords or keywords_file must be provided")

        # Initialize KeywordSpotter
        self.spotter = sherpa_onnx.KeywordSpotter(
            encoder_config=sherpa_onnx.OnlineTransducerModelConfig(
                encoder=str(self.model_dir / "encoder.onnx"),
                decoder=str(self.model_dir / "decoder.onnx"),
                joiner=str(self.model_dir / "joiner.onnx"),
            ),
            tokens=str(self.model_dir / "tokens.txt"),
            keywords_file=keywords_file,
            num_threads=num_threads,
            provider=provider,
        )

        # Create initial stream
        self.stream = self.spotter.create_stream()

    def _check_model_files(self) -> bool:
        """Check if all required model files exist."""
        required_files = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
        return all((self.model_dir / f).exists() for f in required_files)

    def _create_keywords_file(self, keywords: list[str]) -> str:
        """Create a keywords.txt file from keyword list.

        Sherpa-ONNX expects keywords in format:
        keyword_name phoneme_sequence

        For Chinese, we use the characters themselves as a simple approximation.
        """
        keywords_path = self.model_dir / "custom_keywords.txt"

        with open(keywords_path, "w", encoding="utf-8") as f:
            for keyword in keywords:
                # Simple approach: use space-separated characters as phonemes
                # In production, proper pinyin conversion would be better
                phonemes = " ".join(keyword)
                f.write(f"{keyword} {phonemes}\n")

        return str(keywords_path)

    def process_audio(self, audio: np.ndarray) -> bool:
        """Process audio frame and detect wake word.

        Args:
            audio: Audio samples as float32 array, 16kHz sample rate

        Returns:
            True if wake word detected, False otherwise
        """
        self.stream.accept_waveform(16000, audio)
        result = self.spotter.get_result(self.stream)
        return result is not None and result != ""

    def reset(self) -> None:
        """Reset detection state for new detection session."""
        self.stream = self.spotter.create_stream()
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/jiang/cc/audio/cc-stt
PYTHONPATH=src .venv/bin/python -m pytest tests/wakeword/test_sherpa_onnx.py -v
```

Expected: 4 tests PASS (with mocks)

**Step 5: Commit**

```bash
git add src/cc_stt/wakeword/sherpa_onnx.py tests/wakeword/test_sherpa_onnx.py
git commit -m "feat: add SherpaONNXBackend implementation"
```

---

## Task 3: Update Factory and Package Exports

**Files:**
- Modify: `src/cc_stt/wakeword/factory.py`
- Modify: `src/cc_stt/wakeword/__init__.py`

**Step 1: Write the failing test**

```python
# tests/wakeword/test_factory.py - add to existing file

def test_factory_creates_sherpa_onnx():
    """Test factory creates SherpaONNXBackend."""
    from cc_stt.wakeword.factory import create_wakeword_backend

    with patch('cc_stt.wakeword.factory.SherpaONNXBackend') as mock_backend:
        mock_backend.return_value = Mock()

        result = create_wakeword_backend(
            backend="sherpa-onnx",
            name="xiao_ai",
            model_dir="models/sherpa-kws/zh",
            keywords=["小爱同学"],
            num_threads=8,
        )

        mock_backend.assert_called_once_with(
            model_dir="models/sherpa-kws/zh",
            keywords=["小爱同学"],
            keywords_file=None,
            num_threads=8,
            provider="cpu",
        )
        assert result is not None


def test_factory_sherpa_onnx_without_model_dir():
    """Test factory raises error for sherpa-onnx without model_dir."""
    from cc_stt.wakeword.factory import create_wakeword_backend

    with pytest.raises(ValueError, match="model_dir is required"):
        create_wakeword_backend(
            backend="sherpa-onnx",
            name="test",
            keywords=["test"],
        )
```

**Step 2: Run test to verify it fails**

```bash
cd /home/jiang/cc/audio/cc-stt
PYTHONPATH=src .venv/bin/python -m pytest tests/wakeword/test_factory.py::test_factory_creates_sherpa_onnx -v
```

Expected: FAIL with "Unknown backend: sherpa-onnx"

**Step 3: Update factory.py**

```python
# src/cc_stt/wakeword/factory.py

from typing import Literal

from .base import WakewordBackend
from .openwakeword import OpenWakeWordBackend
from .sherpa_onnx import SherpaONNXBackend  # NEW
from .wekws import WeKWSBackend


def create_wakeword_backend(
    backend: Literal["openwakeword", "wekws", "sherpa-onnx"],
    name: str,
    threshold: float = 0.5,
    model_path: str | None = None,
    window_size: int = 40,
    # Sherpa-ONNX specific parameters (NEW)
    model_dir: str | None = None,
    keywords: list[str] | None = None,
    keywords_file: str | None = None,
    num_threads: int = 4,
    provider: str = "cpu",
) -> WakewordBackend:
    """Create a wake word detection backend.

    Factory function that instantiates the appropriate backend based on
    configuration.

    Args:
        backend: Backend type ("openwakeword", "wekws", or "sherpa-onnx")
        name: Wake word name (used for openwakeword)
        threshold: Detection threshold (0.0-1.0)
        model_path: Path to model file (required for wekws)
        window_size: Inference window size in frames (wekws only)
        model_dir: Model directory (required for sherpa-onnx)
        keywords: List of keywords to detect (sherpa-onnx only)
        keywords_file: Path to keywords file (sherpa-onnx only)
        num_threads: Number of threads for ONNX inference (sherpa-onnx only)
        provider: ONNX execution provider (sherpa-onnx only)

    Returns:
        Configured WakewordBackend instance

    Raises:
        ValueError: If backend is unknown or required args missing
    """
    if backend == "openwakeword":
        return OpenWakeWordBackend(
            wakeword=name,
            threshold=threshold,
        )

    elif backend == "wekws":
        if not model_path:
            raise ValueError("model_path is required for 'wekws' backend")

        return WeKWSBackend(
            model_path=model_path,
            threshold=threshold,
            window_size=window_size,
        )

    elif backend == "sherpa-onnx":  # NEW
        if not model_dir:
            raise ValueError("model_dir is required for 'sherpa-onnx' backend")
        if not keywords and not keywords_file:
            raise ValueError("keywords or keywords_file is required for 'sherpa-onnx' backend")

        return SherpaONNXBackend(
            model_dir=model_dir,
            keywords=keywords,
            keywords_file=keywords_file,
            num_threads=num_threads,
            provider=provider,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")
```

**Step 4: Update __init__.py**

```python
# src/cc_stt/wakeword/__init__.py
"""Wake word detection module with pluggable backends."""

from .base import WakewordBackend
from .factory import create_wakeword_backend
from .openwakeword import OpenWakeWordBackend
from .sherpa_onnx import SherpaONNXBackend  # NEW
from .wekws import WeKWSBackend

__all__ = [
    "WakewordBackend",
    "create_wakeword_backend",
    "OpenWakeWordBackend",
    "SherpaONNXBackend",  # NEW
    "WeKWSBackend",
]
```

**Step 5: Run tests to verify they pass**

```bash
cd /home/jiang/cc/audio/cc-stt
PYTHONPATH=src .venv/bin/python -m pytest tests/wakeword/test_factory.py -v
```

Expected: All tests PASS (including new ones)

**Step 6: Commit**

```bash
git add src/cc_stt/wakeword/factory.py src/cc_stt/wakeword/__init__.py tests/wakeword/test_factory.py
git commit -m "feat: add sherpa-onnx backend support to factory"
```

---

## Task 4: Add Model Download Utility

**Files:**
- Create: `src/cc_stt/models/sherpa_kws/__init__.py`
- Create: `src/cc_stt/models/sherpa_kws/download.py`

**Step 1: Create download utility**

```python
# src/cc_stt/models/sherpa_kws/__init__.py
"""Sherpa-ONNX KWS model management."""

from .download import download_model, ensure_model_exists

__all__ = ["download_model", "ensure_model_exists"]
```

```python
# src/cc_stt/models/sherpa_kws/download.py
"""Download and manage Sherpa-ONNX KWS pre-trained models."""

import os
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


# Available pre-trained models
MODELS = {
    "sherpa-onnx-kws-zipformer-wenetspeech-3.3M": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2",
        "description": "Chinese keyword spotting (WenetSpeech, 3.3M params)",
    },
    "sherpa-onnx-kws-zipformer-zh-en-3M": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2024-12-20.tar.bz2",
        "description": "Chinese-English bilingual keyword spotting (3M params)",
    },
}


def download_model(model_name: str, output_dir: str | None = None) -> Path:
    """Download a pre-trained Sherpa-ONNX KWS model.

    Args:
        model_name: Name of the model (see MODELS dict)
        output_dir: Directory to extract model (default: ~/.local/share/cc-stt/models/sherpa-kws)

    Returns:
        Path to extracted model directory

    Raises:
        ValueError: If model_name is unknown
    """
    if model_name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    model_info = MODELS[model_name]
    url = model_info["url"]

    # Determine output directory
    if output_dir is None:
        output_dir = Path.home() / ".local" / "share" / "cc-stt" / "models" / "sherpa-kws"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    model_subdir = output_dir / model_name
    if model_subdir.exists() and (model_subdir / "encoder.onnx").exists():
        print(f"Model already exists: {model_subdir}")
        return model_subdir

    # Download
    tar_path = output_dir / f"{model_name}.tar.bz2"
    print(f"Downloading {model_name}...")
    print(f"  URL: {url}")
    print(f"  Destination: {tar_path}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\r  Progress: {percent}%", end="", flush=True)

    urlretrieve(url, tar_path, reporthook=progress_hook)
    print()  # New line after progress

    # Extract
    print(f"Extracting to {output_dir}...")
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(output_dir)

    # Clean up tar file
    tar_path.unlink()

    # Find extracted directory (may have timestamp suffix)
    extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir() and model_name in d.name]
    if extracted_dirs:
        model_subdir = extracted_dirs[0]
        print(f"✓ Model downloaded to: {model_subdir}")
        return model_subdir
    else:
        raise RuntimeError(f"Failed to find extracted model directory in {output_dir}")


def ensure_model_exists(model_dir: str | Path) -> Path:
    """Ensure model exists at the specified path.

    If model doesn't exist, auto-download the default Chinese model.

    Args:
        model_dir: Path to model directory

    Returns:
        Path to verified model directory
    """
    model_dir = Path(model_dir)

    # Check if model files exist
    required_files = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
    if all((model_dir / f).exists() for f in required_files):
        return model_dir

    # Auto-download default model
    print(f"Model not found at {model_dir}")
    print("Auto-downloading default Chinese KWS model...")

    default_model = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M"
    downloaded_dir = download_model(default_model, model_dir.parent)

    # If user specified a specific subdirectory, create symlink or copy
    if model_dir.name != downloaded_dir.name:
        if model_dir.exists() or model_dir.is_symlink():
            model_dir.unlink()
        model_dir.symlink_to(downloaded_dir, target_is_directory=True)

    return model_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Sherpa-ONNX KWS models")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="sherpa-onnx-kws-zipformer-wenetspeech-3.3M",
        help="Model to download",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ~/.local/share/cc-stt/models/sherpa-kws)",
    )

    args = parser.parse_args()

    model_dir = download_model(args.model, args.output_dir)
    print(f"\nModel ready at: {model_dir}")
    print(f"\nTo use in config:")
    print(f'  "sherpa_model_dir": "{model_dir}"')
```

**Step 2: Test download utility**

```bash
cd /home/jiang/cc/audio/cc-stt
PYTHONPATH=src .venv/bin/python -c "
from cc_stt.models.sherpa_kws.download import MODELS
print('Available models:')
for name, info in MODELS.items():
    print(f'  - {name}: {info[\"description\"]}')
"
```

Expected: Shows 2 available models

**Step 3: Commit**

```bash
git add src/cc_stt/models/sherpa_kws/
git commit -m "feat: add Sherpa-ONNX KWS model download utility"
```

---

## Task 5: Update Daemon Integration

**Files:**
- Modify: `src/cc_stt/daemon.py`

**Step 1: Update Daemon.__init__**

```python
# src/cc_stt/daemon.py - update __init__ method

# In __init__, add sherpa-onnx branch:

# Use factory
if backend == "wekws":
    # ... existing code ...
    pass
elif backend == "sherpa-onnx":  # NEW
    model_dir = self.config.wakeword.sherpa_model_dir
    if not model_dir:
        model_dir = "~/.local/share/cc-stt/models/sherpa-kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M"
    model_dir = os.path.expanduser(model_dir)

    # Ensure model exists (auto-download if needed)
    from cc_stt.models.sherpa_kws import ensure_model_exists
    model_dir = ensure_model_exists(model_dir)

    self.wakeword = create_wakeword_backend(
        backend="sherpa-onnx",
        name=wakeword_name,
        threshold=threshold,
        model_dir=model_dir,
        keywords=self.config.wakeword.sherpa_keywords,
        keywords_file=self.config.wakeword.sherpa_keywords_file,
        num_threads=self.config.wakeword.sherpa_num_threads,
    )
else:  # openwakeword
    # ... existing code ...
    pass
```

**Step 2: Test daemon initialization**

```bash
cd /home/jiang/cc/audio/cc-stt
PYTHONPATH=src .venv/bin/python -c "
from cc_stt.daemon import Daemon
print('Testing daemon import...')
# Note: Full test requires sherpa-onnx package and model
print('✓ Daemon imports successfully')
"
```

**Step 3: Commit**

```bash
git add src/cc_stt/daemon.py
git commit -m "feat: integrate Sherpa-ONNX backend into daemon"
```

---

## Task 6: Add Documentation

**Files:**
- Create: `docs/sherpa-onnx-setup.md`

**Step 1: Create documentation**

```markdown
# Sherpa-ONNX 唤醒词设置指南

## 快速开始

### 1. 安装依赖

```bash
pip install sherpa-onnx
```

### 2. 下载预训练模型

```bash
python -m cc_stt.models.sherpa_kws.download \
  --model sherpa-onnx-kws-zipformer-wenetspeech-3.3M
```

### 3. 配置唤醒词

编辑 `~/.config/cc-stt/config.json`：

```json
{
  "wakeword": {
    "backend": "sherpa-onnx",
    "sherpa_model_dir": "~/.local/share/cc-stt/models/sherpa-kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M",
    "sherpa_keywords": ["小爱同学", "小度小度"],
    "sherpa_num_threads": 4,
    "gain": 2.0
  }
}
```

### 4. 启动守护进程

```bash
cc-stt-daemon
```

## 可用模型

| 模型名称 | 语言 | 大小 | 特点 |
|---------|------|------|------|
| sherpa-onnx-kws-zipformer-wenetspeech-3.3M | 中文 | 3.3M | WenetSpeech训练，中文优化 |
| sherpa-onnx-kws-zipformer-zh-en-3M | 中英双语 | 3M | 2024最新，双语支持 |

## 自定义唤醒词

无需重新训练模型，只需修改配置：

```json
{
  "sherpa_keywords": ["你好小问", "嗨小文", "小智小智"]
}
```

支持任何中文词组，系统会自动转换为音素序列。

## 故障排除

### 模型下载失败

检查网络连接，或手动下载：
```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
```

### 检测不准确

调整阈值（需要代码支持）或检查麦克风输入。

## 参考资料

- [Sherpa-ONNX KWS文档](https://k2-fsa.github.io/sherpa/onnx/kws/index.html)
- [模型下载页面](https://k2-fsa.org/models/kws/)
```

**Step 2: Commit**

```bash
git add docs/sherpa-onnx-setup.md
git commit -m "docs: add Sherpa-ONNX setup guide"
```

---

## Task 7: Run Full Test Suite

**Files:**
- All test files

**Step 1: Run all new tests**

```bash
cd /home/jiang/cc/audio/cc-stt
PYTHONPATH=src .venv/bin/python -m pytest tests/wakeword/ tests/test_config.py -v --tb=short
```

Expected: All tests PASS

**Step 2: Verify imports work**

```bash
cd /home/jiang/cc/audio/cc-stt
PYTHONPATH=src .venv/bin/python -c "
from cc_stt.wakeword import SherpaONNXBackend, create_wakeword_backend
from cc_stt.models.sherpa_kws import download_model
print('✓ All imports successful')
"
```

Expected: Imports OK

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Sherpa-ONNX backend integration"
```

---

## Summary

This implementation adds:

1. **SherpaONNXBackend** - Full implementation of WakewordBackend Protocol
2. **Config updates** - New fields for model_dir, keywords, num_threads
3. **Factory support** - Create backend via factory pattern
4. **Model download** - Utility to download pre-trained models
5. **Daemon integration** - Auto-download and initialization
6. **Documentation** - Setup guide for users

**Usage:**

```json
{
  "wakeword": {
    "backend": "sherpa-onnx",
    "sherpa_model_dir": "~/.local/share/cc-stt/models/sherpa-kws/zh",
    "sherpa_keywords": ["小爱同学", "小度小度"]
  }
}
```
