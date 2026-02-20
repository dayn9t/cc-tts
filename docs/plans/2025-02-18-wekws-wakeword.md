# WeKWS Wake Word Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a WeKWS-based Chinese wake word detection backend with configurable backend switching.

**Architecture:** Create Protocol-based abstraction layer (`WakewordBackend`) with two implementations: `OpenWakeWordBackend` (existing) and `WeKWSBackend` (new). Shareable `FeatureExtractor` for audio preprocessing. Backend selected via config `wakeword.backend` field.

**Tech Stack:** onnxruntime, librosa, numpy, typing.Protocol

---

## Prerequisites Check

**Run before starting:**
```bash
# Verify project structure
ls src/cc_stt/
# Should show: wakeword.py, config.py, daemon.py, etc.

# Check Python version
python --version
# Should be 3.12+
```

---

## Task 1: Create FeatureExtractor for Audio Preprocessing

**Files:**
- Create: `src/cc_stt/features/__init__.py`
- Create: `src/cc_stt/features/audio.py`
- Create: `tests/features/test_audio.py`

**Step 1: Write the failing test**

```python
# tests/features/test_audio.py
import numpy as np
import pytest

from cc_stt.features.audio import FeatureExtractor


def test_fbank_feature_shape():
    """Test FBank feature extraction returns correct shape."""
    extractor = FeatureExtractor(sample_rate=16000, n_mels=80)
    # 1 second of audio at 16kHz
    audio = np.random.randn(16000).astype(np.float32)
    features = extractor.extract(audio)

    # Should return (n_frames, n_mels)
    assert features.ndim == 2
    assert features.shape[1] == 80
    # 1 second audio ~ 100 frames (with 10ms shift)
    assert 90 <= features.shape[0] <= 110


def test_fbank_feature_values():
    """Test FBank features are non-negative (power spectrum)."""
    extractor = FeatureExtractor(sample_rate=16000, n_mels=80)
    audio = np.random.randn(16000).astype(np.float32) * 0.1
    features = extractor.extract(audio)

    # FBank features should be non-negative
    assert np.all(features >= 0)


def test_empty_audio():
    """Test handling of empty audio."""
    extractor = FeatureExtractor(sample_rate=16000, n_mels=80)
    audio = np.array([], dtype=np.float32)
    features = extractor.extract(audio)

    assert features.shape[0] == 0


def test_different_n_mels():
    """Test FeatureExtractor with different n_mels values."""
    for n_mels in [40, 64, 80]:
        extractor = FeatureExtractor(sample_rate=16000, n_mels=n_mels)
        audio = np.random.randn(16000).astype(np.float32)
        features = extractor.extract(audio)
        assert features.shape[1] == n_mels
```

**Step 2: Run test to verify it fails**

```bash
cd /home/jiang/cc/audio/cc-stt
python -m pytest tests/features/test_audio.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'cc_stt.features'"

**Step 3: Write minimal implementation**

```python
# src/cc_stt/features/__init__.py
"""Audio feature extraction module."""

from .audio import FeatureExtractor

__all__ = ["FeatureExtractor"]
```

```python
# src/cc_stt/features/audio.py
"""Audio feature extraction using librosa."""

import numpy as np


class FeatureExtractor:
    """Extract FBank (filterbank) features from audio.

    This class extracts mel-frequency filterbank features suitable for
    speech recognition and wake word detection models.

    Args:
        sample_rate: Audio sample rate in Hz (default: 16000)
        n_mels: Number of mel filterbank bins (default: 80)
        frame_length: Frame length in milliseconds (default: 25)
        frame_shift: Frame shift in milliseconds (default: 10)
        n_fft: FFT size (default: None, auto-calculated from frame_length)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        n_fft: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.n_fft = n_fft or int(frame_length * sample_rate / 1000)
        self.hop_length = int(frame_shift * sample_rate / 1000)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract FBank features from audio.

        Args:
            audio: Audio samples as float32 array, shape (n_samples,)

        Returns:
            FBank features as float32 array, shape (n_frames, n_mels)
        """
        import librosa

        if len(audio) == 0:
            return np.array([], dtype=np.float32).reshape(0, self.n_mels)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=20,
            fmax=self.sample_rate // 2,
        )

        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Transpose to (n_frames, n_mels)
        features = log_mel_spec.T.astype(np.float32)

        return features
```

**Step 4: Run test to verify it passes**

```bash
cd /home/jiang/cc/audio/cc-stt
python -m pytest tests/features/test_audio.py -v
```

Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add src/cc_stt/features/ tests/features/
git commit -m "feat: add FeatureExtractor for FBank audio features"
```

---

## Task 2: Create WakewordBackend Protocol and Base Interface

**Files:**
- Create: `src/cc_stt/wakeword/__init__.py`
- Create: `src/cc_stt/wakeword/base.py`

**Step 1: Write the Protocol definition**

```python
# src/cc_stt/wakeword/base.py
"""Base protocol for wake word detection backends."""

from typing import Protocol
import numpy as np


class WakewordBackend(Protocol):
    """Protocol for wake word detection backends.

    Implementations must provide:
    - process_audio(): Process audio frames and return detection result
    - reset(): Reset internal state for new detection session
    """

    def process_audio(self, audio: np.ndarray) -> bool:
        """Process audio frame and detect wake word.

        Args:
            audio: Audio samples as float32 array, typically 1280 samples
                   at 16kHz (80ms frame for openwakeword compatibility)

        Returns:
            True if wake word detected, False otherwise
        """
        ...

    def reset(self) -> None:
        """Reset internal state.

        Called after wake word detection to prepare for next detection.
        """
        ...
```

**Step 2: Update wakeword package init**

```python
# src/cc_stt/wakeword/__init__.py
"""Wake word detection module with pluggable backends."""

from .base import WakewordBackend

__all__ = ["WakewordBackend"]
```

**Step 3: Verify import works**

```bash
cd /home/jiang/cc/audio/cc-stt
python -c "from cc_stt.wakeword.base import WakewordBackend; print('OK')"
```

Expected: OK

**Step 4: Commit**

```bash
git add src/cc_stt/wakeword/
git commit -m "feat: add WakewordBackend Protocol abstraction"
```

---

## Task 3: Refactor Existing WakewordDetector to OpenWakeWordBackend

**Files:**
- Create: `src/cc_stt/wakeword/openwakeword.py`
- Modify: `src/cc_stt/wakeword.py` (mark as deprecated)

**Step 1: Create OpenWakeWordBackend implementation**

```python
# src/cc_stt/wakeword/openwakeword.py
"""OpenWakeWord backend implementation."""

import sys
import numpy as np
import openwakeword
from openwakeword.model import Model


class OpenWakeWordBackend:
    """Wake word detection using OpenWakeWord.

    Supports pre-trained English wake word models from openwakeword.
    """

    def __init__(self, wakeword: str = "alexa", threshold: float = 0.5) -> None:
        """Initialize OpenWakeWord backend.

        Args:
            wakeword: Name of the wake word model (e.g., "alexa", "hey_jarvis")
            threshold: Detection threshold (0.0-1.0), higher = more strict
        """
        print("正在初始化 OpenWakeWordBackend...", file=sys.stderr, flush=True)
        self.wakeword = wakeword
        self.threshold = threshold

        # Get the full path to the pretrained model
        print("获取预训练模型路径...", file=sys.stderr, flush=True)
        model_paths = openwakeword.get_pretrained_model_paths()
        matching_model = [p for p in model_paths if wakeword in p]

        if not matching_model:
            raise ValueError(f"Model '{wakeword}' not found in pretrained models")

        print(f"加载模型: {matching_model[0]}", file=sys.stderr, flush=True)
        self.model = Model(wakeword_model_paths=[matching_model[0]])
        # Store the actual model key (e.g., 'alexa_v0.1')
        self.model_key = list(self.model.models.keys())[0]
        print("OpenWakeWordBackend 初始化完成", file=sys.stderr, flush=True)

    def process_audio(self, audio: np.ndarray) -> bool:
        """Process audio frame and detect wake word."""
        prediction = self.model.predict(audio)
        score = prediction.get(self.model_key, 0)
        # DEBUG: 记录接近阈值的分数
        if score > 0.3:
            print(f"[wakeword] 检测分数: {score:.3f} (阈值: {self.threshold})", file=sys.stderr, flush=True)
        if score > self.threshold:
            print(f"[wakeword] 检测到唤醒词! 分数: {score:.3f}", file=sys.stderr, flush=True)
            self.model.reset()
            return True
        return False

    def reset(self) -> None:
        """Reset detection state."""
        self.model.reset()
```

**Step 2: Update wakeword/__init__.py**

```python
# src/cc_stt/wakeword/__init__.py
"""Wake word detection module with pluggable backends."""

from .base import WakewordBackend
from .openwakeword import OpenWakeWordBackend

__all__ = ["WakewordBackend", "OpenWakeWordBackend"]
```

**Step 3: Verify imports work**

```bash
cd /home/jiang/cc/audio/cc-stt
python -c "from cc_stt.wakeword import OpenWakeWordBackend; print('OK')"
```

Expected: OK

**Step 4: Commit**

```bash
git add src/cc_stt/wakeword/openwakeword.py src/cc_stt/wakeword/__init__.py
git commit -m "refactor: extract OpenWakeWordBackend from wakeword.py"
```

---

## Task 4: Update Config for Backend Selection

**Files:**
- Modify: `src/cc_stt/config.py`

**Step 1: Write test for new config**

```python
# tests/test_config.py
import json
import tempfile
import os

from cc_stt.config import Config, WakewordConfig


def test_wakeword_config_defaults():
    """Test WakewordConfig default values."""
    config = WakewordConfig()
    assert config.backend == "openwakeword"
    assert config.name == "alexa"
    assert config.threshold == 0.3
    assert config.gain == 2.0
    assert config.model_path is None
    assert config.window_size == 40


def test_wakeword_config_wekws_values():
    """Test WakewordConfig with WeKWS backend."""
    config = WakewordConfig(
        backend="wekws",
        name="xiao_ai",
        threshold=0.6,
        model_path="models/wekws/kws.onnx",
        window_size=50,
    )
    assert config.backend == "wekws"
    assert config.model_path == "models/wekws/kws.onnx"


def test_config_load_default():
    """Test Config.load() creates default config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        config = Config.load(config_path)

        assert config.wakeword.backend == "openwakeword"
        assert os.path.exists(config_path)
```

**Step 2: Update config.py**

```python
# src/cc_stt/config.py
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
    """Wake word detection configuration.

    Supports multiple backends: openwakeword (English), wekws (Chinese)
    """
    # Backend selection
    backend: Literal["openwakeword", "wekws"] = "openwakeword"

    # Common settings
    name: str = "alexa"  # Wake word name (model identifier)
    threshold: float = 0.3  # Detection threshold
    gain: float = 2.0  # Audio gain multiplier

    # WeKWS-specific settings
    model_path: str | None = None  # Path to ONNX model (None = use default)
    window_size: int = 40  # Inference window size in frames (~400ms)


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
                },
                "hotwords": {"file": config.hotwords_file}
            }, indent=2))
            return config

        data = json.loads(config_path.read_text())
        wakeword_data = data.get("wakeword", {})

        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            model=ModelConfig(**data.get("model", {})),
            wakeword=WakewordConfig(
                backend=wakeword_data.get("backend", "openwakeword"),
                name=wakeword_data.get("name", "alexa"),
                threshold=wakeword_data.get("threshold", 0.3),
                gain=wakeword_data.get("gain", 2.0),
                model_path=wakeword_data.get("model_path"),
                window_size=wakeword_data.get("window_size", 40),
            ),
            hotwords_file=data.get("hotwords", {}).get("file", "~/.config/cc-stt/hotwords.txt")
        )
```

**Step 3: Run tests**

```bash
cd /home/jiang/cc/audio/cc-stt
python -m pytest tests/test_config.py -v
```

Expected: 3 tests PASS

**Step 4: Commit**

```bash
git add src/cc_stt/config.py tests/test_config.py
git commit -m "feat: add backend selection to WakewordConfig"
```

---

## Task 5: Create WeKWS Backend Implementation

**Files:**
- Create: `src/cc_stt/wakeword/wekws.py`
- Create: `tests/wakeword/test_wekws.py`

**Step 1: Write tests for WeKWS backend**

```python
# tests/wakeword/test_wekws.py
import numpy as np
import pytest
from unittest.mock import Mock, patch

from cc_stt.wakeword.wekws import WeKWSBackend, RingBuffer


def test_ring_buffer_basic():
    """Test RingBuffer basic operations."""
    buf = RingBuffer(maxlen=5)

    # Append items
    buf.append(1)
    buf.append(2)
    buf.append(3)

    assert len(buf) == 3
    assert list(buf) == [1, 2, 3]

    # Fill to capacity
    buf.append(4)
    buf.append(5)
    buf.append(6)  # Should evict 1

    assert len(buf) == 5
    assert list(buf) == [2, 3, 4, 5, 6]


def test_ring_buffer_to_array():
    """Test RingBuffer to_array method."""
    buf = RingBuffer(maxlen=3)
    buf.append(np.array([1.0, 2.0]))
    buf.append(np.array([3.0, 4.0]))

    arr = buf.to_array()
    assert arr.shape == (2, 2)
    assert np.allclose(arr, [[1.0, 2.0], [3.0, 4.0]])


def test_wekws_backend_initialization():
    """Test WeKWSBackend initialization with mocked model."""
    with patch("onnxruntime.InferenceSession") as mock_session:
        mock_session.return_value = Mock()

        backend = WeKWSBackend(
            model_path="fake_model.onnx",
            threshold=0.6,
            window_size=40,
        )

        assert backend.threshold == 0.6
        assert backend.window_size == 40
        mock_session.assert_called_once_with("fake_model.onnx")


def test_wekws_process_audio_silence():
    """Test WeKWSBackend with silent audio (no detection)."""
    with patch("onnxruntime.InferenceSession") as mock_session:
        # Mock model to return low score
        mock_output = Mock()
        mock_output.__getitem__ = Mock(return_value=np.array([[0.1]]))
        mock_session.return_value.run = Mock(return_value=[mock_output])
        mock_session.return_value.get_inputs = Mock(return_value=[Mock(shape=[None, 40, 80])])

        backend = WeKWSBackend(
            model_path="fake_model.onnx",
            threshold=0.5,
            window_size=40,
        )

        # Process silent audio (40 frames worth)
        audio = np.zeros(16000, dtype=np.float32)  # 1 second
        result = backend.process_audio(audio)

        assert result is False


def test_wekws_process_audio_detection():
    """Test WeKWSBackend with detection."""
    with patch("onnxruntime.InferenceSession") as mock_session:
        # Mock model to return high score
        mock_output = Mock()
        mock_output.__getitem__ = Mock(return_value=np.array([[0.8]]))
        mock_session.return_value.run = Mock(return_value=[mock_output])
        mock_session.return_value.get_inputs = Mock(return_value=[Mock(shape=[None, 40, 80])])

        backend = WeKWSBackend(
            model_path="fake_model.onnx",
            threshold=0.5,
            window_size=40,
        )

        # Process audio (40 frames worth)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = backend.process_audio(audio)

        assert result is True


def test_wekws_reset():
    """Test WeKWSBackend reset clears buffer."""
    with patch("onnxruntime.InferenceSession") as mock_session:
        mock_session.return_value = Mock()

        backend = WeKWSBackend(model_path="fake_model.onnx")

        # Add some data to buffer
        backend.feature_buffer.append(np.zeros(80))
        backend.feature_buffer.append(np.zeros(80))
        assert len(backend.feature_buffer) == 2

        # Reset should clear buffer
        backend.reset()
        assert len(backend.feature_buffer) == 0
```

**Step 2: Implement WeKWSBackend**

```python
# src/cc_stt/wakeword/wekws.py
"""WeKWS backend implementation for Chinese wake word detection."""

import sys
from collections import deque
from typing import Any

import numpy as np

from cc_stt.features.audio import FeatureExtractor


class RingBuffer:
    """Fixed-size ring buffer for feature frames.

    Maintains a sliding window of feature vectors for streaming inference.
    """

    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen
        self._buffer: deque[np.ndarray] = deque(maxlen=maxlen)

    def append(self, item: np.ndarray) -> None:
        """Append a feature frame to the buffer."""
        self._buffer.append(item)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    def clear(self) -> None:
        """Clear all items from buffer."""
        self._buffer.clear()

    def to_array(self) -> np.ndarray:
        """Convert buffer to numpy array (n_frames, n_features)."""
        if len(self._buffer) == 0:
            return np.array([], dtype=np.float32).reshape(0, 0)
        return np.stack(list(self._buffer))


class WeKWSBackend:
    """Wake word detection using WeKWS (Chinese optimized).

    Uses ONNX Runtime for inference with sliding window approach.
    Supports custom models trained with WeKWS toolkit.

    Args:
        model_path: Path to ONNX model file
        threshold: Detection threshold (0.0-1.0), default 0.5
        window_size: Number of frames in inference window, default 40 (~400ms)
        feature_extractor: Optional custom FeatureExtractor
        sample_rate: Audio sample rate, default 16000
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        window_size: int = 40,
        feature_extractor: FeatureExtractor | None = None,
        sample_rate: int = 16000,
    ) -> None:
        print(f"正在初始化 WeKWSBackend...", file=sys.stderr, flush=True)
        print(f"模型路径: {model_path}", file=sys.stderr, flush=True)

        self.model_path = model_path
        self.threshold = threshold
        self.window_size = window_size

        # Initialize feature extractor
        self.feature_extractor = feature_extractor or FeatureExtractor(
            sample_rate=sample_rate,
            n_mels=80,
        )

        # Initialize ONNX Runtime session
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Feature buffer for sliding window
        self.feature_buffer: RingBuffer = RingBuffer(maxlen=window_size)

        print(f"WeKWSBackend 初始化完成 (阈值: {threshold}, 窗口: {window_size})",
              file=sys.stderr, flush=True)

    def process_audio(self, audio: np.ndarray) -> bool:
        """Process audio frame and detect wake word.

        Uses sliding window approach: extracts features from audio,
        accumulates in buffer, runs inference when buffer is full.

        Args:
            audio: Audio samples as float32 array

        Returns:
            True if wake word detected in this frame, False otherwise
        """
        # Extract FBank features
        features = self.feature_extractor.extract(audio)

        if len(features) == 0:
            return False

        # Add frames to buffer
        for frame in features:
            self.feature_buffer.append(frame)

            # Run inference when buffer is full
            if len(self.feature_buffer) == self.window_size:
                score = self._infer()
                if score > self.threshold:
                    print(f"[wekws] 检测到唤醒词! 分数: {score:.3f}",
                          file=sys.stderr, flush=True)
                    return True

        return False

    def _infer(self) -> float:
        """Run inference on current feature buffer.

        Returns:
            Detection confidence score (0.0-1.0)
        """
        # Get features from buffer
        features = self.feature_buffer.to_array()

        # Add batch dimension: (window_size, n_mels) -> (1, window_size, n_mels)
        input_data = features[np.newaxis, ...].astype(np.float32)

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_data})

        # Extract score from output (assuming single output, batch size 1)
        # Output shape typically: (batch_size, 1) or (batch_size,)
        score = float(outputs[0][0][0] if outputs[0].ndim > 1 else outputs[0][0])

        return score

    def reset(self) -> None:
        """Reset detection state.

        Clears feature buffer to prepare for new detection session.
        """
        self.feature_buffer.clear()
```

**Step 3: Update wakeword/__init__.py**

```python
# src/cc_stt/wakeword/__init__.py
"""Wake word detection module with pluggable backends."""

from .base import WakewordBackend
from .openwakeword import OpenWakeWordBackend
from .wekws import WeKWSBackend

__all__ = ["WakewordBackend", "OpenWakeWordBackend", "WeKWSBackend"]
```

**Step 4: Run tests**

```bash
cd /home/jiang/cc/audio/cc-stt
python -m pytest tests/wakeword/test_wekws.py -v
```

Expected: 6 tests PASS

**Step 5: Commit**

```bash
git add src/cc_stt/wakeword/wekws.py tests/wakeword/test_wekws.py src/cc_stt/wakeword/__init__.py
git commit -m "feat: add WeKWS backend with RingBuffer and sliding window inference"
```

---

## Task 6: Create Backend Factory and Unified Interface

**Files:**
- Create: `src/cc_stt/wakeword/factory.py`
- Create: `tests/wakeword/test_factory.py`

**Step 1: Write tests for factory**

```python
# tests/wakeword/test_factory.py
import pytest
from unittest.mock import Mock, patch

from cc_stt.wakeword.factory import create_wakeword_backend
from cc_stt.wakeword.openwakeword import OpenWakeWordBackend
from cc_stt.wakeword.wekws import WeKWSBackend


def test_factory_creates_openwakeword():
    """Test factory creates OpenWakeWordBackend."""
    with patch("cc_stt.wakeword.factory.OpenWakeWordBackend") as mock_backend:
        mock_backend.return_value = Mock(spec=OpenWakeWordBackend)

        result = create_wakeword_backend(
            backend="openwakeword",
            name="alexa",
            threshold=0.5,
        )

        mock_backend.assert_called_once_with(wakeword="alexa", threshold=0.5)
        assert result is not None


def test_factory_creates_wekws():
    """Test factory creates WeKWSBackend."""
    with patch("cc_stt.wakeword.factory.WeKWSBackend") as mock_backend:
        mock_backend.return_value = Mock(spec=WeKWSBackend)

        result = create_wakeword_backend(
            backend="wekws",
            name="xiao_ai",
            threshold=0.6,
            model_path="models/wekws/kws.onnx",
            window_size=50,
        )

        mock_backend.assert_called_once_with(
            model_path="models/wekws/kws.onnx",
            threshold=0.6,
            window_size=50,
        )


def test_factory_invalid_backend():
    """Test factory raises error for invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend"):
        create_wakeword_backend(backend="invalid", name="test")


def test_factory_wekws_without_model_path():
    """Test factory raises error for wekws without model_path."""
    with pytest.raises(ValueError, match="model_path is required"):
        create_wakeword_backend(backend="wekws", name="test")
```

**Step 2: Implement factory**

```python
# src/cc_stt/wakeword/factory.py
"""Factory for creating wake word detection backends."""

from typing import Literal

from .base import WakewordBackend
from .openwakeword import OpenWakeWordBackend
from .wekws import WeKWSBackend


def create_wakeword_backend(
    backend: Literal["openwakeword", "wekws"],
    name: str,
    threshold: float = 0.5,
    model_path: str | None = None,
    window_size: int = 40,
) -> WakewordBackend:
    """Create a wake word detection backend.

    Factory function that instantiates the appropriate backend based on
    configuration.

    Args:
        backend: Backend type ("openwakeword" or "wekws")
        name: Wake word name (e.g., "alexa", "xiao_ai")
        threshold: Detection threshold (0.0-1.0)
        model_path: Path to model file (required for wekws)
        window_size: Inference window size in frames (wekws only)

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
            raise ValueError("model_path is required for wekws backend")

        return WeKWSBackend(
            model_path=model_path,
            threshold=threshold,
            window_size=window_size,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")
```

**Step 3: Update wakeword/__init__.py**

```python
# src/cc_stt/wakeword/__init__.py
"""Wake word detection module with pluggable backends."""

from .base import WakewordBackend
from .factory import create_wakeword_backend
from .openwakeword import OpenWakeWordBackend
from .wekws import WeKWSBackend

__all__ = [
    "WakewordBackend",
    "create_wakeword_backend",
    "OpenWakeWordBackend",
    "WeKWSBackend",
]
```

**Step 4: Run tests**

```bash
cd /home/jiang/cc/audio/cc-stt
python -m pytest tests/wakeword/test_factory.py -v
```

Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add src/cc_stt/wakeword/factory.py tests/wakeword/test_factory.py src/cc_stt/wakeword/__init__.py
git commit -m "feat: add wakeword backend factory for runtime selection"
```

---

## Task 7: Update Daemon to Use Factory

**Files:**
- Modify: `src/cc_stt/daemon.py`
- Modify: `src/cc_stt/daemon_simple.py` (if exists)

**Step 1: Update daemon.py imports**

```python
# src/cc_stt/daemon.py - update imports
import os
import sys
import numpy as np
import sounddevice as sd

# Ensure DISPLAY is set for X11 (awesome WM compatibility)
if not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'

from cc_stt.config import Config
from cc_stt.wakeword import create_wakeword_backend
from cc_stt.recorder import AudioRecorder
from cc_stt.transcriber import SpeechTranscriber
from cc_stt.editor import EditorWindow
from cc_stt.voice_edit import VoiceEditor
from cc_stt.sender import Sender
```

**Step 2: Update Daemon.__init__ method**

```python
# src/cc_stt/daemon.py - update __init__ method
def __init__(self, wakeword: str = "alexa"):
    self.config = Config.load()

    # 从配置读取唤醒词设置
    wakeword_name = getattr(self.config.wakeword, 'name', wakeword)
    threshold = self.config.wakeword.threshold
    gain = self.config.wakeword.gain
    backend = getattr(self.config.wakeword, 'backend', 'openwakeword')

    log(f"[daemon] 唤醒词配置: name={wakeword_name}, backend={backend}, "
        f"threshold={threshold}, gain={gain}x")

    # 使用工厂创建后端
    if backend == "wekws":
        model_path = self.config.wakeword.model_path
        window_size = self.config.wakeword.window_size

        if not model_path:
            # 使用默认模型路径
            model_path = os.path.expanduser(
                "~/.local/share/cc-stt/models/wekws/kws_zh.onnx"
            )
            log(f"[daemon] 使用默认 WeKWS 模型路径: {model_path}")

        self.wakeword = create_wakeword_backend(
            backend="wekws",
            name=wakeword_name,
            threshold=threshold,
            model_path=model_path,
            window_size=window_size,
        )
    else:
        self.wakeword = create_wakeword_backend(
            backend="openwakeword",
            name=wakeword_name,
            threshold=threshold,
        )

    self.audio_gain = gain
    # ... rest of initialization unchanged
    self.recorder = AudioRecorder(
        sample_rate=self.config.audio.sample_rate,
        channels=self.config.audio.channels
    )
    self.transcriber = SpeechTranscriber(self.config.model.name)
    self.voice_editor = VoiceEditor()
    self.sender = Sender()
    self.running = False
    self.triggered = False
    self.current_text = ""
```

**Step 3: Update run() method logging**

```python
# src/cc_stt/daemon.py - update run() method
def run(self):
    """主循环"""
    self.running = True
    backend_type = self.config.wakeword.backend
    log(f"[daemon] 语音助手已启动，唤醒词: {self.config.wakeword.name} "
        f"(后端: {backend_type})")
    log(f"[daemon] 采样率: {self.config.audio.sample_rate} Hz, "
        f"声道: {self.config.audio.channels}")
    # ... rest unchanged
```

**Step 4: Verify daemon imports work**

```bash
cd /home/jiang/cc/audio/cc-stt
python -c "from cc_stt.daemon import Daemon; print('OK')"
```

Expected: OK (or warning about DISPLAY, but no import errors)

**Step 5: Commit**

```bash
git add src/cc_stt/daemon.py
git commit -m "feat: update daemon to use backend factory for runtime selection"
```

---

## Task 8: Add ONNX Runtime Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add onnxruntime to dependencies**

```toml
# pyproject.toml - add to dependencies
dependencies = [
    "faster-whisper>=1.1.0",
    "mcp>=1.6.0",
    "openwakeword>=0.5.0",
    "onnxruntime>=1.16",  # ADD THIS LINE
    "pyaudio>=0.2.11",
    "sounddevice>=0.4.6",
    "numpy>=1.24",
    "librosa>=0.10",
    # ... other dependencies
]
```

**Step 2: Verify toml syntax**

```bash
cd /home/jiang/cc/audio/cc-stt
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb')); print('OK')"
```

Expected: OK

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add onnxruntime dependency for WeKWS backend"
```

---

## Task 9: Create Placeholder Model Directory and Documentation

**Files:**
- Create: `models/wekws/.gitkeep`
- Create: `models/wekws/README.md`
- Create: `models/.gitattributes`

**Step 1: Create README for models**

```markdown
# WeKWS Models

This directory contains pre-trained WeKWS models for Chinese wake word detection.

## Model Format

Models must be in ONNX format (.onnx) exported from WeKWS training.

Expected input shape: `(batch_size, window_size, n_mels)`
- `batch_size`: 1 (inference)
- `window_size`: 40 frames (~400ms)
- `n_mels`: 80 (mel filterbank bins)

## Obtaining Models

### Option 1: Use Pre-trained Models

Download from WeKWS official repository:
```bash
wget https://github.com/wenet-e2e/wekws/releases/download/v1.0.0/kws_zh.onnx \
  -O models/wekws/kws_zh.onnx
```

### Option 2: Train Your Own

Follow WeKWS training guide:
https://github.com/wenet-e2e/wekws

Export to ONNX:
```python
import torch
model = load_your_model()
dummy_input = torch.randn(1, 40, 80)
torch.onnx.export(model, dummy_input, "kws_custom.onnx")
```

## Configuration

Edit `~/.config/cc-stt/config.json`:

```json
{
  "wakeword": {
    "backend": "wekws",
    "name": "xiao_ai",
    "threshold": 0.5,
    "model_path": "models/wekws/kws_zh.onnx",
    "window_size": 40
  }
}
```

## Supported Wake Words

Pre-trained models may support:
- "小爱同学" (xiao_ai)
- "你好天猫" (tianmao)
- "小度小度" (xiaodu)

Custom models can be trained for any phrase.
```

**Step 2: Create .gitattributes for git-lfs**

```
# models/.gitattributes
*.onnx filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
```

**Step 3: Commit**

```bash
git add models/
git commit -m "docs: add WeKWS model directory with README and git-lfs config"
```

---

## Task 10: Run Full Test Suite

**Files:**
- All test files

**Step 1: Run all new tests**

```bash
cd /home/jiang/cc/audio/cc-stt
python -m pytest tests/features/ tests/wakeword/ tests/test_config.py -v
```

Expected: All tests PASS (14+ tests)

**Step 2: Verify imports work correctly**

```bash
cd /home/jiang/cc/audio/cc-stt
python -c "
from cc_stt.wakeword import create_wakeword_backend, OpenWakeWordBackend, WeKWSBackend
from cc_stt.features import FeatureExtractor
from cc_stt.config import Config, WakewordConfig
print('All imports OK')
"
```

Expected: All imports OK

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete WeKWS backend integration with factory pattern"
```

---

## Summary

This implementation adds:

1. **FeatureExtractor** - Reusable FBank feature extraction
2. **WakewordBackend Protocol** - Clean abstraction for pluggable backends
3. **OpenWakeWordBackend** - Refactored existing implementation
4. **WeKWSBackend** - New Chinese-optimized backend
5. **Backend Factory** - Runtime backend selection via config
6. **Updated Config** - `wakeword.backend` field for selection
7. **Updated Daemon** - Uses factory for backend instantiation
8. **Model Directory** - Place for pre-trained models with git-lfs

**Usage:**

```json
// ~/.config/cc-stt/config.json
{
  "wakeword": {
    "backend": "wekws",
    "name": "xiao_ai",
    "threshold": 0.5,
    "model_path": "models/wekws/kws_zh.onnx"
  }
}
```

**Backwards Compatibility:** Default backend remains `openwakeword`, existing configs continue to work.
