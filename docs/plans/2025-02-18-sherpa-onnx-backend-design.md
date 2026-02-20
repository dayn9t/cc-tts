# Sherpa-ONNX Backend Design

> **Status:** Approved for implementation
> **Date:** 2025-02-18

## Overview

Add Sherpa-ONNX KWS (Keyword Spotting) backend support to enable Chinese wake word detection using pre-trained models from the k2-fsa project.

## Goals

1. Enable Chinese wake word detection without training
2. Support custom keywords without model retraining
3. Maintain clean architecture with existing WakewordBackend Protocol
4. Provide model download/management tools

## Architecture

```
WakewordBackend (Protocol)
├── OpenWakeWordBackend
├── WeKWSBackend
└── SherpaONNXBackend (NEW)
    └── sherpa_onnx.KeywordSpotter
        ├── encoder.onnx
        ├── decoder.onnx
        └── joiner.onnx
```

## Components

### 1. SherpaONNXBackend Class

**Location:** `src/cc_stt/wakeword/sherpa_onnx.py`

**Interface:**
```python
class SherpaONNXBackend:
    def __init__(
        self,
        model_dir: str,
        keywords: list[str] | None = None,
        keywords_file: str | None = None,
        num_threads: int = 4,
        provider: str = "cpu",
    ) -> None

    def process_audio(self, audio: np.ndarray) -> bool
    def reset(self) -> None
```

**Key Features:**
- Lazy model download (first use)
- Support both keywords list and keywords file
- Automatic phoneme conversion for Chinese
- Stream-based inference for real-time detection

### 2. Config Update

**Location:** `src/cc_stt/config.py`

**New fields in WakewordConfig:**
```python
backend: Literal["openwakeword", "wekws", "sherpa-onnx"] = "openwakeword"

# Sherpa-ONNX specific
sherpa_model_dir: str | None = None
sherpa_keywords: list[str] | None = None
sherpa_keywords_file: str | None = None
sherpa_num_threads: int = 4
```

### 3. Factory Update

**Location:** `src/cc_stt/wakeword/factory.py`

Add `sherpa-onnx` branch in `create_wakeword_backend()`:
```python
elif backend == "sherpa-onnx":
    return SherpaONNXBackend(
        model_dir=config.sherpa_model_dir,
        keywords=config.sherpa_keywords,
        keywords_file=config.sherpa_keywords_file,
        num_threads=config.sherpa_num_threads,
    )
```

### 4. Model Download Tool

**Location:** `src/cc_stt/models/sherpa_kws/download.py`

**Supported Models:**
- `sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01` (Chinese)
- `sherpa-onnx-kws-zipformer-zh-en-3M-2024-12-20` (Chinese+English)

**Usage:**
```bash
python -m cc_stt.models.sherpa_kws.download \
  --model sherpa-onnx-kws-zipformer-wenetspeech-3.3M \
  --output-dir ~/.local/share/cc-stt/models/sherpa-kws/zh
```

## Data Flow

```
Audio Input (16kHz, float32)
    ↓
SherpaONNXBackend.process_audio()
    ↓
KeywordSpotterStream.accept_waveform()
    ↓
Transducer Inference (encoder → decoder → joiner)
    ↓
Keyword Detection Result
    ↓
Boolean (True if keyword detected)
```

## Error Handling

1. **Model not found:** Auto-download on first use
2. **Invalid keywords:** Validate phoneme conversion
3. **ONNX Runtime error:** Fallback to CPU provider
4. **Missing sherpa-onnx package:** Clear installation instructions

## Testing Strategy

1. Unit tests for SherpaONNXBackend
2. Integration tests with downloaded model
3. Mock tests for KeywordSpotter
4. Config loading tests

## Dependencies

```toml
[project.optional-dependencies]
sherpa = ["sherpa-onnx>=1.10.0"]
```

## Backwards Compatibility

- Default backend remains `openwakeword`
- Existing configs work without changes
- Sherpa-ONNX is optional dependency

## References

- Sherpa-ONNX Docs: https://k2-fsa.github.io/sherpa/onnx/kws/index.html
- Model Zoo: https://k2-fsa.org/models/kws/
- GitHub: https://github.com/k2-fsa/sherpa-onnx
