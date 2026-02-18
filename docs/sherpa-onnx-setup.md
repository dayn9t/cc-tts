# Sherpa-ONNX Wake Word Detection Setup Guide

This guide covers setting up Sherpa-ONNX for wake word detection in cc-stt.

## Quick Start

### 1. Install Dependencies

```bash
# Install sherpa-onnx Python package
pip install sherpa-onnx

# Or with uv
uv pip install sherpa-onnx
```

### 2. Download Model

Download a pre-trained WeKWS model:

```bash
# Create models directory
mkdir -p models/wekws/wenwen

# Download wenwen (文文) Chinese wake word model
# Model files should include:
# - model.onnx (inference model)
# - tokens.txt (token mapping)
# - keywords.txt (keyword definitions)
```

See [Available Models](#available-models) section for download links.

### 3. Configure

Create or edit your configuration file:

```json
{
  "wakeword": {
    "backend": "sherpa-onnx",
    "model_path": "models/wekws/wenwen/model.onnx",
    "tokens_path": "models/wekws/wenwen/tokens.txt",
    "keywords_path": "models/wekws/wenwen/keywords.txt",
    "threshold": 0.5,
    "sample_rate": 16000
  }
}
```

### 4. Run

```bash
# Start the daemon with Sherpa-ONNX backend
python -m cc_stt.daemon --config config.json

# Or use the simple daemon
python src/cc_stt/daemon_simple.py
```

## Available Models

| Model | Language | Description | Use Case |
|-------|----------|-------------|----------|
| wenwen | Chinese (zh) | "文文" wake word | Chinese-only applications |
| zh-en-bilingual | Chinese + English | Bilingual wake words | Mixed language applications |

### Model Files Structure

```
models/wekws/
├── wenwen/
│   ├── model.onnx      # ONNX inference model
│   ├── tokens.txt      # Token vocabulary
│   └── keywords.txt    # Keyword definitions
└── zh-en-bilingual/
    ├── model.onnx
    ├── tokens.txt
    └── keywords.txt
```

## Custom Keywords Configuration

### keywords.txt Format

The keywords file defines wake words and their detection thresholds:

```
# Format: <keyword_id> <threshold> <phoneme_sequence>
# Example for "wenwen" (文文):
wenwen 0.5 wen wen

# Multiple keywords
hello 0.6 h eh l ow
world 0.5 w er l d
```

### Creating Custom Keywords

1. **Identify phonemes** for your wake word using the model's token set
2. **Set threshold** (0.0-1.0, higher = stricter detection)
3. **Test and tune** threshold for your environment

### Threshold Guidelines

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.3-0.4 | High sensitivity, more false positives | Quiet environments |
| 0.5-0.6 | Balanced (default) | General purpose |
| 0.7-0.8 | Strict, fewer false positives | Noisy environments |

## Troubleshooting

### Model Loading Errors

**Error**: `Failed to load ONNX model`

- Verify model file path is correct
- Check model file is not corrupted (compare checksum if available)
- Ensure model format matches sherpa-onnx version

**Error**: `Token file not found`

- Verify tokens.txt exists at specified path
- Check file permissions

### Detection Issues

**Problem**: Wake word not detected

- Check microphone is working: `arecord -l`
- Verify audio input levels (not too quiet)
- Lower the threshold value
- Check sample rate matches model requirements (usually 16kHz)

**Problem**: False positives

- Increase threshold value
- Check for background noise
- Verify keywords.txt is correct

### Performance Issues

**Problem**: High CPU usage

- Reduce audio buffer size
- Use quantized model if available
- Consider using a lighter model variant

**Problem**: Detection latency

- Check audio device latency
- Reduce inference batch size
- Use GPU acceleration if available (CUDA/ROCm)

### Audio Device Issues

**Error**: `No audio input device found`

```bash
# List audio devices
arecord -l

# Test recording
arecord -d 5 test.wav
```

**Error**: `Sample rate not supported`

- Most models require 16kHz sample rate
- Configure your audio device: `alsamixer`
- Use resampling if necessary

## Reference Links

- [Sherpa-ONNX GitHub](https://github.com/k2-fsa/sherpa-onnx)
- [WeKWS Documentation](https://github.com/wenet-e2e/wekws)
- [ONNX Runtime](https://onnxruntime.ai/)
- [cc-stt Project](https://github.com/yourusername/cc-stt)

## Model Sources

Pre-trained models can be downloaded from:

- [WeKWS Model Zoo](https://github.com/wenet-e2e/wekws/releases)
- [Sherpa-ONNX Pre-trained Models](https://github.com/k2-fsa/sherpa-onnx/releases)

## Configuration Reference

Full configuration options:

```json
{
  "wakeword": {
    "backend": "sherpa-onnx",
    "model_path": "path/to/model.onnx",
    "tokens_path": "path/to/tokens.txt",
    "keywords_path": "path/to/keywords.txt",
    "threshold": 0.5,
    "sample_rate": 16000,
    "num_threads": 4,
    "provider": "cpu"
  }
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| backend | string | required | Must be "sherpa-onnx" |
| model_path | string | required | Path to ONNX model file |
| tokens_path | string | required | Path to tokens.txt |
| keywords_path | string | required | Path to keywords.txt |
| threshold | float | 0.5 | Detection threshold |
| sample_rate | int | 16000 | Audio sample rate |
| num_threads | int | 4 | ONNX runtime threads |
| provider | string | "cpu" | "cpu", "cuda", or "rocm" |
