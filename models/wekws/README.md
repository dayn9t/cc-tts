# WeKWS Models

This directory contains pre-trained wake word detection models using the WeKWS framework.

## Model Format

Models are stored in **ONNX format** (`.onnx`) for cross-platform compatibility and efficient inference.

## Input Specifications

The models expect the following input tensor shape:

```
(batch_size, window_size, n_mels)
```

Where:
- `batch_size`: Number of audio frames to process (typically 1 for real-time inference)
- `window_size`: Number of frames in the sliding window (model-dependent, typically 100-300)
- `n_mels`: Number of mel-frequency bins (typically 40-80)

Input features should be **log-mel spectrogram** features extracted from audio at 16kHz sample rate.

## Obtaining Pre-trained Models

### Download from WeKWS Releases

Pre-trained models can be downloaded from the official WeKWS repository releases:

```bash
# Example: Download a pre-trained model
wget https://github.com/wenet-e2e/wekws/releases/download/v0.1/your-model.onnx -O models/wekws/your-model.onnx
```

Visit the [WeKWS GitHub Releases](https://github.com/wenet-e2e/wekws/releases) page for available models.

## Training Custom Models

To train a custom wake word model:

1. **Prepare your dataset** with audio recordings of your wake word and negative samples
2. **Follow the WeKWS training guide**: https://github.com/wenet-e2e/wekws
3. **Export to ONNX** after training:

```bash
# Export trained model to ONNX format
python tools/export_onnx.py --config config.yaml --checkpoint model.pt --output model.onnx
```

4. **Copy the ONNX model** to this directory

## Configuration Example

Add the following to your `config.json`:

```json
{
  "wakeword": {
    "enabled": true,
    "model_path": "models/wekws/your-model.onnx",
    "threshold": 0.8,
    "window_size": 200,
    "n_mels": 64,
    "sample_rate": 16000,
    "frame_shift_ms": 10,
    "frame_length_ms": 25
  }
}
```

Configuration parameters:
- `model_path`: Path to the ONNX model file
- `threshold`: Detection threshold (0.0-1.0), higher values reduce false positives
- `window_size`: Number of frames in the sliding window (must match model training)
- `n_mels`: Number of mel bins (must match model training)
- `sample_rate`: Audio sample rate in Hz (typically 16000)
- `frame_shift_ms`: Frame shift in milliseconds
- `frame_length_ms`: Frame length in milliseconds

## Supported Wake Words

The following wake words have pre-trained models available:

| Wake Word | Model File | Description |
|-----------|------------|-------------|
| "Hey Siri" | `hey_siri.onnx` | Apple-style wake word |
| "OK Google" | `ok_google.onnx` | Google-style wake word |
| "Alexa" | `alexa.onnx` | Amazon-style wake word |
| "Hi Bixby" | `hi_bixby.onnx` | Samsung-style wake word |

Custom wake words can be trained using the WeKWS framework with your own dataset.

## Model Storage

Model files are tracked using **Git LFS** due to their large size. Ensure Git LFS is installed before committing models:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.onnx"
```

See `../.gitattributes` for the LFS configuration.
