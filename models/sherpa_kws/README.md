# Sherpa-ONNX KWS Models

This directory contains pre-trained wake word detection models using the Sherpa-ONNX framework with streaming Zipformer-Transducer architecture.

## Model Format

Sherpa-ONNX KWS models consist of multiple ONNX files:

- \`encoder.onnx\`: Encoder network for acoustic feature encoding
- \`decoder.onnx\`: Decoder network for label generation
- \`joiner.onnx\`: Joiner network combining encoder and decoder outputs
- \`tokens.txt\`: Token vocabulary file mapping IDs to characters/subwords

## Input Specifications

The models expect:
- **Audio format**: Raw PCM audio at 16kHz sample rate
- **Data type**: Float32, values typically in range [-1.0, 1.0]
- **Frame size**: Variable, typically 320-640 samples (20-40ms)

Features are computed internally by the Sherpa-ONNX KeywordSpotter.

## Keywords File Format

Create a keywords file with one keyword per line:

\`\`\`
你好小德 : boost=1.0
你好小艺 : boost=1.0
小爱同学 : boost=1.0
\`\`\`

Format: \`keyword : boost=value\`
- \`keyword\`: The wake word phrase to detect (supports Chinese characters)
- \`boost\`: Confidence boost value (default: 1.0), higher values make detection more sensitive

## Obtaining Pre-trained Models

### Download from k2-fsa/sherpa-onnx

Pre-trained Chinese models are available from the Sherpa-ONNX model zoo:

\`\`\`bash
# Or use the provided download utility
python -m cc_stt.models.sherpa_kws.download --model wenetspeech --output models/sherpa_kws/
\`\`\`

### Available Models

| Model | Size | Language | Description |
|-------|------|----------|-------------|
| \`sherpa-onnx-kws-zipformer-wenetspeech-3.3M\` | 3.3M | Chinese (zh) | Trained on WenetSpeech dataset |
| \`sherpa-onnx-kws-zipformer-zh-en-3M\` | 3M | Bilingual (zh/en) | Supports both Chinese and English |

## Configuration Example

Add the following to your \`config.json\`:

\`\`\`json
{
  "wakeword": {
    "enabled": true,
    "backend": "sherpa-onnx",
    "sherpa_model_dir": "models/sherpa_kws/wenetspeech",
    "sherpa_keywords": ["你好小德", "你好小艺"],
    "sherpa_num_threads": 4,
    "sherpa_provider": "cpu"
  }
}
\`\`\`

## References

- [Sherpa-ONNX GitHub](https://github.com/k2-fsa/sherpa-onnx)
- [Sherpa-ONNX KWS Documentation](https://k2-fsa.github.io/sherpa/onnx/kws/index.html)
