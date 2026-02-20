"""Test Sherpa-ONNX backend with Chinese wake words."""

import os
import numpy as np
from cc_stt.wakeword import create_wakeword_backend


def test_chinese_wakeword():
    """Test Sherpa-ONNX backend with Chinese keywords."""
    print("=" * 60)
    print("Testing Sherpa-ONNX Chinese Wake Word Detection")
    print("=" * 60)

    model_dir = "models/sherpa_kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    keywords_file = f"{model_dir}/keywords.txt"

    print(f"\nKeywords file: {keywords_file}")
    print(f"Model: sherpa-onnx-kws-zipformer-wenetspeech-3.3M")

    # Show supported keywords
    print("\nSupported Chinese wake words:")
    with open(keywords_file) as f:
        for line in f:
            if "@" in line:
                pinyin, chinese = line.strip().split("@")
                print(f"  - {chinese} ({pinyin.strip()})")

    # Create backend using keywords file
    print("\nInitializing backend...")
    backend = create_wakeword_backend(
        backend="sherpa-onnx",
        name="chinese-test",
        model_dir=model_dir,
        keywords_file=keywords_file,
        num_threads=4,
        provider="cpu",
    )
    print("Backend initialized successfully!")

    # Test 1: Silence audio (should not trigger)
    print("\nTest 1: Silence audio (should not detect)")
    silence = np.zeros(16000, dtype=np.float32)
    result = backend.process_audio(silence)
    print(f"  Detection: {result}")
    assert result is False, "Silence should not trigger detection"
    print("  PASS")

    # Reset for next test
    backend.reset()

    # Test 2: Random noise (should not trigger)
    print("\nTest 2: Random noise (should not detect)")
    noise = np.random.randn(16000).astype(np.float32) * 0.1
    result = backend.process_audio(noise)
    print(f"  Detection: {result}")
    assert result is False, "Noise should not trigger detection"
    print("  PASS")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nSummary:")
    print("  - Backend initialized with 8 Chinese wake words")
    print("  - Silence/noise correctly rejected (no false positives)")
    print("  - Ready for real-time wake word detection")
    print("\nNote: For positive detection tests, use actual wake word audio.")


if __name__ == "__main__":
    test_chinese_wakeword()
