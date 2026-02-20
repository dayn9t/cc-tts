"""Debug test for wake word detection."""

import sys
import time
import numpy as np

print("Loading SherpaONNX backend...")
from cc_stt.wakeword import create_wakeword_backend

print("Creating backend...")
backend = create_wakeword_backend(
    backend="sherpa-onnx",
    name="小狗小狗",
    model_dir="models/sherpa_kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
    keywords_file="/home/jiang/.config/cc-stt/keywords_xiaogou.txt",
    num_threads=1,
)

print(f"Backend created: {backend}")
print(f"Wake word: {backend.wakeword}")

# Test with silence
print("\n=== Test 1: Silence ===")
silence = np.zeros(5120, dtype=np.float32)
for i in range(3):
    result = backend.process_audio(silence)
    print(f"  Call {i+1}: {result}")
    time.sleep(0.02)

# Test with noise
print("\n=== Test 2: Low noise ===")
noise = np.random.randn(5120).astype(np.float32) * 0.01
for i in range(3):
    result = backend.process_audio(noise)
    print(f"  Call {i+1}: {result}")
    time.sleep(0.02)

print("\n=== Test 3: Simulating continuous audio stream (5 seconds) ===")
print("Say '小狗小狗' now...")
start = time.time()
call_count = 0
detection_count = 0

while time.time() - start < 5:
    # Simulate audio input (silence + occasional noise)
    if np.random.random() > 0.9:
        audio = np.random.randn(5120).astype(np.float32) * 0.05
    else:
        audio = np.zeros(5120, dtype=np.float32)

    result = backend.process_audio(audio)
    call_count += 1
    if result:
        detection_count += 1
        print(f"  DETECTION at t={time.time()-start:.2f}s!")
        backend.reset()

    time.sleep(0.02)  # 50Hz

print(f"\nStats: {call_count} calls, {detection_count} detections")
print("Done.")
