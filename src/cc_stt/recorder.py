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
