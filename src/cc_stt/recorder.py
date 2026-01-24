import sounddevice as sd
import numpy as np

class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames: list[np.ndarray] = []

    def record(self, max_duration: int = 30, silence_threshold: float = 0.01,
               silence_duration: float = 2.0) -> np.ndarray:
        self.frames = []
        silence_frames = int(silence_duration * self.sample_rate / 1024)
        silent_chunks = 0

        def callback(indata, frames, time, status):
            nonlocal silent_chunks
            rms = np.sqrt(np.mean(indata**2))
            if rms < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0
            self.frames.append(indata.copy())
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

        return np.concatenate(self.frames, axis=0).flatten()

    def get_audio_devices(self) -> list[dict]:
        devices = sd.query_devices()
        return [
            {'id': i, 'name': d['name'], 'channels': d['max_input_channels']}
            for i, d in enumerate(devices) if d['max_input_channels'] > 0
        ]
