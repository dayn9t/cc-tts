"""Audio feature extraction using librosa."""

import numpy as np
import librosa


class FeatureExtractor:
    """Extract FBank (mel-frequency filterbank) features from audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        n_fft: int | None = None,
    ):
        """Initialize FeatureExtractor.

        Args:
            sample_rate: Audio sample rate in Hz.
            n_mels: Number of mel filterbank bins.
            frame_length: Frame length in milliseconds.
            frame_shift: Frame shift in milliseconds.
            n_fft: FFT size. If None, calculated from frame_length.
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.n_fft = n_fft or int(frame_length * sample_rate / 1000)
        self.hop_length = int(frame_shift * sample_rate / 1000)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract FBank features from audio.

        Args:
            audio: Input audio array of shape (n_samples,).

        Returns:
            FBank features of shape (n_frames, n_mels) as float32.
        """
        if audio.size == 0:
            return np.array([], dtype=np.float32).reshape(0, self.n_mels)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        # Convert to dB scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Transpose to (n_frames, n_mels) and convert to float32
        features = log_mel_spec.T.astype(np.float32)

        return features
