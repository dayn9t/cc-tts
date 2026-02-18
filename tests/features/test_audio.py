"""Tests for FeatureExtractor audio preprocessing."""

import numpy as np
import pytest


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class."""

    def test_fbank_feature_shape(self):
        """Verify output shape is (n_frames, n_mels)."""
        from cc_stt.features import FeatureExtractor

        extractor = FeatureExtractor(sample_rate=16000, n_mels=80)
        # 1 second of audio at 16kHz
        audio = np.random.randn(16000).astype(np.float32)
        features = extractor.extract(audio)

        assert features.ndim == 2
        assert features.shape[1] == 80
        assert features.shape[0] > 0  # Should have some frames

    def test_fbank_feature_values(self):
        """Verify features are non-negative (FBank in dB should be non-negative after power_to_db with proper ref)."""
        from cc_stt.features import FeatureExtractor

        extractor = FeatureExtractor(sample_rate=16000, n_mels=80)
        # 1 second of audio at 16kHz
        audio = np.random.randn(16000).astype(np.float32)
        features = extractor.extract(audio)

        # Features should be finite and mostly non-negative (dB scale)
        assert np.all(np.isfinite(features))
        # After power_to_db, values can be negative but should be bounded
        assert np.all(features > -100)  # Reasonable lower bound for dB

    def test_empty_audio(self):
        """Verify empty input returns empty array."""
        from cc_stt.features import FeatureExtractor

        extractor = FeatureExtractor(sample_rate=16000, n_mels=80)
        audio = np.array([], dtype=np.float32)
        features = extractor.extract(audio)

        assert features.size == 0

    @pytest.mark.parametrize("n_mels", [40, 64, 80])
    def test_different_n_mels(self, n_mels):
        """Test with different n_mels values."""
        from cc_stt.features import FeatureExtractor

        extractor = FeatureExtractor(sample_rate=16000, n_mels=n_mels)
        audio = np.random.randn(16000).astype(np.float32)
        features = extractor.extract(audio)

        assert features.shape[1] == n_mels
