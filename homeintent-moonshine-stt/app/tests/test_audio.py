"""Tests for audio format conversion."""

import numpy as np
import pytest

from app.audio import float32_to_pcm_int16, pcm_int16_to_float32, validate_audio_format


class TestPcmConversion:
    """Tests for PCM int16 to float32 conversion."""

    def test_silent_audio(self):
        """Test conversion of silent (zero) audio."""
        pcm_bytes = b"\x00\x00" * 100  # 100 samples of silence
        result = pcm_int16_to_float32(pcm_bytes)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 100
        assert np.allclose(result, 0.0)

    def test_max_positive(self):
        """Test conversion of maximum positive PCM value (32767)."""
        # 32767 in int16 (big-endian: 0x7FFF)
        pcm_bytes = b"\xff\x7f"  # Little-endian: 0x7FFF = 32767
        result = pcm_int16_to_float32(pcm_bytes)
        expected = 32767.0 / 32768.0  # ~0.99997
        assert np.isclose(result[0], expected)

    def test_max_negative(self):
        """Test conversion of minimum (most negative) PCM value (-32768)."""
        # -32768 in int16 (little-endian: 0x8000)
        pcm_bytes = b"\x00\x80"  # Little-endian: 0x8000 = -32768
        result = pcm_int16_to_float32(pcm_bytes)
        expected = -32768.0 / 32768.0  # -1.0
        assert np.isclose(result[0], expected)

    def test_multiple_samples(self):
        """Test conversion of multiple samples."""
        # Create test data: [0, 32767, -32768, 16384]
        pcm_bytes = (
            b"\x00\x00"  # 0
            b"\xff\x7f"  # 32767
            b"\x00\x80"  # -32768
            b"\x00\x40"  # 16384
        )
        result = pcm_int16_to_float32(pcm_bytes)
        assert len(result) == 4
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 32767.0 / 32768.0)
        assert np.isclose(result[2], -1.0)
        assert np.isclose(result[3], 16384.0 / 32768.0)

    def test_odd_length_bytes(self):
        """Test that odd-length byte sequence raises ValueError."""
        pcm_bytes = b"\x00"  # Only 1 byte (not even)
        with pytest.raises(ValueError, match="even length"):
            pcm_int16_to_float32(pcm_bytes)

    def test_empty_audio(self):
        """Test conversion of empty audio."""
        result = pcm_int16_to_float32(b"")
        assert len(result) == 0
        assert result.dtype == np.float32

    def test_range_normalization(self):
        """Test that all values are normalized to [-1.0, 1.0]."""
        # Create random int16 samples
        samples_int16 = np.random.randint(-32768, 32767, size=1000, dtype=np.int16)
        pcm_bytes = samples_int16.tobytes()
        result = pcm_int16_to_float32(pcm_bytes)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


class TestFloat32ToPcmInt16:
    """Tests for float32 -> int16 PCM conversion (Pocket TTS output, B8)."""

    def test_silence_round_trips_to_zero(self):
        pcm = float32_to_pcm_int16(np.zeros(10, dtype=np.float32))
        assert pcm == b"\x00\x00" * 10

    def test_positive_full_scale(self):
        pcm = float32_to_pcm_int16(np.array([1.0], dtype=np.float32))
        result = np.frombuffer(pcm, dtype=np.int16)
        assert result[0] == 32767

    def test_negative_full_scale(self):
        pcm = float32_to_pcm_int16(np.array([-1.0], dtype=np.float32))
        result = np.frombuffer(pcm, dtype=np.int16)
        assert result[0] == -32767

    def test_clips_overshoot_instead_of_wrapping(self):
        """A model value slightly outside [-1.0, 1.0] must clip cleanly,
        not wrap around to the opposite sign (which raw casting would do)."""
        pcm = float32_to_pcm_int16(np.array([1.5, -1.5], dtype=np.float32))
        result = np.frombuffer(pcm, dtype=np.int16)
        assert result[0] == 32767
        assert result[1] == -32767

    def test_output_byte_length_matches_sample_count(self):
        samples = np.linspace(-1.0, 1.0, num=137, dtype=np.float32)
        pcm = float32_to_pcm_int16(samples)
        assert len(pcm) == 137 * 2

    def test_empty_input_produces_empty_output(self):
        pcm = float32_to_pcm_int16(np.array([], dtype=np.float32))
        assert pcm == b""

    def test_no_clipping_within_range(self):
        samples = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        pcm = float32_to_pcm_int16(samples)
        result = np.frombuffer(pcm, dtype=np.int16)
        np.testing.assert_array_equal(result, (samples * 32767.0).astype(np.int16))


class TestAudioFormatValidation:
    """Tests for Wyoming audio format validation."""

    def test_valid_format(self):
        """Test validation of correct Wyoming format (16kHz, 16-bit, mono)."""
        error = validate_audio_format(rate=16000, width=2, channels=1)
        assert error is None

    def test_wrong_sample_rate(self):
        """Test validation rejects wrong sample rate."""
        error = validate_audio_format(rate=44100, width=2, channels=1)
        assert error is not None
        assert "44100Hz" in error

    def test_wrong_width(self):
        """Test validation rejects wrong sample width."""
        error = validate_audio_format(rate=16000, width=4, channels=1)
        assert error is not None
        assert "4 bytes" in error

    def test_stereo_rejected(self):
        """Test validation rejects stereo (2 channels)."""
        error = validate_audio_format(rate=16000, width=2, channels=2)
        assert error is not None
        assert "2" in error

    def test_custom_context_in_error(self):
        """Test that context string appears in error message."""
        error = validate_audio_format(rate=8000, width=2, channels=1, context="audio-start")
        assert error is not None
        assert "audio-start" in error
