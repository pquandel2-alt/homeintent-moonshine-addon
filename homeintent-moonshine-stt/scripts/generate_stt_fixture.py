"""Regenerate the deterministic German STT test fixture.

Run manually (rarely -- only if the fixture itself needs to change) via the
``generate-stt-fixture`` workflow_dispatch-only CI job (see
.github/workflows/build.yml), since this sandbox's own network policy
blocks download.moonshine.ai and this script needs a real network to
download the Piper voice used for synthesis.

Writes a 16kHz mono 16-bit PCM WAV to the given output path using the same
Piper de_DE-thorsten-medium voice (MIT-licensed model, CC0 dataset -- see
app/tests/fixtures/README.md) previously used for on-the-fly synthesis in
app/tests/test_e2e_transcribe.py. The output of this script is committed
once to the repository as a fixed fixture; the real STT e2e test no longer
runs TTS at all.
"""

import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

GERMAN_SENTENCE = "Schalte das Licht im Wohnzimmer ein."
TTS_LANGUAGE = "de-de"
TTS_VOICE = "piper_de_DE-thorsten-medium"
SAMPLE_RATE = 16000


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    # A naive linear-interpolation resample (np.interp) has no anti-aliasing
    # filter: it previously produced a fixture that visibly distorted
    # high-frequency consonants (e.g. Moonshine misheard "Licht" as "jetzt"
    # from a linearly-resampled fixture, but transcribed the vowel-heavy
    # "Wohnzimmer" correctly). resample_poly does polyphase filtering with a
    # proper anti-aliasing lowpass, which a naive interpolation lacks.
    from math import gcd

    g = gcd(source_rate, target_rate)
    up, down = target_rate // g, source_rate // g
    resampled: np.ndarray = resample_poly(samples, up, down).astype(np.float32)
    return resampled


def generate(output_path: Path) -> None:
    from moonshine_voice import TextToSpeech

    tts = TextToSpeech().language(TTS_LANGUAGE).voice(TTS_VOICE)
    try:
        tts.load()
        samples, source_rate = tts.synthesize(GERMAN_SENTENCE)
    finally:
        tts.close()

    audio = np.asarray(samples, dtype=np.float32)
    if source_rate != SAMPLE_RATE:
        audio = _resample(audio, source_rate, SAMPLE_RATE)

    pcm_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_int16.tobytes())

    duration_s = len(pcm_int16) / SAMPLE_RATE
    print(f"Wrote {output_path} ({duration_s:.2f}s, {len(pcm_int16)} samples)")


if __name__ == "__main__":
    output = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("app/tests/fixtures/schalte_licht_wohnzimmer.wav")
    )
    generate(output)
