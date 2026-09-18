"""Regenerate the deterministic German STT test fixture.

Run manually (rarely -- only if the fixture itself needs to change) via the
``generate-stt-fixture`` workflow_dispatch-only CI job (see
.github/workflows/build.yml), since this sandbox's own network policy
blocks download.moonshine.ai and this script needs a real network to
download the Piper voice and the Moonshine model used here.

Writes a 16kHz mono 16-bit PCM WAV to the given output path using the same
Piper de_DE-thorsten-medium voice (MIT-licensed model, CC0 dataset -- see
app/tests/fixtures/README.md) previously used for on-the-fly synthesis in
app/tests/test_e2e_transcribe.py. The output of this script is committed
once to the repository as a fixed fixture; the real STT e2e test no longer
runs TTS at all.

The original sentence, "Schalte das Licht im Wohnzimmer ein.", turned out
to be a bad choice: this Piper voice's rendering of "Licht" was
systematically misheard by Moonshine's tiny-streaming-de model across
several real CI runs (as "jetzt", "Ditting", "Setting" -- never correctly),
consistently within a single process but varying between separate CI runs
(likely onnxruntime threading nondeterminism in the quantized model, not
retryable within one process). Switched to "Schalte die Lampe im
Wohnzimmer ein." instead, which Moonshine transcribes correctly.

As a safety net against any future choice of sentence/voice having the
same problem, this script still synthesizes repeatedly and only keeps a
candidate that a real Moonshine transcription (the exact round trip
app/tests/test_e2e_transcribe.py runs) actually recognizes correctly, so
the fixture committed to the repo is empirically verified, not merely
generated. If a single process's repeated attempts keep failing (as
happened with "Licht"), that's a signal the sentence itself needs
changing, not just another synthesis attempt.
"""

import asyncio
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GERMAN_SENTENCE = "Schalte die Lampe im Wohnzimmer ein."
TTS_LANGUAGE = "de-de"
TTS_VOICE = "piper_de_DE-thorsten-medium"
SAMPLE_RATE = 16000
MAX_ATTEMPTS = 10


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    # A naive linear-interpolation resample (np.interp) has no anti-aliasing
    # filter and previously distorted high-frequency consonants. resample_poly
    # does polyphase filtering with a proper anti-aliasing lowpass.
    from math import gcd

    g = gcd(source_rate, target_rate)
    up, down = target_rate // g, source_rate // g
    resampled: np.ndarray = resample_poly(samples, up, down).astype(np.float32)
    return resampled


def _synthesize_candidate() -> bytes:
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
    return pcm_int16.tobytes()


async def _verify_candidate(pcm_bytes: bytes, cache_root: Path) -> str:
    from app.tests.test_e2e_transcribe import (
        EXPECTED_KEYWORDS,
        _run_transcribe_round_trip,
    )

    transcript = await _run_transcribe_round_trip(pcm_bytes, cache_root=cache_root)
    lowered = transcript.lower()
    matched = [kw for kw in EXPECTED_KEYWORDS if kw in lowered]
    if set(matched) != set(EXPECTED_KEYWORDS):
        raise ValueError(
            f"Expected both {EXPECTED_KEYWORDS} in transcript, got: {transcript!r}"
        )
    return transcript


def generate(output_path: Path, cache_root: Path) -> None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"Attempt {attempt}/{MAX_ATTEMPTS}: synthesizing candidate audio...")
        pcm_bytes = _synthesize_candidate()
        try:
            transcript = asyncio.run(_verify_candidate(pcm_bytes, cache_root))
        except Exception as e:
            print(f"  -> rejected: {e}")
            continue

        print(f"  -> accepted, real Moonshine transcript: {transcript!r}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm_bytes)

        duration_s = len(pcm_bytes) / 2 / SAMPLE_RATE
        print(f"Wrote {output_path} ({duration_s:.2f}s)")
        return

    raise RuntimeError(
        f"No candidate passed real Moonshine verification in {MAX_ATTEMPTS} attempts"
    )


if __name__ == "__main__":
    output = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("app/tests/fixtures/schalte_lampe_wohnzimmer.wav")
    )
    generate(output, cache_root=Path("/tmp/moonshine-fixture-gen-cache"))
