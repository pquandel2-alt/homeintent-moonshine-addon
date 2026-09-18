"""Real end-to-end Wyoming round-trip test: German audio in, transcript out.

Marked ``e2e`` and excluded from the default test run (see pyproject.toml's
``addopts = "-m 'not e2e'"``) because it downloads a real Moonshine STT model
and a real Piper TTS voice over the network and runs actual inference -- slow,
and not something a hermetic unit-test run should depend on. Run explicitly
with ``pytest -m e2e``.

No third-party audio is committed to this repository. The German test
utterance is synthesized on the fly with moonshine_voice's own TextToSpeech,
so there is no audio-licensing question to resolve -- nothing is shipped or
stored, only downloaded ephemerally at test time and discarded afterwards.

If the network/infrastructure is genuinely unavailable, the test is
skipped with a clear reason. Anything else is treated as a real regression
and must fail (see app/tests/e2e_infra.py).
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.server import AsyncTcpServer

from app.handler import MoonshineAsrHandler
from app.models import load_transcriber
from app.tests.e2e_infra import is_infra_failure

GERMAN_SENTENCE = "schalte das licht im wohnzimmer ein"
EXPECTED_KEYWORDS = ("licht", "wohnzimmer")

TTS_LANGUAGE = "de-de"
TTS_VOICE = "piper_de_DE-thorsten-medium"
WYOMING_SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1024  # ~64ms per chunk at 16kHz, a realistic Wyoming chunk size


def _synthesize_german_pcm() -> bytes:
    """Synthesize GERMAN_SENTENCE and return 16kHz mono 16-bit PCM bytes.

    Raises whatever moonshine_voice raises (network error, missing voice,
    etc.) -- the caller is responsible for turning that into a skip.
    """
    from moonshine_voice import TextToSpeech

    tts = TextToSpeech().language(TTS_LANGUAGE).voice(TTS_VOICE)
    try:
        tts.load()
        samples, source_rate = tts.synthesize(GERMAN_SENTENCE)
    finally:
        tts.close()

    audio = np.asarray(samples, dtype=np.float32)
    if source_rate != WYOMING_SAMPLE_RATE:
        audio = _resample_linear(audio, source_rate, WYOMING_SAMPLE_RATE)

    pcm_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    return bytes(pcm_int16.tobytes())


def _resample_linear(
    samples: np.ndarray[tuple[int], np.dtype[np.float32]], source_rate: int, target_rate: int
) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    """Numpy-only linear-interpolation resample of mono float32 audio."""
    duration_s = samples.shape[0] / source_rate
    n_target = int(round(duration_s * target_rate))
    source_times = np.arange(samples.shape[0]) / source_rate
    target_times = np.arange(n_target) / target_rate
    resampled = np.interp(target_times, source_times, samples)
    return resampled.astype(np.float32)


async def _run_transcribe_round_trip(pcm_bytes: bytes, cache_root: Path) -> str:
    """Start a real Wyoming TCP server backed by a real Transcriber, send
    ``pcm_bytes`` through a real AsyncTcpClient, and return the transcript.
    """
    transcriber = load_transcriber(model="tiny", language="de", cache_root=cache_root)

    def handler_factory(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> MoonshineAsrHandler:
        return MoonshineAsrHandler(
            reader=reader,
            writer=writer,
            transcriber=transcriber,
            model_name="tiny",
            language="de",
            moonshine_lock=asyncio.Lock(),
        )

    server = AsyncTcpServer("127.0.0.1", 0)
    await server.start(handler_factory)
    assert server._server is not None
    port = server._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]

    try:
        async with AsyncTcpClient("127.0.0.1", port, read_timeout=30.0) as client:
            await client.write_event(
                AudioStart(rate=WYOMING_SAMPLE_RATE, width=2, channels=1).event()
            )
            for offset in range(0, len(pcm_bytes), CHUNK_SAMPLES * 2):
                chunk = pcm_bytes[offset : offset + CHUNK_SAMPLES * 2]
                await client.write_event(
                    AudioChunk(rate=WYOMING_SAMPLE_RATE, width=2, channels=1, audio=chunk).event()
                )
            await client.write_event(AudioStop().event())

            while True:
                event = await client.read_event()
                assert event is not None, "Server closed connection before sending a Transcript"
                if Transcript.is_type(event.type):
                    return Transcript.from_event(event).text
    finally:
        await server.stop()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_german_audio_round_trip_produces_transcript(tmp_path: Path) -> None:
    try:
        pcm_bytes = _synthesize_german_pcm()
    except Exception as e:
        if is_infra_failure(e):
            pytest.skip(f"German TTS voice unreachable ({type(e).__name__}: {e})")
        raise

    try:
        transcript = await _run_transcribe_round_trip(pcm_bytes, cache_root=tmp_path)
    except Exception as e:
        if is_infra_failure(e):
            pytest.skip(f"Moonshine STT model unreachable ({type(e).__name__}: {e})")
        raise

    assert transcript.strip(), "Expected a non-empty transcript for real German speech"

    lowered = transcript.lower()
    matched = [kw for kw in EXPECTED_KEYWORDS if kw in lowered]
    assert matched, (
        f"Expected at least one of {EXPECTED_KEYWORDS} in transcript, got: {transcript!r}"
    )


@pytest.mark.e2e
def test_synthesized_audio_is_valid_wyoming_pcm() -> None:
    try:
        pcm_bytes = _synthesize_german_pcm()
    except Exception as e:
        if is_infra_failure(e):
            pytest.skip(f"German TTS voice unreachable ({type(e).__name__}: {e})")
        raise

    assert len(pcm_bytes) % 2 == 0
    duration_s = (len(pcm_bytes) / 2) / WYOMING_SAMPLE_RATE
    assert 0.3 < duration_s < 10.0, f"Unexpected synthesized duration: {duration_s}s"
