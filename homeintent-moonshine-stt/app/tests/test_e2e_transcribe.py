"""Real end-to-end Wyoming round-trip test: German audio in, transcript out.

Marked ``e2e`` and excluded from the default test run (see pyproject.toml's
``addopts = "-m 'not e2e'"``) because it downloads a real Moonshine STT
model over the network and runs actual inference -- slow, and not
something a hermetic unit-test run should depend on. Run explicitly with
``pytest -m e2e``.

Tests ONLY the STT path: fixed audio fixture -> real Moonshine model ->
real Wyoming server -> real Wyoming client -> transcript. This
deliberately does NOT also synthesize the test audio with a TTS voice on
every run (that used to make the test's outcome depend on two independent
models' behavior at once, and was flaky in practice -- see
ABSCHLUSSBERICHT_V0.2.3.md). Pocket TTS has its own, separate e2e test
(test_e2e_tts.py).

The audio fixture is a small, committed, deterministic WAV file -- see
app/tests/fixtures/README.md for its exact text, generation method,
source, and license. It is never regenerated as part of a normal test run
(see scripts/generate_stt_fixture.py and the ``generate-stt-fixture``
workflow_dispatch-only CI job for that).

If the network/infrastructure is genuinely unavailable, the test is
skipped with a clear reason -- unless ``E2E_REQUIRE_ONLINE=1`` (release-gate
CI on a version tag), in which case any failure, infra or not, fails the
test outright (see app/tests/e2e_infra.py).
"""

import asyncio
import wave
from pathlib import Path

import pytest
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.server import AsyncTcpServer

from app.handler import MoonshineAsrHandler
from app.models import load_transcriber
from app.tests.e2e_infra import handle_infra_or_reraise

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "schalte_licht_wohnzimmer.wav"

# The fixture's spoken sentence is "Schalte das Licht im Wohnzimmer ein."
# -- both keywords must appear for the test to pass; a real German ASR
# model recognizing this short, clear smart-home command must get both,
# not just one, or something is meaningfully wrong.
EXPECTED_KEYWORDS = ("licht", "wohnzimmer")

WYOMING_SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1024  # ~64ms per chunk at 16kHz, a realistic Wyoming chunk size


def _load_fixture_pcm() -> bytes:
    with wave.open(str(FIXTURE_PATH), "rb") as wav_file:
        assert wav_file.getnchannels() == 1, "Fixture must be mono"
        assert wav_file.getsampwidth() == 2, "Fixture must be 16-bit PCM"
        assert wav_file.getframerate() == WYOMING_SAMPLE_RATE, (
            f"Fixture must be {WYOMING_SAMPLE_RATE}Hz"
        )
        return wav_file.readframes(wav_file.getnframes())


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
    pcm_bytes = _load_fixture_pcm()

    try:
        transcript = await _run_transcribe_round_trip(pcm_bytes, cache_root=tmp_path)
    except Exception as e:
        handle_infra_or_reraise(e, "Moonshine STT model unreachable")
        return  # unreachable: handle_infra_or_reraise() always skips or raises

    assert transcript.strip(), "Expected a non-empty transcript for real German speech"

    lowered = transcript.lower()
    matched = [kw for kw in EXPECTED_KEYWORDS if kw in lowered]
    assert set(matched) == set(EXPECTED_KEYWORDS), (
        f"Expected both {EXPECTED_KEYWORDS} in transcript, got: {transcript!r}"
    )
