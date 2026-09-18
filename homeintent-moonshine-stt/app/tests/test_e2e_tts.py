"""Real end-to-end Wyoming round-trip test: German text in, Pocket TTS
audio out.

Marked ``e2e`` and excluded from the default test run (see pyproject.toml's
``addopts = "-m 'not e2e'"``) because it downloads a real Pocket TTS model
(and the "juergen" voice state) over the network and runs actual inference
-- slow, and not something a hermetic unit-test run should depend on. Run
explicitly with ``pytest -m e2e``.

If the network/infrastructure is genuinely unavailable (DNS/connection
failure, HTTP error reaching Hugging Face, ...), the test is skipped with a
clear reason -- unless ``E2E_REQUIRE_ONLINE=1`` (release-gate CI on a
version tag), in which case any failure, infra or not, fails the test
outright: a SKIPPED result there is not proof the real model/voice
actually works (see app/tests/e2e_infra.py). Not run in the default CI
push/PR pipeline (see .github/workflows/build.yml's ``e2e-tts-synthesize``
job, gated to workflow_dispatch and version tags) to avoid a real model
download on every push -- see ABSCHLUSSBERICHT_V0.2.0.md.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.server import AsyncTcpServer
from wyoming.tts import Synthesize

from app.handler import MoonshineAsrHandler
from app.tests.e2e_infra import handle_infra_or_reraise
from app.tts import load_tts_model
from app.tts_session import PocketTtsSynthesizer

EXPECTED_SAMPLE_RATE = 24000
EXPECTED_WIDTH = 2  # 16-bit PCM
EXPECTED_CHANNELS = 1  # mono

# Deliberately includes German-specific formatting Pocket TTS itself is
# expected to handle without our own text normalization (B23): a time,
# a temperature, and a percentage.
GERMAN_SENTENCES = (
    "Das Licht im Wohnzimmer wurde eingeschaltet.",
    "Draussen sind es 18 Grad.",
    "Die Waschmaschine ist um 14 Uhr 35 fertig.",
)


@dataclass
class _RoundTripResult:
    pcm: bytes = b""
    audio_start_count: int = 0
    audio_chunk_count: int = 0
    audio_stop_seen: bool = False
    start_rates: list[int] = field(default_factory=list)
    start_widths: list[int] = field(default_factory=list)
    start_channels: list[int] = field(default_factory=list)


async def _run_synthesize_round_trip(text: str, cache_dir: Path) -> _RoundTripResult:
    """Start a real Wyoming TCP server backed by a real PocketTtsSynthesizer,
    send a synthesize request through a real AsyncTcpClient, and record
    every audio-start/-chunk/-stop event received.
    """
    tts_model = load_tts_model(model="german", cache_dir=cache_dir)
    synthesizer = PocketTtsSynthesizer(tts_model, default_voice="juergen")

    def handler_factory(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> MoonshineAsrHandler:
        return MoonshineAsrHandler(
            reader=reader,
            writer=writer,
            transcriber=None,
            tts_synthesizer=synthesizer,
            tts_model_name="german",
        )

    server = AsyncTcpServer("127.0.0.1", 0)
    await server.start(handler_factory)
    assert server._server is not None
    port = server._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]

    result = _RoundTripResult()
    pcm = bytearray()
    try:
        async with AsyncTcpClient("127.0.0.1", port, read_timeout=60.0) as client:
            await client.write_event(Synthesize(text=text).event())

            while True:
                event = await client.read_event()
                assert event is not None, "Server closed connection before sending audio-stop"
                if AudioStart.is_type(event.type):
                    start = AudioStart.from_event(event)
                    result.audio_start_count += 1
                    result.start_rates.append(start.rate)
                    result.start_widths.append(start.width)
                    result.start_channels.append(start.channels)
                elif AudioChunk.is_type(event.type):
                    result.audio_chunk_count += 1
                    pcm.extend(AudioChunk.from_event(event).audio)
                elif AudioStop.is_type(event.type):
                    result.audio_stop_seen = True
                    result.pcm = bytes(pcm)
                    return result
    finally:
        await server.stop()


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("sentence", GERMAN_SENTENCES)
async def test_german_text_round_trip_produces_valid_pcm(tmp_path: Path, sentence: str) -> None:
    try:
        result = await _run_synthesize_round_trip(sentence, cache_dir=tmp_path)
    except Exception as e:
        handle_infra_or_reraise(e, "Pocket TTS model/voice unreachable")
        return  # unreachable: handle_infra_or_reraise() always skips or raises

    # Exactly one audio-start per request (see app/handler.py's
    # _stream_synthesis_chunks()/AudioStart state machine fix).
    assert result.audio_start_count == 1, (
        f"Expected exactly one audio-start, got {result.audio_start_count}"
    )
    assert result.start_rates == [EXPECTED_SAMPLE_RATE]
    assert result.start_widths == [EXPECTED_WIDTH]
    assert result.start_channels == [EXPECTED_CHANNELS]
    assert result.audio_chunk_count > 0, "Expected at least one audio-chunk"
    assert result.audio_stop_seen, "Expected an audio-stop"

    pcm_bytes = result.pcm
    assert len(pcm_bytes) > 0, "Expected non-empty synthesized audio"
    assert len(pcm_bytes) % 2 == 0, "16-bit PCM must have an even byte length"

    duration_s = (len(pcm_bytes) / 2) / EXPECTED_SAMPLE_RATE
    assert 0.2 < duration_s < 30.0, f"Unexpected synthesized duration: {duration_s}s"
