"""Real end-to-end Wyoming round-trip test: German text in, Kokoro ONNX
audio out.

Marked ``e2e`` (excluded from the default test run, see pyproject.toml's
``addopts = "-m 'not e2e'"``) AND additionally gated on ``RUN_KOKORO_E2E=1``
(unlike Pocket TTS's own e2e test): this downloads a real, several-hundred-
MB Kokoro German model over the network on every run, so it must never run
by accident just because someone passes ``-m e2e`` for STT/Pocket TTS
testing. Run explicitly with ``RUN_KOKORO_E2E=1 pytest -m e2e``.

Same infra-failure-vs-real-regression handling as test_e2e_tts.py (see
app/tests/e2e_infra.py) -- a SKIPPED result on a version-tag release gate
(``E2E_REQUIRE_ONLINE=1``) is never proof the real model actually works.
"""

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.server import AsyncTcpServer
from wyoming.tts import Synthesize

from app.handler import MoonshineAsrHandler
from app.kokoro_session import KokoroOnnxSynthesizer
from app.kokoro_tts import load_kokoro_model
from app.tests.e2e_infra import handle_infra_or_reraise

RUN_KOKORO_E2E = os.environ.get("RUN_KOKORO_E2E") == "1"

EXPECTED_SAMPLE_RATE = 24000
EXPECTED_WIDTH = 2  # 16-bit PCM
EXPECTED_CHANNELS = 1  # mono

GERMAN_SENTENCES = (
    "Das Küchenfenster ist geöffnet.",
    "Im Wohnzimmer sind 21,5 Grad Celsius.",
)


@dataclass
class _RoundTripResult:
    pcm: bytes = b""
    audio_start_count: int = 0
    audio_chunk_count: int = 0
    audio_stop_seen: bool = False
    start_rates: list[int] = field(default_factory=list)


async def _run_synthesize_round_trip(text: str, cache_dir: Path) -> _RoundTripResult:
    kokoro_model = load_kokoro_model(cache_dir=cache_dir)
    synthesizer = KokoroOnnxSynthesizer(kokoro_model, default_voice="martin")

    def handler_factory(reader, writer) -> MoonshineAsrHandler:
        return MoonshineAsrHandler(
            reader=reader,
            writer=writer,
            transcriber=None,
            tts_synthesizer=synthesizer,
            tts_model_name=synthesizer.model_name,
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
    if not RUN_KOKORO_E2E:
        pytest.skip("Set RUN_KOKORO_E2E=1 to run the real Kokoro ONNX model download/e2e test")

    try:
        result = await _run_synthesize_round_trip(sentence, cache_dir=tmp_path)
    except Exception as e:
        handle_infra_or_reraise(e, "Kokoro ONNX model unreachable")
        return  # unreachable: handle_infra_or_reraise() always skips or raises

    assert result.audio_start_count == 1, (
        f"Expected exactly one audio-start, got {result.audio_start_count}"
    )
    assert result.start_rates == [EXPECTED_SAMPLE_RATE]
    assert result.audio_chunk_count > 0, "Expected at least one audio-chunk"
    assert result.audio_stop_seen, "Expected an audio-stop"

    pcm_bytes = result.pcm
    assert len(pcm_bytes) > 0, "Expected non-empty synthesized audio"
    assert len(pcm_bytes) % 2 == 0, "16-bit PCM must have an even byte length"

    import struct

    samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)
    assert all(-32768 <= s <= 32767 for s in samples), "PCM samples out of 16-bit range"

    duration_s = (len(pcm_bytes) / 2) / EXPECTED_SAMPLE_RATE
    assert 0.2 < duration_s < 30.0, f"Unexpected synthesized duration: {duration_s}s"


@pytest.mark.e2e
def test_ttfa_is_measurable(tmp_path: Path) -> None:
    """First-audio-chunk latency must be measurable and bounded -- a
    regression back to "generate everything, then chunk it" would still
    pass the round-trip test above but silently lose the whole point of
    using Kokoro (see ABSCHLUSSBERICHT_V0.3.0.md, "wichtigstes
    Performance-Ziel")."""
    if not RUN_KOKORO_E2E:
        pytest.skip("Set RUN_KOKORO_E2E=1 to run the real Kokoro ONNX model download/e2e test")

    import time

    async def _measure() -> float | None:
        kokoro_model = load_kokoro_model(cache_dir=tmp_path)
        synthesizer = KokoroOnnxSynthesizer(kokoro_model, default_voice="martin")
        started = time.monotonic()
        first_chunk_at: float | None = None
        async for _chunk in synthesizer.synthesize_stream(
            "Das Küchenfenster ist geöffnet.", "martin"
        ):
            if first_chunk_at is None:
                first_chunk_at = time.monotonic()
        return (first_chunk_at - started) if first_chunk_at is not None else None

    try:
        ttfa = asyncio.run(_measure())
    except Exception as e:
        handle_infra_or_reraise(e, "Kokoro ONNX model unreachable")
        return

    assert ttfa is not None, "Expected at least one audio chunk"
    assert 0.0 < ttfa < 30.0, f"Unexpected TTFA: {ttfa}s"
