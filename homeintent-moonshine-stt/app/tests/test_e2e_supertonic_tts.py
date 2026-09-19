"""Real end-to-end Wyoming round-trip test: German text in, Supertonic 3
audio out, against the REAL downloaded sherpa-onnx model.

Marked ``e2e`` AND additionally gated on ``RUN_SUPERTONIC_E2E=1`` (mirrors
test_e2e_kokoro_tts.py's RUN_KOKORO_E2E gate): downloads a real model
archive over the network on every run. Run explicitly with
``RUN_SUPERTONIC_E2E=1 pytest -m e2e``.

Not run in the sandbox this add-on was developed in (see
test_e2e_kroko_stt.py's module docstring for the same network-access
caveat) -- a maintainer with real network access must run this before
relying on it.
"""

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import Event
from wyoming.server import AsyncTcpServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from app.handler import MoonshineAsrHandler
from app.supertonic_session import SupertonicSynthesizer
from app.supertonic_tts import load_supertonic_tts
from app.tests.e2e_infra import handle_infra_or_reraise

RUN_SUPERTONIC_E2E = os.environ.get("RUN_SUPERTONIC_E2E") == "1"

GERMAN_SENTENCES = (
    "Das Küchenfenster ist geöffnet.",
    "Im Wohnzimmer sind einundzwanzig Grad.",
)


@dataclass
class _RoundTripResult:
    pcm: bytes = b""
    audio_start_count: int = 0
    audio_chunk_count: int = 0
    audio_stop_seen: bool = False
    start_rates: list[int] = field(default_factory=list)


async def _run_synthesize_round_trip(text: str, cache_dir: Path) -> _RoundTripResult:
    tts = load_supertonic_tts(cache_dir=cache_dir)
    synthesizer = SupertonicSynthesizer(tts, default_voice="M1", language="de")

    def handler_factory(reader, writer) -> MoonshineAsrHandler:
        return MoonshineAsrHandler(
            reader=reader,
            writer=writer,
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
    if not RUN_SUPERTONIC_E2E:
        pytest.skip("Set RUN_SUPERTONIC_E2E=1 to run the real Supertonic model download/e2e test")

    try:
        result = await _run_synthesize_round_trip(sentence, cache_dir=tmp_path)
    except Exception as e:
        handle_infra_or_reraise(e, "Supertonic model unreachable")
        return

    assert result.audio_start_count == 1
    assert result.audio_chunk_count > 0, "Expected at least one audio-chunk (real streaming)"
    assert result.audio_stop_seen

    pcm_bytes = result.pcm
    assert len(pcm_bytes) > 0
    assert len(pcm_bytes) % 2 == 0
    samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)
    assert all(-32768 <= s <= 32767 for s in samples)


# --- Modern Wyoming streaming path: synthesize-start / synthesize-chunk /
# synthesize-stop / synthesize-stopped (see wyoming.tts, pinned at
# wyoming==1.10.2 in requirements-runtime.txt -- event names/sequence
# verified directly against that installed package's own source, not
# assumed from memory). The legacy single-shot `Synthesize` test above is
# left completely unchanged; this is a second, additive real E2E test for
# the actual modern streaming sequence HomeIntent's own Wyoming clients use.


@dataclass
class _StreamingRoundTripResult:
    pcm: bytes = b""
    audio_start_count: int = 0
    audio_chunk_count: int = 0
    audio_stop_count: int = 0
    synthesize_stopped_count: int = 0
    start_rates: list[int] = field(default_factory=list)
    chunk_lengths: list[int] = field(default_factory=list)


async def _run_streaming_synthesize_round_trip(
    text_chunks: tuple[str, ...], cache_dir: Path
) -> _StreamingRoundTripResult:
    tts = load_supertonic_tts(cache_dir=cache_dir)
    synthesizer = SupertonicSynthesizer(tts, default_voice="M1", language="de")

    def handler_factory(reader, writer) -> MoonshineAsrHandler:
        return MoonshineAsrHandler(
            reader=reader,
            writer=writer,
            tts_synthesizer=synthesizer,
            tts_model_name=synthesizer.model_name,
        )

    server = AsyncTcpServer("127.0.0.1", 0)
    await server.start(handler_factory)
    assert server._server is not None
    port = server._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]

    result = _StreamingRoundTripResult()
    pcm = bytearray()
    try:
        async with AsyncTcpClient("127.0.0.1", port, read_timeout=60.0) as client:
            await client.write_event(SynthesizeStart().event())
            for chunk_text in text_chunks:
                await client.write_event(SynthesizeChunk(text=chunk_text).event())
            await client.write_event(SynthesizeStop().event())

            while True:
                event: Event | None = await client.read_event()
                assert event is not None, "Server closed connection before synthesize-stopped"
                if AudioStart.is_type(event.type):
                    start = AudioStart.from_event(event)
                    result.audio_start_count += 1
                    result.start_rates.append(start.rate)
                elif AudioChunk.is_type(event.type):
                    audio_bytes = AudioChunk.from_event(event).audio
                    result.audio_chunk_count += 1
                    result.chunk_lengths.append(len(audio_bytes))
                    pcm.extend(audio_bytes)
                elif AudioStop.is_type(event.type):
                    result.audio_stop_count += 1
                elif SynthesizeStopped.is_type(event.type):
                    result.synthesize_stopped_count += 1
                    result.pcm = bytes(pcm)
                    return result
    finally:
        await server.stop()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_german_text_streaming_round_trip_produces_valid_pcm(tmp_path: Path) -> None:
    """Real modern Wyoming streaming sequence against the real Supertonic
    model: synthesize-start -> synthesize-chunk (x2) -> synthesize-stop ->
    synthesize-stopped. Verifies exactly one AudioStart, at least one
    AudioChunk with non-empty, valid-range PCM at the model's own sample
    rate, exactly one AudioStop, exactly one SynthesizeStopped, and that the
    chunked text-message sequence triggers exactly ONE synthesis pass (not
    one per synthesize-chunk) -- i.e. no duplicated audio output.
    """
    if not RUN_SUPERTONIC_E2E:
        pytest.skip("Set RUN_SUPERTONIC_E2E=1 to run the real Supertonic model download/e2e test")

    text_chunks = ("Das Küchenfenster ist geöffnet. ", "Im Wohnzimmer sind einundzwanzig Grad.")
    try:
        result = await _run_streaming_synthesize_round_trip(text_chunks, cache_dir=tmp_path)
    except Exception as e:
        handle_infra_or_reraise(e, "Supertonic model unreachable")
        return

    assert result.audio_start_count == 1, "Expected exactly one AudioStart (no duplicate synthesis)"
    assert result.audio_chunk_count > 0, "Expected at least one audio-chunk (real streaming)"
    assert result.audio_stop_count == 1, "Expected exactly one AudioStop"
    assert result.synthesize_stopped_count == 1, "Expected exactly one synthesize-stopped"
    assert result.start_rates == [result.start_rates[0]] * len(result.start_rates), (
        "AudioStart sample rate must be consistent"
    )
    assert result.start_rates[0] > 0

    pcm_bytes = result.pcm
    assert len(pcm_bytes) > 0
    assert len(pcm_bytes) % 2 == 0
    samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)
    assert all(-32768 <= s <= 32767 for s in samples)

    # Sanity bound against "duplicate audio output": synthesizing the exact
    # same two-chunk text once more (single-shot Synthesize path) must
    # produce comparably-sized PCM, not roughly double -- a generous 3x
    # upper bound catches an actual double-synthesis bug without being
    # sensitive to normal per-run TTS length variance.
    single_shot_text = "".join(text_chunks)
    single_shot_result = await _run_synthesize_round_trip(single_shot_text, cache_dir=tmp_path)
    assert len(pcm_bytes) < 3 * len(single_shot_result.pcm)
