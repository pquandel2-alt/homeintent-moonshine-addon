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
from wyoming.server import AsyncTcpServer
from wyoming.tts import Synthesize

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
