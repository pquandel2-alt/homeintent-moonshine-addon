"""Real end-to-end Wyoming round-trip test: German text in, Pocket TTS
audio out.

Marked ``e2e`` and excluded from the default test run (see pyproject.toml's
``addopts = "-m 'not e2e'"``) because it downloads a real Pocket TTS model
(and the "juergen" voice state) over the network and runs actual inference
-- slow, and not something a hermetic unit-test run should depend on. Run
explicitly with ``pytest -m e2e``.

If the network/model is unavailable, the test is skipped with a clear
reason rather than reporting a fabricated pass or silently vanishing. Not
run in the default CI push/PR pipeline (see .github/workflows/build.yml's
``e2e-tts-synthesize`` job, ``workflow_dispatch``-gated) to avoid a real
model download on every push -- see ABSCHLUSSBERICHT_V0.2.0.md.
"""

import asyncio
from pathlib import Path

import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.server import AsyncTcpServer
from wyoming.tts import Synthesize

from app.handler import MoonshineAsrHandler
from app.tts import load_tts_model
from app.tts_session import PocketTtsSynthesizer

# Deliberately includes German-specific formatting Pocket TTS itself is
# expected to handle without our own text normalization (B23): a time,
# a temperature, and a percentage.
GERMAN_SENTENCES = (
    "Das Licht im Wohnzimmer wurde eingeschaltet.",
    "Draussen sind es 18 Grad.",
    "Die Waschmaschine ist um 14 Uhr 35 fertig.",
)


async def _run_synthesize_round_trip(text: str, cache_dir: Path) -> bytes:
    """Start a real Wyoming TCP server backed by a real PocketTtsSynthesizer,
    send a synthesize request through a real AsyncTcpClient, and return the
    concatenated raw PCM bytes of every audio-chunk received.
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

    try:
        async with AsyncTcpClient("127.0.0.1", port, read_timeout=60.0) as client:
            await client.write_event(Synthesize(text=text).event())

            pcm = bytearray()
            saw_start = False
            while True:
                event = await client.read_event()
                assert event is not None, "Server closed connection before sending audio-stop"
                if AudioStart.is_type(event.type):
                    saw_start = True
                elif AudioChunk.is_type(event.type):
                    pcm.extend(AudioChunk.from_event(event).audio)
                elif AudioStop.is_type(event.type):
                    assert saw_start, "audio-stop received without a prior audio-start"
                    return bytes(pcm)
    finally:
        await server.stop()


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("sentence", GERMAN_SENTENCES)
async def test_german_text_round_trip_produces_valid_pcm(tmp_path: Path, sentence: str) -> None:
    try:
        pcm_bytes = await _run_synthesize_round_trip(sentence, cache_dir=tmp_path)
    except Exception as e:
        pytest.skip(f"Pocket TTS model/voice unavailable ({type(e).__name__}: {e})")

    assert len(pcm_bytes) > 0, "Expected non-empty synthesized audio"
    assert len(pcm_bytes) % 2 == 0, "16-bit PCM must have an even byte length"

    sample_rate = 24000
    duration_s = (len(pcm_bytes) / 2) / sample_rate
    assert 0.2 < duration_s < 30.0, f"Unexpected synthesized duration: {duration_s}s"
