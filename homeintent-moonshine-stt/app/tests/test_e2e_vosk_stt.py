"""Real end-to-end Wyoming round-trip test: German audio in, Vosk
transcript out, against the REAL downloaded vosk-model-small-de-0.15 model.

Marked ``e2e`` (excluded from the default test run, see pyproject.toml's
``addopts = "-m 'not e2e'"``) AND additionally gated on ``RUN_VOSK_E2E=1``
(mirrors test_e2e_kroko_stt.py's own RUN_KROKO_E2E gate): this downloads a
real ~45MB model archive over the network on every run, so it must never
run by accident. Run explicitly with ``RUN_VOSK_E2E=1 pytest -m e2e``.

This test was NOT run in the sandbox this add-on was developed in --
alphacephei.com was blocked by that sandbox's own egress policy for
unauthenticated downloads (only a git clone of alphacep/vosk-api and the
alphacep/vosk-space catalogue repo were reachable, which is how the model
name/API/download-URL convention were verified -- see
app/vosk_model.py's module docstring). It must be run by a maintainer
with real network access before relying on it.
"""

import asyncio
import os
from pathlib import Path

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.server import AsyncTcpServer

from app.handler import MoonshineAsrHandler
from app.tests.e2e_infra import handle_infra_or_reraise
from app.vosk_engine import load_vosk_engine

RUN_VOSK_E2E = os.environ.get("RUN_VOSK_E2E") == "1"

EXPECTED_SAMPLE_RATE = 16000


async def _run_transcribe_round_trip(wav_pcm: bytes, cache_dir: Path) -> str:
    engine = load_vosk_engine(cache_dir=cache_dir)

    def handler_factory(reader, writer) -> MoonshineAsrHandler:
        return MoonshineAsrHandler(reader=reader, writer=writer, stt_engine=engine)

    server = AsyncTcpServer("127.0.0.1", 0)
    await server.start(handler_factory)
    assert server._server is not None
    port = server._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]

    try:
        async with AsyncTcpClient("127.0.0.1", port, read_timeout=60.0) as client:
            await client.write_event(Transcribe().event())
            await client.write_event(
                AudioStart(rate=EXPECTED_SAMPLE_RATE, width=2, channels=1).event()
            )
            chunk_size = 3200
            for i in range(0, len(wav_pcm), chunk_size):
                await client.write_event(
                    AudioChunk(
                        rate=EXPECTED_SAMPLE_RATE,
                        width=2,
                        channels=1,
                        audio=wav_pcm[i : i + chunk_size],
                    ).event()
                )
            await client.write_event(AudioStop().event())

            while True:
                event = await client.read_event()
                assert event is not None
                if Transcript.is_type(event.type):
                    return Transcript.from_event(event).text
    finally:
        await server.stop()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_vosk_transcribes_real_german_fixture(tmp_path: Path) -> None:
    if not RUN_VOSK_E2E:
        pytest.skip("Set RUN_VOSK_E2E=1 to run the real Vosk model download/e2e test")

    fixture = Path(__file__).parent / "fixtures" / "schalte_lampe_wohnzimmer.wav"
    wav_bytes = fixture.read_bytes()
    # Strip the 44-byte canonical WAV header -- the fixture is documented
    # (fixtures/README.md) as 16kHz/16-bit/mono PCM, matching Vosk's own
    # expected sample rate.
    pcm = wav_bytes[44:]

    try:
        text = await asyncio.wait_for(_run_transcribe_round_trip(pcm, tmp_path), timeout=120)
    except Exception as e:
        handle_infra_or_reraise(e, "Vosk model unreachable")
        return

    assert isinstance(text, str)
    assert len(text) > 0, "Expected a non-empty transcript from a real spoken command"
