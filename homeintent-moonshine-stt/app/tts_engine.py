"""Generic TTS engine abstraction shared by every synthesizer backend.

app/handler.py depends only on this structural interface, never on a
concrete engine -- it is what lets Pocket TTS (app/tts_session.py) and
Kokoro ONNX (app/kokoro_session.py) sit behind the exact same Wyoming
synthesize-start/-chunk/-stop/-stopped handling (app/tts_stream.py) without
the handler ever branching on which engine is active. Every field an
engine needs to answer a Wyoming ``describe`` request (see
handler.py's ``_handle_describe``) or the performance log (see
``_log_tts_performance``) lives on the synthesizer object itself, not
hard-coded per engine in the handler.

Extracted out of app/tts_session.py (which used to hold both this generic
interface and the Pocket-TTS-specific implementation) when a second engine
(Kokoro ONNX) was added, so the generic pieces have one home instead of
living inside a file named after one specific engine.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from wyoming.info import Attribution


@dataclass
class TtsSynthesisStats:
    """Per-request timing breakdown, populated by synthesize_stream() as it
    runs. A fresh instance per request -- not shared or reused across
    calls, and safe to read only after the request has finished (normally
    or with an error), since the values are written incrementally while
    the request is in flight (from a background thread for Pocket TTS, an
    asyncio task for Kokoro ONNX -- see each engine's own module).

    Only the synthesizer itself can distinguish real model compute time
    from time spent waiting for the shared lock: both are opaque to
    app/handler.py, which measures the outer request wall-time and its own
    Wyoming I/O time instead (see app/handler.py's _SynthesisStats).
    """

    lock_wait_seconds: float = 0.0
    model_generation_seconds: float = 0.0


class TtsSynthesizer(Protocol):
    """Structural interface app/handler.py depends on for TTS.

    Implemented identically by every engine (PocketTtsSynthesizer,
    KokoroOnnxSynthesizer, ...) so the handler and Wyoming service
    discovery never need an ``if engine == ...`` branch: they read
    whichever synthesizer was loaded at startup through this interface
    alone. Lets tests substitute a lightweight fake without constructing a
    real model.
    """

    @property
    def engine_id(self) -> str:
        """Stable machine identifier, e.g. "pocket_tts"/"kokoro_onnx" --
        matches the add-on's own ``tts_engine`` option values."""
        ...

    @property
    def sample_rate(self) -> int: ...

    @property
    def default_voice(self) -> str: ...

    @property
    def model_name(self) -> str:
        """Short model identifier for logging (e.g. "german"/
        "german-martin"), never the full description."""
        ...

    @property
    def program_name(self) -> str:
        """Wyoming ``TtsProgram.name`` advertised in ``describe``, e.g.
        "homeintent-pocket-tts"/"homeintent-kokoro-onnx"."""
        ...

    @property
    def attribution(self) -> Attribution:
        """Upstream model/runtime attribution for Wyoming ``describe`` --
        must be the actual engine's own upstream, never another engine's
        (e.g. Kokoro must never show Kyutai's attribution)."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description for Wyoming ``describe``."""
        ...

    def synthesize_stream(
        self, text: str, voice: str | None = None, stats: TtsSynthesisStats | None = None
    ) -> AsyncGenerator[np.ndarray, None]:
        """Yield mono float32 numpy chunks as the engine generates them,
        real incremental streaming (not "generate everything, then chunk
        it") -- see each engine's own module for the upstream API this
        wraps."""
        ...
