"""Shared helper for e2e tests (test_e2e_transcribe.py/test_e2e_tts.py): tell
a real external-infrastructure failure (network unreachable, DNS failure,
connection reset, timeout, ...) apart from an actual code/API regression.

An e2e test's whole point is to catch a real regression -- silently
swallowing every exception into a ``pytest.skip()`` would let a genuine
break in our own integration, or a breaking upstream API change, pass CI
unnoticed as a "skip" instead of failing loudly (see item 13 of the
v0.2.2 quality/stability task). Only exceptions that are unambiguously
about *reaching* the network (not what came back once reached) are treated
as infra failures; anything else -- a changed function signature, a
removed attribute, an invalid parameter, a model that no longer loads for
a reason that isn't "couldn't connect" -- is left to fail the test.
"""

import socket
from urllib.error import URLError

_INFRA_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    ConnectionError,  # covers ConnectionRefusedError/ConnectionResetError/...
    TimeoutError,
    socket.timeout,
    socket.gaierror,  # DNS resolution failure
    URLError,
)

try:
    from huggingface_hub.utils import (  # type: ignore[attr-defined]
        HfHubHTTPError,
        LocalEntryNotFoundError,
    )

    _INFRA_EXCEPTION_TYPES += (HfHubHTTPError, LocalEntryNotFoundError)
except ImportError:  # pragma: no cover - huggingface_hub always installed transitively
    pass

try:
    import requests.exceptions

    _INFRA_EXCEPTION_TYPES += (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    )
except ImportError:  # pragma: no cover - requests always installed transitively
    pass

try:
    # huggingface_hub's current transport is httpx-based; httpx.TransportError
    # is the common base for every connection-level failure (ProxyError,
    # ConnectError, ConnectTimeout, ReadTimeout, ...) and surfaces directly
    # here, sometimes before huggingface_hub gets a chance to wrap it as its
    # own HfHubHTTPError. A ProxyError specifically is exactly what this
    # sandbox's own network policy produces for a blocked host (confirmed
    # via /root/.ccr/README.md and the agent proxy's own status endpoint).
    import httpx

    _INFRA_EXCEPTION_TYPES += (httpx.TransportError,)
except ImportError:  # pragma: no cover - httpx always installed transitively
    pass


def is_infra_failure(exc: BaseException) -> bool:
    """True if ``exc`` (or anything in its __cause__/__context__ chain)
    looks like a real network/infrastructure failure rather than a code or
    API regression."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _INFRA_EXCEPTION_TYPES):
            return True
        current = current.__cause__ or current.__context__
    return False
