"""Docker HEALTHCHECK: a real Wyoming describe round-trip against the running server.

Uses wyoming.client.AsyncTcpClient (the same library the server itself runs on) so this
speaks the actual length-prefixed wire protocol instead of a bare JSON line, and validates
the "info" response rather than grepping raw bytes.
"""

import asyncio
import os
import sys

from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info

HOST = os.environ.get("MOONSHINE_HEALTHCHECK_HOST", "localhost")
# Overridable only for CI, where two add-on containers under test are
# published on different host ports (10300, 10301) on the same runner; the
# Docker HEALTHCHECK itself always runs inside the container at 10300.
PORT = int(os.environ.get("MOONSHINE_HEALTHCHECK_PORT", "10300"))
TIMEOUT = 5.0


async def check() -> bool:
    async with AsyncTcpClient(HOST, PORT, connect_timeout=TIMEOUT, read_timeout=TIMEOUT) as client:
        await client.write_event(Describe().event())
        while True:
            event = await client.read_event()
            if event is None:
                return False
            if Info.is_type(event.type):
                return True


def main() -> int:
    try:
        ok = asyncio.run(check())
    except (TimeoutError, OSError) as err:
        print(f"healthcheck: {err}", file=sys.stderr)
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
