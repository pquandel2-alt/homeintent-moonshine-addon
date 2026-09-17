"""Minimal stand-in for the Home Assistant Supervisor REST API, CI-only.

bashio::config (used by the add-on's s6 run script) hard-depends on a live
Supervisor reachable at http://supervisor for GET /addons/self/options/config
(see hassio-addons/bashio lib/apps.sh + lib/api.sh). There is no real
Supervisor in a plain CI runner, so this stub answers with exactly the
envelope bashio expects ({"result": "ok", "data": {...}}) carrying the same
options the add-on's config.yaml declares. It is not part of the shipped
add-on image and is only used to smoke-test the real /init -> s6 -> bashio
startup path in CI.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

OPTIONS = {
    "model": os.environ.get("CI_MODEL", "tiny"),
    "language": "de",
    "log_level": "INFO",
    "log_transcripts": False,
    "log_performance": True,
    "extra_keyterms": "",
    "keyterm_boost": 2.0,
    "use_ha_vocabulary": True,
    "ha_vocabulary_refresh_minutes": 30,
    "transcription_interval": 0.5,
    "vad_threshold": 0.5,
    "decode_incomplete_lines": True,
    "save_debug_audio": False,
    "debug_audio_max_files": 100,
}


class Handler(BaseHTTPRequestHandler):
    def _reply(self, data: object) -> None:
        body = json.dumps({"result": "ok", "data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/addons/self/options/config":
            self._reply(OPTIONS)
        else:
            self._reply({})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self._reply({})

    def log_message(self, format: str, *args: object) -> None:
        print("[fake-supervisor]", format % args)


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 80), Handler).serve_forever()
