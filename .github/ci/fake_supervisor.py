"""Minimal stand-in for the Home Assistant Supervisor REST API, CI-only.

bashio::config (used by the add-on's s6 run script) hard-depends on a live
Supervisor reachable at http://supervisor for GET /addons/self/options/config
(see hassio-addons/bashio lib/apps.sh + lib/api.sh). There is no real
Supervisor in a plain CI runner, so this stub answers with exactly the
envelope bashio expects ({"result": "ok", "data": {...}}) carrying options
data.

Options are NOT hand-duplicated here: they are parsed straight out of the
real config.yaml's ``options:`` block (mounted read-only at
CONFIG_YAML_PATH), so this stub can never drift out of sync with the add-on's
actual option set the way a hard-coded copy previously did (see
ABSCHLUSSBERICHT_V0.1.2.md). Only a dependency-free parser is used here
(rather than PyYAML) because this stub runs inside a bare python:3.11-slim
container with no repo/dependencies installed.

Two scenarios are supported via environment variables, both exercised by
.github/workflows/build.yml:

- CI_MODEL: overrides the "model" option (e.g. to force the small "tiny"
  model in CI for speed).
- CI_OMIT_OPTIONS: comma-separated option keys to drop entirely from the
  response, simulating an add-on upgrade where a user's persisted
  options.json predates a newer config.yaml option -- bashio::config must
  then fall back to its own default rather than the run script crashing.
- CI_NULL_OPTIONS: comma-separated option keys whose value is replaced with
  JSON null, simulating a partially-migrated/corrupted persisted config.

This stub is not part of the shipped add-on image and is only used to
smoke-test the real /init -> s6 -> bashio startup path in CI.
"""

import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

CONFIG_YAML_PATH = os.environ.get("CONFIG_YAML_PATH", "/config.yaml")


def _coerce_scalar(raw: str) -> object:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    if raw == "true":
        return True
    if raw == "false":
        return False
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return raw


def parse_options_block(config_yaml_text: str) -> dict[str, object]:
    """Extract the flat ``options:`` mapping from an add-on config.yaml.

    Only handles the simple ``key: scalar`` shape this add-on's config.yaml
    actually uses (no lists/nested maps under options) -- deliberately not a
    general YAML parser.
    """
    lines = config_yaml_text.splitlines()
    options: dict[str, object] = {}
    in_options = False
    options_indent = None
    for line in lines:
        if not in_options:
            if re.match(r"^options:\s*$", line):
                in_options = True
            continue
        if not line.strip():
            continue
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if indent == 0:
            break  # dedented back to top level: options block ended
        if options_indent is None:
            options_indent = indent
        if indent != options_indent:
            continue  # nested value under an option we don't need
        match = re.match(r"([A-Za-z0-9_]+):\s*(.*)$", stripped)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        options[key] = _coerce_scalar(value) if value else ""
    return options


def _load_options() -> dict[str, object]:
    with open(CONFIG_YAML_PATH, encoding="utf-8") as f:
        options = parse_options_block(f.read())

    model_override = os.environ.get("CI_MODEL")
    if model_override:
        options["model"] = model_override

    for key in _split_env_list("CI_OMIT_OPTIONS"):
        options.pop(key, None)

    for key in _split_env_list("CI_NULL_OPTIONS"):
        options[key] = None

    return options


def _split_env_list(var_name: str) -> list[str]:
    raw = os.environ.get(var_name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


OPTIONS = _load_options()


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
    print(f"[fake-supervisor] loaded {len(OPTIONS)} options from {CONFIG_YAML_PATH}: {OPTIONS}")
    HTTPServer(("0.0.0.0", 80), Handler).serve_forever()
