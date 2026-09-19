"""Tests for app/speechcatcher_model.py: tag resolution and error handling.

Never installs or imports the real (heavy, git-only) speechcatcher package
-- ``load_speechcatcher_model`` imports it lazily via
``importlib.import_module("speechcatcher.speechcatcher")``, so a fake
module can be injected into ``sys.modules`` under that exact name before
calling it, the same technique test_kroko_model.py uses to avoid a real
sherpa-onnx/model download."""

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from app import speechcatcher_model


class _FakeSpeechcatcherCli(types.ModuleType):
    """Mimics speechcatcher.speechcatcher's own surface used by this
    add-on: the `tags` dict and `load_model()` function."""

    def __init__(self, *, raise_on_load: Exception | None = None) -> None:
        super().__init__("speechcatcher.speechcatcher")
        self.tags = {
            "de_streaming_transformer_m": "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_m_raw_de_bpe1024",
            "de_streaming_transformer_l": "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_l_raw_de_bpe1024",
        }
        self.load_calls: list[dict[str, Any]] = []
        self._raise_on_load = raise_on_load

    def load_model(self, **kwargs: Any) -> Any:
        self.load_calls.append(kwargs)
        if self._raise_on_load is not None:
            raise self._raise_on_load
        return object()


def _inject_fake_speechcatcher(
    monkeypatch: pytest.MonkeyPatch, fake: _FakeSpeechcatcherCli
) -> None:
    # importlib.import_module("speechcatcher.speechcatcher") requires the
    # parent package "speechcatcher" to exist in sys.modules too.
    parent = types.ModuleType("speechcatcher")
    monkeypatch.setitem(sys.modules, "speechcatcher", parent)
    monkeypatch.setitem(sys.modules, "speechcatcher.speechcatcher", fake)


def test_load_speechcatcher_model_m_resolves_correct_tag_and_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeSpeechcatcherCli()
    _inject_fake_speechcatcher(monkeypatch, fake)

    result = speechcatcher_model.load_speechcatcher_model(
        "speechcatcher_m", cache_dir=tmp_path, beam_size=7
    )
    assert result is not None
    assert len(fake.load_calls) == 1
    call = fake.load_calls[0]
    assert call["tag"] == (
        "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_m_raw_de_bpe1024"
    )
    assert call["device"] == "cpu"
    assert call["beam_size"] == 7
    assert call["decoder_impl"] == "espnet"
    assert call["cache_dir"] == str(tmp_path)


def test_load_speechcatcher_model_l_resolves_correct_tag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeSpeechcatcherCli()
    _inject_fake_speechcatcher(monkeypatch, fake)

    speechcatcher_model.load_speechcatcher_model("speechcatcher_l", cache_dir=tmp_path)
    assert fake.load_calls[0]["tag"] == (
        "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_l_raw_de_bpe1024"
    )


def test_load_speechcatcher_model_unknown_engine_id_raises(tmp_path: Path) -> None:
    with pytest.raises(speechcatcher_model.SpeechcatcherModelDownloadError):
        speechcatcher_model.load_speechcatcher_model("speechcatcher_xl", cache_dir=tmp_path)


def test_load_speechcatcher_model_import_failure_raises_download_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delitem(sys.modules, "speechcatcher.speechcatcher", raising=False)
    monkeypatch.delitem(sys.modules, "speechcatcher", raising=False)

    def _fail_import(name: str) -> Any:
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr("importlib.import_module", _fail_import)

    with pytest.raises(speechcatcher_model.SpeechcatcherModelDownloadError):
        speechcatcher_model.load_speechcatcher_model("speechcatcher_m", cache_dir=tmp_path)


def test_load_speechcatcher_model_load_failure_raises_download_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeSpeechcatcherCli(raise_on_load=RuntimeError("network unreachable"))
    _inject_fake_speechcatcher(monkeypatch, fake)

    with pytest.raises(speechcatcher_model.SpeechcatcherModelDownloadError):
        speechcatcher_model.load_speechcatcher_model("speechcatcher_m", cache_dir=tmp_path)


def test_load_speechcatcher_model_creates_cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeSpeechcatcherCli()
    _inject_fake_speechcatcher(monkeypatch, fake)
    cache_dir = tmp_path / "speechcatcher"
    assert not cache_dir.exists()

    speechcatcher_model.load_speechcatcher_model("speechcatcher_m", cache_dir=cache_dir)
    assert cache_dir.is_dir()
