"""Only the selected STT/TTS engine is ever loaded -- selecting
stt_engine=moonshine must never touch Kroko's loader (or vice versa), and
selecting a TTS engine must never touch the other two TTS loaders. This is
what actually prevents double-loading two full models into memory."""

import argparse

import app.__main__ as main_module


def _base_args(**overrides: object) -> argparse.Namespace:
    parser = main_module.build_arg_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_moonshine_selected_never_calls_kroko_loader(monkeypatch) -> None:
    called = {"kroko": False}

    def _fake_load_kroko_engine(*a: object, **k: object) -> object:
        called["kroko"] = True
        raise RuntimeError("must not be called for stt_engine=moonshine")

    monkeypatch.setattr(main_module, "load_kroko_engine", _fake_load_kroko_engine)

    class _FakeEngine:
        pass

    monkeypatch.setattr(
        main_module,
        "MoonshineSttEngine",
        lambda *a, **k: _FakeEngine(),
    )
    monkeypatch.setattr(main_module, "load_transcriber", lambda **k: object())
    monkeypatch.setattr(main_module, "_resolve_keyterms", lambda args: ([], []))

    def _fake_set_keyterms(self, terms, fallback_terms=None):
        return [], True

    monkeypatch.setattr(_FakeEngine, "set_keyterms", _fake_set_keyterms, raising=False)

    args = _base_args(stt_engine="moonshine")
    result = main_module._load_and_bias_stt_engine(args)
    assert result is not None
    assert called["kroko"] is False


def test_kroko_selected_never_calls_moonshine_load_transcriber(monkeypatch) -> None:
    called = {"moonshine": False}

    def _fake_load_transcriber(**kwargs: object) -> object:
        called["moonshine"] = True
        raise RuntimeError("must not be called for stt_engine=kroko")

    monkeypatch.setattr(main_module, "load_transcriber", _fake_load_transcriber)

    class _FakeKrokoEngine:
        def set_keyterms(self, terms, fallback_terms=None):
            return [], True

    monkeypatch.setattr(main_module, "load_kroko_engine", lambda **k: _FakeKrokoEngine())
    monkeypatch.setattr(main_module, "_resolve_keyterms", lambda args: ([], []))

    args = _base_args(stt_engine="kroko")
    result = main_module._load_and_bias_stt_engine(args)
    assert result is not None
    assert called["moonshine"] is False


def test_speechcatcher_selected_never_calls_moonshine_or_kroko_loaders(monkeypatch) -> None:
    called = {"moonshine": False, "kroko": False}

    def _fake_load_transcriber(**kwargs: object) -> object:
        called["moonshine"] = True
        raise RuntimeError("must not be called for stt_engine=speechcatcher_m")

    def _fake_load_kroko_engine(*a: object, **k: object) -> object:
        called["kroko"] = True
        raise RuntimeError("must not be called for stt_engine=speechcatcher_m")

    monkeypatch.setattr(main_module, "load_transcriber", _fake_load_transcriber)
    monkeypatch.setattr(main_module, "load_kroko_engine", _fake_load_kroko_engine)

    class _FakeSpeechcatcherEngine:
        def set_keyterms(self, terms, fallback_terms=None):
            return [], True

    monkeypatch.setattr(
        main_module, "load_speechcatcher_engine", lambda *a, **k: _FakeSpeechcatcherEngine()
    )
    monkeypatch.setattr(main_module, "_resolve_keyterms", lambda args: ([], []))

    args = _base_args(stt_engine="speechcatcher_m")
    result = main_module._load_and_bias_stt_engine(args)
    assert result is not None
    assert called == {"moonshine": False, "kroko": False}


def test_moonshine_selected_never_calls_speechcatcher_loader(monkeypatch) -> None:
    called = {"speechcatcher": False}

    def _fake_load_speechcatcher_engine(*a: object, **k: object) -> object:
        called["speechcatcher"] = True
        raise RuntimeError("must not be called for stt_engine=moonshine")

    monkeypatch.setattr(main_module, "load_speechcatcher_engine", _fake_load_speechcatcher_engine)

    class _FakeEngine:
        pass

    monkeypatch.setattr(main_module, "MoonshineSttEngine", lambda *a, **k: _FakeEngine())
    monkeypatch.setattr(main_module, "load_transcriber", lambda **k: object())
    monkeypatch.setattr(main_module, "_resolve_keyterms", lambda args: ([], []))

    def _fake_set_keyterms(self, terms, fallback_terms=None):
        return [], True

    monkeypatch.setattr(_FakeEngine, "set_keyterms", _fake_set_keyterms, raising=False)

    args = _base_args(stt_engine="moonshine")
    result = main_module._load_and_bias_stt_engine(args)
    assert result is not None
    assert called["speechcatcher"] is False


def test_pocket_tts_selected_never_calls_other_tts_loaders(monkeypatch) -> None:
    called = {"kokoro": False, "supertonic": False}
    monkeypatch.setattr(
        main_module,
        "_load_kokoro_synthesizer",
        lambda args: called.__setitem__("kokoro", True),
    )
    monkeypatch.setattr(
        main_module,
        "_load_supertonic_synthesizer",
        lambda args: called.__setitem__("supertonic", True),
    )
    monkeypatch.setattr(main_module, "_load_pocket_tts_synthesizer", lambda args: object())

    args = _base_args(tts_engine="pocket_tts")
    main_module._load_tts_synthesizer(args)
    assert called == {"kokoro": False, "supertonic": False}


def test_supertonic_selected_never_calls_other_tts_loaders(monkeypatch) -> None:
    called = {"kokoro": False, "pocket": False}
    monkeypatch.setattr(
        main_module,
        "_load_kokoro_synthesizer",
        lambda args: called.__setitem__("kokoro", True),
    )
    monkeypatch.setattr(
        main_module,
        "_load_pocket_tts_synthesizer",
        lambda args: called.__setitem__("pocket", True),
    )
    monkeypatch.setattr(main_module, "_load_supertonic_synthesizer", lambda args: object())

    args = _base_args(tts_engine="supertonic_3")
    main_module._load_tts_synthesizer(args)
    assert called == {"kokoro": False, "pocket": False}
