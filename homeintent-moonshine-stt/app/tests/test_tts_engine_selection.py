"""Tests for TTS engine selection (app/__main__.py): choosing between
Pocket TTS and Kokoro ONNX, and the backward-compatibility guarantee that
an existing installation without ``tts_engine`` set at all keeps using
Pocket TTS exactly as before.
"""

from unittest.mock import MagicMock, patch

from app.__main__ import (
    _load_kokoro_synthesizer,
    _load_pocket_tts_synthesizer,
    _load_tts_synthesizer,
    _validate_args,
    build_arg_parser,
)
from app.kokoro_tts import KokoroModelDownloadError


def _parse(argv: list[str]):
    return build_arg_parser().parse_args(argv)


class TestDefaultEngine:
    def test_default_tts_engine_is_pocket_tts(self):
        """Backward compatibility: an existing v0.2.7 config predates this
        option entirely -- it must resolve to Pocket TTS, never silently
        switch engines on upgrade."""
        args = _parse([])
        assert args.tts_engine == "pocket_tts"

    def test_explicit_pocket_tts(self):
        args = _parse(["--tts-engine", "pocket_tts"])
        assert args.tts_engine == "pocket_tts"

    def test_explicit_kokoro_onnx(self):
        args = _parse(["--tts-engine", "kokoro_onnx"])
        assert args.tts_engine == "kokoro_onnx"

    def test_default_kokoro_voice_is_martin(self):
        args = _parse([])
        assert args.kokoro_voice == "martin"

    def test_default_kokoro_speed_and_pauses_match_upstream(self):
        """Defaults verified directly against kokoro_onnx.Kokoro.create_stream()'s
        own real signature, not guessed."""
        args = _parse([])
        assert args.kokoro_speed == 1.0
        assert args.kokoro_sentence_pause == 0.25
        assert args.kokoro_clause_pause == 0.1
        assert args.kokoro_threads == 0


class TestLoadTtsSynthesizerDispatch:
    def test_pocket_tts_engine_calls_pocket_loader_only(self):
        args = _parse(["--tts-engine", "pocket_tts"])
        with (
            patch("app.__main__._load_pocket_tts_synthesizer") as mock_pocket,
            patch("app.__main__._load_kokoro_synthesizer") as mock_kokoro,
        ):
            mock_pocket.return_value = MagicMock()
            result = _load_tts_synthesizer(args)
            mock_pocket.assert_called_once_with(args)
            mock_kokoro.assert_not_called()
            assert result is mock_pocket.return_value

    def test_kokoro_onnx_engine_calls_kokoro_loader_only(self):
        args = _parse(["--tts-engine", "kokoro_onnx"])
        with (
            patch("app.__main__._load_pocket_tts_synthesizer") as mock_pocket,
            patch("app.__main__._load_kokoro_synthesizer") as mock_kokoro,
        ):
            mock_kokoro.return_value = MagicMock()
            result = _load_tts_synthesizer(args)
            mock_kokoro.assert_called_once_with(args)
            mock_pocket.assert_not_called()
            assert result is mock_kokoro.return_value


class TestLoadKokoroSynthesizer:
    def test_model_download_failure_returns_none_no_fallback(self, caplog):
        """A Kokoro download/load failure must be a visible, fatal startup
        error -- never a silent fallback to Pocket TTS."""
        args = _parse(["--tts-engine", "kokoro_onnx"])
        with (
            patch(
                "app.__main__.load_kokoro_model",
                side_effect=KokoroModelDownloadError("network unreachable"),
            ),
            caplog.at_level("ERROR"),
        ):
            result = _load_kokoro_synthesizer(args)
        assert result is None
        assert "Kokoro" in caplog.text

    def test_invalid_voice_returns_none(self):
        args = _parse(["--tts-engine", "kokoro_onnx", "--kokoro-voice", "not-a-real-voice"])
        fake_kokoro_model = MagicMock()
        with (
            patch("app.__main__.load_kokoro_model", return_value=fake_kokoro_model),
            patch("app.kokoro_session.KokoroOnnxSynthesizer.preload_default_voice") as mock_preload,
        ):
            mock_preload.side_effect = ValueError("Kokoro voice 'not-a-real-voice' not found")
            result = _load_kokoro_synthesizer(args)
        assert result is None

    def test_successful_load_returns_synthesizer_with_configured_voice(self):
        args = _parse(["--tts-engine", "kokoro_onnx", "--kokoro-voice", "martin"])
        fake_kokoro_model = MagicMock()
        with (
            patch("app.__main__.load_kokoro_model", return_value=fake_kokoro_model) as mock_load,
            patch("app.kokoro_session.KokoroOnnxSynthesizer.preload_default_voice"),
        ):
            args.tts_warmup = False
            result = _load_kokoro_synthesizer(args)
        assert result is not None
        assert result.default_voice == "martin"
        mock_load.assert_called_once()

    def test_threads_forwarded_to_load_kokoro_model(self):
        args = _parse(["--tts-engine", "kokoro_onnx", "--kokoro-threads", "4"])
        args.tts_warmup = False
        with (
            patch("app.__main__.load_kokoro_model", return_value=MagicMock()) as mock_load,
            patch("app.kokoro_session.KokoroOnnxSynthesizer.preload_default_voice"),
        ):
            _load_kokoro_synthesizer(args)
        assert mock_load.call_args.kwargs["intra_op_num_threads"] == 4

    def test_warmup_disabled_skips_warmup(self):
        args = _parse(["--tts-engine", "kokoro_onnx", "--no-tts-warmup"])
        with (
            patch("app.__main__.load_kokoro_model", return_value=MagicMock()),
            patch("app.kokoro_session.KokoroOnnxSynthesizer.preload_default_voice"),
            patch("app.kokoro_session.KokoroOnnxSynthesizer.warmup") as mock_warmup,
        ):
            result = _load_kokoro_synthesizer(args)
        assert result is not None
        mock_warmup.assert_not_called()

    def test_warmup_failure_is_not_fatal(self, caplog):
        args = _parse(["--tts-engine", "kokoro_onnx"])
        with (
            patch("app.__main__.load_kokoro_model", return_value=MagicMock()),
            patch("app.kokoro_session.KokoroOnnxSynthesizer.preload_default_voice"),
            patch(
                "app.kokoro_session.KokoroOnnxSynthesizer.warmup",
                side_effect=RuntimeError("warmup boom"),
            ),
            caplog.at_level("WARNING"),
        ):
            result = _load_kokoro_synthesizer(args)
        assert result is not None  # warmup failure never fails startup


class TestLoadPocketTtsSynthesizerUnaffected:
    """Guards against a regression where generalizing the loader for two
    engines accidentally changed Pocket TTS's own behavior."""

    def test_invalid_pocket_voice_returns_none(self):
        args = _parse(["--tts-engine", "pocket_tts", "--tts-voice", "not-a-real-voice"])
        fake_model = MagicMock()
        with (
            patch("app.__main__.load_tts_model", return_value=fake_model),
            patch(
                "app.tts_session.PocketTtsSynthesizer.preload_default_voice",
                side_effect=ValueError("not found"),
            ),
        ):
            result = _load_pocket_tts_synthesizer(args)
        assert result is None

    def test_model_name_forwarded_to_synthesizer(self):
        args = _parse(["--tts-engine", "pocket_tts", "--tts-model", "german_24l"])
        args.tts_warmup = False
        fake_model = MagicMock()
        with (
            patch("app.__main__.load_tts_model", return_value=fake_model),
            patch("app.tts_session.PocketTtsSynthesizer.preload_default_voice"),
        ):
            result = _load_pocket_tts_synthesizer(args)
        assert result is not None
        assert result.model_name == "german_24l"


class TestValidateArgs:
    def test_invalid_tts_engine_rejected(self):
        args = _parse([])
        args.tts_engine = "not_a_real_engine"
        assert _validate_args(args) is False

    def test_kokoro_speed_out_of_range_rejected(self):
        args = _parse([])
        args.kokoro_speed = 3.0
        assert _validate_args(args) is False

    def test_kokoro_speed_in_range_accepted(self):
        args = _parse([])
        args.kokoro_speed = 1.25
        assert _validate_args(args) is True

    def test_kokoro_threads_out_of_range_rejected(self):
        args = _parse([])
        args.kokoro_threads = -1
        assert _validate_args(args) is False

    def test_valid_default_args_pass(self):
        args = _parse([])
        assert _validate_args(args) is True


class TestNoBrokenVoiceEngineCombination:
    """Item 5's explicit requirement: a combination like
    "Kokoro engine + Pocket voice 'juergen'" must never silently pass
    through -- each engine only ever reads its own voice option, so
    there is structurally no way for Pocket's tts_voice to leak into a
    Kokoro request or vice versa."""

    def test_kokoro_engine_never_reads_tts_voice_option(self):
        args = _parse(
            ["--tts-engine", "kokoro_onnx", "--tts-voice", "juergen", "--kokoro-voice", "martin"]
        )
        fake_kokoro_model = MagicMock()
        with (
            patch("app.__main__.load_kokoro_model", return_value=fake_kokoro_model),
            patch("app.kokoro_session.KokoroOnnxSynthesizer.preload_default_voice"),
        ):
            args.tts_warmup = False
            result = _load_kokoro_synthesizer(args)
        assert result is not None
        assert result.default_voice == "martin"  # never "juergen"
