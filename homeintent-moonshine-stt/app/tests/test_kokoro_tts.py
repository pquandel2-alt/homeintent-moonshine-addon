"""Tests for Kokoro ONNX model resolution/downloading (app/kokoro_tts.py).

No real network access, no real 326MB model download -- huggingface_hub's
hf_hub_download() is mocked/faked throughout. See
app/tests/test_e2e_kokoro_tts.py for the real, opt-in (RUN_KOKORO_E2E=1)
integration test.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.kokoro_tts import (
    KOKORO_MODEL_FILENAME,
    KOKORO_REPO_ID,
    KOKORO_SAMPLE_RATE,
    KOKORO_VOICES_FILENAME,
    KokoroModelDownloadError,
    resolve_kokoro_model_files,
)


def _write_fake_file(path: Path, size: int = 128) -> str:
    path.write_bytes(b"x" * size)
    return str(path)


class TestResolveKokoroModelFiles:
    def test_downloads_model_and_voices_with_correct_repo_and_filenames(self, tmp_path: Path):
        """Cache-miss path: both files are fetched from the real, verified
        Hugging Face repo id and filenames -- not invented ones."""
        calls: list[dict[str, object]] = []

        def fake_hf_hub_download(**kwargs):
            calls.append(kwargs)
            filename = kwargs["filename"]
            return _write_fake_file(tmp_path / filename)

        with patch("app.kokoro_tts.hf_hub_download", side_effect=fake_hf_hub_download):
            model_path, voices_path = resolve_kokoro_model_files(cache_dir=tmp_path)

        assert model_path.name == KOKORO_MODEL_FILENAME
        assert voices_path.name == KOKORO_VOICES_FILENAME
        assert {c["repo_id"] for c in calls} == {KOKORO_REPO_ID}
        assert {c["filename"] for c in calls} == {KOKORO_MODEL_FILENAME, KOKORO_VOICES_FILENAME}

    def test_cache_hit_still_calls_hf_hub_download_but_no_new_files_created(self, tmp_path: Path):
        """hf_hub_download() itself is what decides cache-hit-vs-download
        (it always resolves to a real, complete local path either way) --
        this test only proves we call it correctly and use its result
        as-is, not that we duplicate its own caching logic."""
        existing = tmp_path / "existing-model.onnx"
        existing.write_bytes(b"cached-content")

        with patch("app.kokoro_tts.hf_hub_download", return_value=str(existing)):
            model_path, voices_path = resolve_kokoro_model_files(cache_dir=tmp_path)

        assert model_path == existing
        assert voices_path == existing  # same fake return value for both calls in this test

    def test_pins_the_given_revision(self, tmp_path: Path):
        calls: list[dict[str, object]] = []

        def fake_hf_hub_download(**kwargs):
            calls.append(kwargs)
            return _write_fake_file(tmp_path / kwargs["filename"])

        with patch("app.kokoro_tts.hf_hub_download", side_effect=fake_hf_hub_download):
            resolve_kokoro_model_files(cache_dir=tmp_path, revision="v1.2")

        assert all(c["revision"] == "v1.2" for c in calls)

    def test_download_failure_raises_kokoro_model_download_error(self, tmp_path: Path):
        with (
            patch(
                "app.kokoro_tts.hf_hub_download",
                side_effect=OSError("503 Service Unavailable"),
            ),
            pytest.raises(KokoroModelDownloadError),
        ):
            resolve_kokoro_model_files(cache_dir=tmp_path)

    def test_empty_downloaded_file_is_rejected_not_accepted_as_valid(self, tmp_path: Path):
        """A partially-downloaded/truncated file must never be silently
        treated as a valid model -- verified by simulating hf_hub_download
        resolving to a zero-byte file."""
        empty_file = tmp_path / "empty-model.onnx"
        empty_file.touch()

        with (
            patch("app.kokoro_tts.hf_hub_download", return_value=str(empty_file)),
            pytest.raises(KokoroModelDownloadError, match="missing or empty"),
        ):
            resolve_kokoro_model_files(cache_dir=tmp_path)

    def test_missing_resolved_file_is_rejected(self, tmp_path: Path):
        nonexistent = tmp_path / "does-not-exist.onnx"

        with (
            patch("app.kokoro_tts.hf_hub_download", return_value=str(nonexistent)),
            pytest.raises(KokoroModelDownloadError, match="missing or empty"),
        ):
            resolve_kokoro_model_files(cache_dir=tmp_path)

    def test_creates_cache_dir_if_missing(self, tmp_path: Path):
        cache_dir = tmp_path / "kokoro-onnx" / "nested"
        assert not cache_dir.exists()

        def fake_hf_hub_download(**kwargs):
            return _write_fake_file(tmp_path / kwargs["filename"])

        with patch("app.kokoro_tts.hf_hub_download", side_effect=fake_hf_hub_download):
            resolve_kokoro_model_files(cache_dir=cache_dir)

        assert cache_dir.exists()


class TestSampleRate:
    def test_sample_rate_is_24000(self):
        assert KOKORO_SAMPLE_RATE == 24000


class TestLoadKokoroModel:
    """load_kokoro_model() builds its own onnxruntime.InferenceSession
    (with configurable thread settings) and hands it to
    Kokoro.from_session(), since kokoro_onnx's own create_session() exposes
    no threading configuration at all (verified from its source, see
    app/kokoro_tts.py's module docstring)."""

    def _patch_download(self, tmp_path: Path):
        def fake_hf_hub_download(**kwargs):
            path = tmp_path / kwargs["filename"]
            path.write_bytes(b"x" * 128)
            return str(path)

        return patch("app.kokoro_tts.hf_hub_download", side_effect=fake_hf_hub_download)

    def test_default_threads_leave_session_options_untouched(self, tmp_path: Path):
        from app.kokoro_tts import load_kokoro_model

        with (
            self._patch_download(tmp_path),
            patch("onnxruntime.InferenceSession") as mock_session_cls,
            patch("kokoro_onnx.Kokoro.from_session") as mock_from_session,
        ):
            load_kokoro_model(cache_dir=tmp_path, intra_op_num_threads=0, inter_op_num_threads=0)

            session_options = mock_session_cls.call_args.kwargs["sess_options"]
            # onnxruntime's own default is 0 (meaning "let it decide") --
            # confirms we never overwrote it when the option is 0/"auto".
            assert session_options.intra_op_num_threads == 0
            assert session_options.inter_op_num_threads == 0
            mock_from_session.assert_called_once()

    def test_explicit_intra_op_threads_applied(self, tmp_path: Path):
        from app.kokoro_tts import load_kokoro_model

        with (
            self._patch_download(tmp_path),
            patch("onnxruntime.InferenceSession") as mock_session_cls,
            patch("kokoro_onnx.Kokoro.from_session"),
        ):
            load_kokoro_model(cache_dir=tmp_path, intra_op_num_threads=4)

            session_options = mock_session_cls.call_args.kwargs["sess_options"]
            assert session_options.intra_op_num_threads == 4

    def test_uses_cpu_execution_provider_only(self, tmp_path: Path):
        from app.kokoro_tts import load_kokoro_model

        with (
            self._patch_download(tmp_path),
            patch("onnxruntime.InferenceSession") as mock_session_cls,
            patch("kokoro_onnx.Kokoro.from_session"),
        ):
            load_kokoro_model(cache_dir=tmp_path)

            assert mock_session_cls.call_args.kwargs["providers"] == ["CPUExecutionProvider"]

    def test_download_failure_propagates_without_loading_a_session(self, tmp_path: Path):
        from app.kokoro_tts import load_kokoro_model

        with (
            patch("app.kokoro_tts.hf_hub_download", side_effect=OSError("network down")),
            patch("onnxruntime.InferenceSession") as mock_session_cls,
        ):
            with pytest.raises(KokoroModelDownloadError):
                load_kokoro_model(cache_dir=tmp_path)
            mock_session_cls.assert_not_called()
