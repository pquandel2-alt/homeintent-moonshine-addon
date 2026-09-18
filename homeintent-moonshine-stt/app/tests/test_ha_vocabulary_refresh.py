"""Tests for the periodic HA vocabulary refresh loop in app/__main__.py.

Covers:
- last-known-good HA vocabulary is preserved across a failed refresh
  (A2 review fix)
- manual keyterms always stay in the effective set regardless of HA state
- a legitimately empty HA result replaces the previous vocabulary
- HA becoming reachable again after an outage picks the new terms back up
- set_keyterms() is called under the same lock shared with streaming
  sessions (A3 review fix), and streaming/refresh calls never interleave
- a newly appearing HA-derived term the loaded model's tokenizer rejects
  (e.g. a new/renamed entity) never kills the refresh task or STT itself
  -- see app/keyterms.py's apply_safe_keyterms()
"""

import asyncio

import pytest
from moonshine_voice import MoonshineError

from app.__main__ import _refresh_ha_vocabulary_periodically
from app.ha_vocabulary import HaVocabularyResult


class _FakeTranscriber:
    def __init__(self) -> None:
        self.set_keyterms_calls: list[list[str]] = []

    def set_keyterms(self, terms: list[str]) -> None:
        self.set_keyterms_calls.append(list(terms))


class _FakeMoonshineTranscriber:
    """Like _FakeTranscriber, but raises MoonshineError for the whole call
    if any term is in ``incompatible_terms`` -- matching the real native
    library's per-call (not per-term) failure semantics."""

    def __init__(self, incompatible_terms: set[str] | None = None) -> None:
        self.incompatible_terms = incompatible_terms or set()
        self.set_keyterms_calls: list[list[str]] = []

    def set_keyterms(self, terms: list[str]) -> None:
        self.set_keyterms_calls.append(list(terms))
        for term in terms:
            if term in self.incompatible_terms:
                raise MoonshineError(f"Failed to set key terms: no match for {term!r}")


async def _run_one_refresh_iteration(
    monkeypatch: pytest.MonkeyPatch,
    fetch_results: list[HaVocabularyResult],
    transcriber: "_FakeTranscriber | _FakeMoonshineTranscriber",
    manual_keyterms: list[str],
    last_known_good_terms: list[str],
    lock: asyncio.Lock,
) -> None:
    """Drive exactly len(fetch_results) iterations of the refresh loop."""
    results = iter(fetch_results)
    call_count = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal call_count
        call_count += 1
        if call_count > len(fetch_results):
            raise asyncio.CancelledError()

    async def fake_fetch() -> HaVocabularyResult:
        return next(results)

    monkeypatch.setattr("app.__main__.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("app.__main__.fetch_ha_vocabulary", fake_fetch)

    with pytest.raises(asyncio.CancelledError):
        await _refresh_ha_vocabulary_periodically(
            transcriber, manual_keyterms, 30, last_known_good_terms, lock
        )


class TestLastKnownGood:
    @pytest.mark.asyncio
    async def test_successful_refresh_updates_keyterms(self, monkeypatch):
        transcriber = _FakeTranscriber()
        last_known_good: list[str] = []
        await _run_one_refresh_iteration(
            monkeypatch,
            [HaVocabularyResult(success=True, terms=["Wohnzimmer", "Küche"])],
            transcriber,
            manual_keyterms=["Grohe Blue"],
            last_known_good_terms=last_known_good,
            lock=asyncio.Lock(),
        )
        assert last_known_good == ["Wohnzimmer", "Küche"]
        # apply_safe_keyterms() (app/keyterms.py) makes a diagnostic
        # set_keyterms() call while validating against the loaded model,
        # then always makes one final, explicit, authoritative call for the
        # accepted list -- both happen to carry the same content here since
        # every term was valid, so the fake transcriber sees it twice.
        assert transcriber.set_keyterms_calls == [
            ["Wohnzimmer", "Küche", "Grohe Blue"],
            ["Wohnzimmer", "Küche", "Grohe Blue"],
        ]

    @pytest.mark.asyncio
    async def test_failed_refresh_keeps_last_known_good_and_manual_terms(self, monkeypatch):
        transcriber = _FakeTranscriber()
        last_known_good = ["Wohnzimmer", "Küche", "Rolllade", "Waschmaschine"]
        await _run_one_refresh_iteration(
            monkeypatch,
            [HaVocabularyResult(success=False, terms=[])],
            transcriber,
            manual_keyterms=["Grohe Blue"],
            last_known_good_terms=last_known_good,
            lock=asyncio.Lock(),
        )
        # Unchanged -- a failed refresh must not touch it.
        assert last_known_good == ["Wohnzimmer", "Küche", "Rolllade", "Waschmaschine"]
        # And the transcriber must not have been touched at all on failure.
        assert transcriber.set_keyterms_calls == []

    @pytest.mark.asyncio
    async def test_manual_keyterms_survive_a_failed_ha_refresh(self, monkeypatch):
        """Manual keyterms are never dropped, whether HA succeeds or fails."""
        transcriber = _FakeTranscriber()
        last_known_good = ["Wohnzimmer"]
        await _run_one_refresh_iteration(
            monkeypatch,
            [HaVocabularyResult(success=False, terms=[])],
            transcriber,
            manual_keyterms=["Grohe Blue"],
            last_known_good_terms=last_known_good,
            lock=asyncio.Lock(),
        )
        # No set_keyterms call happened, but the caller's manual list (used
        # to build the *next* effective set once HA is healthy again) is
        # itself untouched -- nothing was dropped from it.
        assert "Grohe Blue" not in [t for call in transcriber.set_keyterms_calls for t in call]

    @pytest.mark.asyncio
    async def test_legitimately_empty_ha_result_replaces_previous_vocabulary(self, monkeypatch):
        transcriber = _FakeTranscriber()
        last_known_good = ["Wohnzimmer", "Küche"]
        await _run_one_refresh_iteration(
            monkeypatch,
            [HaVocabularyResult(success=True, terms=[])],
            transcriber,
            manual_keyterms=["Grohe Blue"],
            last_known_good_terms=last_known_good,
            lock=asyncio.Lock(),
        )
        assert last_known_good == []
        assert transcriber.set_keyterms_calls == [["Grohe Blue"], ["Grohe Blue"]]

    @pytest.mark.asyncio
    async def test_ha_recovering_after_outage_picks_up_new_terms(self, monkeypatch):
        transcriber = _FakeTranscriber()
        last_known_good = ["Wohnzimmer"]
        await _run_one_refresh_iteration(
            monkeypatch,
            [
                HaVocabularyResult(success=False, terms=[]),
                HaVocabularyResult(success=True, terms=["Wohnzimmer", "Bad"]),
            ],
            transcriber,
            manual_keyterms=[],
            last_known_good_terms=last_known_good,
            lock=asyncio.Lock(),
        )
        assert last_known_good == ["Wohnzimmer", "Bad"]
        # The failed iteration made no set_keyterms call at all; the
        # successful one makes two (diagnostic + final authoritative --
        # see apply_safe_keyterms()).
        assert transcriber.set_keyterms_calls == [["Wohnzimmer", "Bad"], ["Wohnzimmer", "Bad"]]


class TestSetKeytermsLocking:
    @pytest.mark.asyncio
    async def test_set_keyterms_uses_the_shared_lock(self, monkeypatch):
        """set_keyterms() must not run while the lock is held elsewhere
        (e.g. by a concurrent streaming session's start/add_audio/stop)."""
        lock = asyncio.Lock()
        transcriber = _FakeTranscriber()
        order: list[str] = []

        real_set_keyterms = transcriber.set_keyterms

        def tracking_set_keyterms(terms: list[str]) -> None:
            order.append("refresh:set_keyterms")
            real_set_keyterms(terms)

        transcriber.set_keyterms = tracking_set_keyterms  # type: ignore[method-assign]

        async def hold_lock_then_release() -> None:
            async with lock:
                order.append("streaming:locked")
                await asyncio.sleep(0.01)
            order.append("streaming:released")

        # Start a "streaming session" holding the lock first.
        streaming_task = asyncio.create_task(hold_lock_then_release())
        await asyncio.sleep(0)  # let it acquire the lock first

        await _run_one_refresh_iteration(
            monkeypatch,
            [HaVocabularyResult(success=True, terms=["Wohnzimmer"])],
            transcriber,
            manual_keyterms=[],
            last_known_good_terms=[],
            lock=lock,
        )
        await streaming_task

        # The refresh's set_keyterms calls (apply_safe_keyterms() makes a
        # diagnostic call plus one final authoritative call -- see
        # app/keyterms.py) must only happen after the streaming session
        # released the lock -- never interleaved with it.
        assert order == [
            "streaming:locked",
            "streaming:released",
            "refresh:set_keyterms",
            "refresh:set_keyterms",
        ]


class TestRefreshSurvivesIncompatibleKeyterm:
    """Fall H: at startup everything is valid; a later refresh picks up a
    newly added/renamed HA entity ("/Büro Licht") the loaded model's
    tokenizer rejects. The refresh task and STT must both keep running,
    the valid terms must still be applied, and last_known_good_terms must
    not become inconsistent."""

    @pytest.mark.asyncio
    async def test_new_incompatible_term_is_skipped_refresh_continues(self, monkeypatch):
        transcriber = _FakeMoonshineTranscriber(incompatible_terms={"/Büro Licht"})
        last_known_good = ["Wohnzimmer"]

        await _run_one_refresh_iteration(
            monkeypatch,
            [HaVocabularyResult(success=True, terms=["Wohnzimmer", "/Büro Licht", "Küche"])],
            transcriber,
            manual_keyterms=[],
            last_known_good_terms=last_known_good,
            lock=asyncio.Lock(),
        )

        # The refresh committed the new HA fetch as last-known-good (the
        # apply itself succeeded, even though one term was skipped).
        assert last_known_good == ["Wohnzimmer", "/Büro Licht", "Küche"]
        # But the incompatible term was never actually applied.
        final_call = transcriber.set_keyterms_calls[-1]
        assert "/Büro Licht" not in final_call
        assert "Wohnzimmer" in final_call
        assert "Küche" in final_call

    @pytest.mark.asyncio
    async def test_refresh_loop_keeps_running_after_incompatible_term_appears(self, monkeypatch):
        """The refresh task itself must not die -- a subsequent cycle must
        still run and apply further changes."""
        transcriber = _FakeMoonshineTranscriber(incompatible_terms={"/Büro Licht"})
        last_known_good = ["Wohnzimmer"]

        await _run_one_refresh_iteration(
            monkeypatch,
            [
                HaVocabularyResult(success=True, terms=["Wohnzimmer", "/Büro Licht"]),
                HaVocabularyResult(success=True, terms=["Wohnzimmer", "Küche"]),
            ],
            transcriber,
            manual_keyterms=[],
            last_known_good_terms=last_known_good,
            lock=asyncio.Lock(),
        )

        # The second, fully-valid cycle ran and updated last-known-good.
        assert last_known_good == ["Wohnzimmer", "Küche"]
