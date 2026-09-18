"""Unit tests for the pure streaming-TTS state machine (app/tts_stream.py).

No I/O, no Wyoming connection, no Pocket TTS -- just the bookkeeping that
prevents Home Assistant's backwards-compatible whole-message `synthesize`
event from causing text to be synthesized (and spoken) twice. See
app/tests/test_handler_tts.py for the full Wyoming-event-level tests.
"""

from app.tts_stream import TtsStreamPhase, TtsStreamState


class TestIdleState:
    def test_starts_idle(self):
        state = TtsStreamState()
        assert state.phase is TtsStreamPhase.IDLE
        assert state.is_streaming_request is False

    def test_add_chunk_while_idle_is_rejected(self):
        state = TtsStreamState()
        assert state.add_chunk("Hallo") is False
        assert state.phase is TtsStreamPhase.IDLE

    def test_begin_synthesis_while_idle_returns_none(self):
        state = TtsStreamState()
        assert state.begin_synthesis() is None
        assert state.begin_synthesis(full_text="Hallo") is None


class TestStartCollecting:
    def test_start_transitions_to_collecting(self):
        state = TtsStreamState()
        state.start(voice_name="juergen")
        assert state.phase is TtsStreamPhase.COLLECTING
        assert state.voice_name == "juergen"
        assert state.is_streaming_request is True

    def test_start_with_no_voice(self):
        state = TtsStreamState()
        state.start(voice_name=None)
        assert state.voice_name is None

    def test_start_resets_any_previous_unfinished_request(self):
        """A second synthesize-start (client protocol violation, or a
        genuinely new request after a disconnect) always starts clean."""
        state = TtsStreamState()
        state.start(voice_name="a")
        state.add_chunk("stale text")
        state.start(voice_name="b")
        assert state.phase is TtsStreamPhase.COLLECTING
        assert state.voice_name == "b"
        assert state.collected_text == ""


class TestAddChunk:
    def test_single_chunk_accumulates(self):
        state = TtsStreamState()
        state.start(voice_name=None)
        assert state.add_chunk("Das Küchenfenster ist offen.") is True
        assert state.collected_text == "Das Küchenfenster ist offen."

    def test_multiple_chunks_concatenate_in_order(self):
        state = TtsStreamState()
        state.start(voice_name=None)
        state.add_chunk("Das Küchenfenster ")
        state.add_chunk("und das Schlafzimmerfenster ")
        state.add_chunk("sind geöffnet.")
        assert (
            state.collected_text == "Das Küchenfenster und das Schlafzimmerfenster sind geöffnet."
        )

    def test_add_chunk_after_synthesis_already_began_is_rejected(self):
        state = TtsStreamState()
        state.start(voice_name=None)
        state.add_chunk("Hallo")
        state.begin_synthesis(full_text="Hallo")
        assert state.add_chunk("more") is False


class TestBeginSynthesis:
    def test_backwards_compatible_full_text_wins_over_buffer(self):
        """Real Home Assistant always sends an identical full text, but
        the compat event's own text is authoritative regardless."""
        state = TtsStreamState()
        state.start(voice_name=None)
        state.add_chunk("Hallo")
        text = state.begin_synthesis(full_text="Hallo Welt")
        assert text == "Hallo Welt"
        assert state.phase is TtsStreamPhase.SYNTHESIZING

    def test_falls_back_to_collected_text_when_no_full_text_given(self):
        """A purely spec-following client that never sends the
        backwards-compatible `synthesize` event -- synthesize-stop is the
        only trigger, using whatever chunks were collected."""
        state = TtsStreamState()
        state.start(voice_name=None)
        state.add_chunk("Teil eins. ")
        state.add_chunk("Teil zwei.")
        text = state.begin_synthesis()
        assert text == "Teil eins. Teil zwei."

    def test_second_call_returns_none_never_double_triggers(self):
        """This is the core guarantee: the backwards-compatible whole-
        message event must never cause a second synthesis."""
        state = TtsStreamState()
        state.start(voice_name=None)
        state.add_chunk("Hallo Welt")
        first = state.begin_synthesis(full_text="Hallo Welt")
        second = state.begin_synthesis(full_text="Hallo Welt")
        assert first == "Hallo Welt"
        assert second is None

    def test_synthesize_stop_after_compat_synthesize_is_a_no_op(self):
        """Models the real Home Assistant order: synthesize-chunk,
        synthesize (compat, triggers synthesis), synthesize-stop (must be
        a no-op since synthesis already started)."""
        state = TtsStreamState()
        state.start(voice_name=None)
        state.add_chunk("Hallo Welt")
        triggered = state.begin_synthesis(full_text="Hallo Welt")
        assert triggered == "Hallo Welt"
        # synthesize-stop arrives next; begin_synthesis() with no override
        # (as the handler calls it for synthesize-stop) must not re-trigger.
        again = state.begin_synthesis()
        assert again is None


class TestFinishCancelReset:
    def test_finish_sets_finished_phase(self):
        state = TtsStreamState()
        state.start(voice_name=None)
        state.begin_synthesis(full_text="Hallo")
        state.finish()
        assert state.phase is TtsStreamPhase.FINISHED
        assert state.is_streaming_request is False

    def test_cancel_sets_cancelled_phase(self):
        state = TtsStreamState()
        state.start(voice_name=None)
        state.cancel()
        assert state.phase is TtsStreamPhase.CANCELLED
        assert state.is_streaming_request is False

    def test_reset_returns_to_idle_and_clears_state(self):
        state = TtsStreamState()
        state.start(voice_name="juergen")
        state.add_chunk("Hallo")
        state.reset()
        assert state.phase is TtsStreamPhase.IDLE
        assert state.voice_name is None
        assert state.collected_text == ""

    def test_start_after_finish_begins_a_fresh_request(self):
        """Two consecutive streaming requests on the same connection."""
        state = TtsStreamState()
        state.start(voice_name=None)
        state.add_chunk("Erste Anfrage.")
        state.begin_synthesis(full_text="Erste Anfrage.")
        state.finish()

        state.start(voice_name=None)
        assert state.phase is TtsStreamPhase.COLLECTING
        assert state.collected_text == ""
        state.add_chunk("Zweite Anfrage.")
        text = state.begin_synthesis(full_text="Zweite Anfrage.")
        assert text == "Zweite Anfrage."
