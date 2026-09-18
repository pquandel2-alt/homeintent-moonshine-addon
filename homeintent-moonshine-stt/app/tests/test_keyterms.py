"""Tests for extra_keyterms parsing, keyterm list merging, and the safe
Moonshine keyterm application layer."""

from moonshine_voice import MoonshineError

from app.keyterms import (
    REASON_CONTROL_CHARACTER,
    REASON_DELIMITER,
    REASON_EMPTY,
    REASON_TOKENIZER_REJECTED,
    STATUS_NORMALIZED,
    STATUS_SPEECH_FALLBACK,
    STATUS_UNCHANGED,
    _basic_validate,
    _generate_speech_fallback,
    apply_safe_keyterms,
    merge_keyterms,
    normalize_keyterm,
    parse_extra_keyterms,
)


class TestParseExtraKeyterms:
    def test_empty_string(self):
        assert parse_extra_keyterms("") == []

    def test_single_term(self):
        assert parse_extra_keyterms("Wohnzimmer") == ["Wohnzimmer"]

    def test_multiple_terms(self):
        assert parse_extra_keyterms("Wohnzimmer,Küche,Schlafzimmer") == [
            "Wohnzimmer",
            "Küche",
            "Schlafzimmer",
        ]

    def test_trims_whitespace_around_terms(self):
        assert parse_extra_keyterms("  Wohnzimmer ,  Küche  ") == ["Wohnzimmer", "Küche"]

    def test_drops_empty_entries_from_trailing_and_double_commas(self):
        assert parse_extra_keyterms("Wohnzimmer,,Küche,") == ["Wohnzimmer", "Küche"]

    def test_dedupes_preserving_first_seen_order(self):
        assert parse_extra_keyterms("Küche,Wohnzimmer,Küche") == ["Küche", "Wohnzimmer"]

    def test_preserves_umlauts_and_hyphens(self):
        assert parse_extra_keyterms("Büro-Schreibtischlampe,Gästezimmer") == [
            "Büro-Schreibtischlampe",
            "Gästezimmer",
        ]

    def test_only_whitespace_and_commas(self):
        assert parse_extra_keyterms(" , , ,") == []


class TestMergeKeyterms:
    def test_merges_two_lists_preserving_order(self):
        assert merge_keyterms(["a", "b"], ["c", "d"]) == ["a", "b", "c", "d"]

    def test_dedupes_across_lists_keeping_first_occurrence(self):
        assert merge_keyterms(["Küche", "Bad"], ["Bad", "Wohnzimmer"]) == [
            "Küche",
            "Bad",
            "Wohnzimmer",
        ]

    def test_empty_lists(self):
        assert merge_keyterms([], []) == []

    def test_single_list(self):
        assert merge_keyterms(["a", "b", "a"]) == ["a", "b"]

    def test_no_lists(self):
        assert merge_keyterms() == []


class _FakeMoonshineTranscriber:
    """Stands in for a real moonshine_voice.Transcriber: raises
    MoonshineError, exactly like the real native library does, if any term
    in the attempted list is in ``incompatible_terms`` -- one call for the
    whole list, matching the real set_keyterms() semantics (see
    app/keyterms.py's module docstring)."""

    def __init__(
        self, incompatible_terms: set[str] | None = None, always_raise: Exception | None = None
    ) -> None:
        self.incompatible_terms = incompatible_terms or set()
        self.always_raise = always_raise
        self.calls: list[list[str]] = []

    def set_keyterms(self, terms: list[str]) -> None:
        self.calls.append(list(terms))
        if self.always_raise is not None:
            raise self.always_raise
        for term in terms:
            if term in self.incompatible_terms:
                raise MoonshineError(f"Failed to set key terms: No match found for term {term!r}")


class TestApplySafeKeyterms:
    """Regression coverage for the real production incident: an
    HA-derived keyterm ("/Büro") that the loaded Moonshine model's
    tokenizer could not represent raised MoonshineError out of
    set_keyterms(), uncaught, crashing the whole add-on (exit code 1,
    restart loop). apply_safe_keyterms() is the single, reusable path that
    must make this impossible."""

    def test_fall_a_exact_production_incident_recovers_via_speech_fallback(self):
        """Fall A: the exact reported case -- '/Büro' among otherwise
        valid HA-derived terms must not crash anything. Since only the
        original, slash-containing form is rejected here (matching a model
        that tokenizes the separator-free name fine), the speech-fallback
        retry recovers it as 'Büro' instead of discarding it outright."""
        transcriber = _FakeMoonshineTranscriber(incompatible_terms={"/Büro"})
        candidates = ["Wohnzimmer", "/Büro", "Küche", "Garage"]

        result = apply_safe_keyterms(transcriber, candidates)

        assert result.apply_succeeded is True
        assert set(result.accepted) == {"Wohnzimmer", "Küche", "Garage", "Büro"}
        assert result.rejected == []
        fallback_entries = [a for a in result.applied if a.status == STATUS_SPEECH_FALLBACK]
        assert len(fallback_entries) == 1
        assert fallback_entries[0].original == "/Büro"
        assert fallback_entries[0].final == "Büro"

    def test_fall_a_variant_rejected_when_speech_fallback_also_fails(self):
        """When even the speech-fallback variant is genuinely incompatible
        with the loaded model, the term is skipped -- not force-applied."""
        transcriber = _FakeMoonshineTranscriber(incompatible_terms={"/Büro", "Büro"})
        candidates = ["Wohnzimmer", "/Büro", "Küche", "Garage"]

        result = apply_safe_keyterms(transcriber, candidates)

        assert result.apply_succeeded is True
        assert set(result.accepted) == {"Wohnzimmer", "Küche", "Garage"}
        assert [r.term for r in result.rejected] == ["/Büro"]
        assert result.rejected[0].reason == REASON_TOKENIZER_REJECTED

    def test_fall_b_german_characters_survive_basic_validation_untransformed(self):
        """Fall B: our own pre-validation must never mangle real German
        orthography -- if the (fake) model accepts them, they must arrive
        at set_keyterms() completely unchanged."""
        transcriber = _FakeMoonshineTranscriber()
        candidates = ["Büro", "Küche", "Außenlicht", "Gäste-WC", "Jürgens Lampe", "Rollladen Süd"]

        result = apply_safe_keyterms(transcriber, candidates)

        assert result.rejected == []
        assert result.accepted == candidates
        # The real transcriber must have actually received the untouched
        # German terms at some point, not a transliterated/stripped version.
        assert candidates in transcriber.calls

    def test_fall_c_mixed_valid_and_invalid(self):
        transcriber = _FakeMoonshineTranscriber(incompatible_terms={"/Büro", "Büro"})
        candidates = ["Wohnzimmer", "/Büro", "Küche", "Garage"]

        result = apply_safe_keyterms(transcriber, candidates)

        assert set(result.accepted) == {"Wohnzimmer", "Küche", "Garage"}
        # Final authoritative call carries exactly the accepted set.
        assert transcriber.calls[-1] == result.accepted

    def test_fall_d_all_invalid_still_starts_without_biasing(self):
        """Even the speech-fallback variants ("Büro", "Foo") are rejected
        here, so nothing survives -- the service must still start cleanly
        with biasing turned off rather than being force-fed a bad list."""
        transcriber = _FakeMoonshineTranscriber(
            incompatible_terms={"/Büro", "\\Foo", "Büro", "Foo"}
        )
        candidates = ["/Büro", "\\Foo"]

        result = apply_safe_keyterms(transcriber, candidates)

        assert result.apply_succeeded is True
        assert result.accepted == []
        assert len(result.rejected) == 2
        # The final explicit call turns biasing off cleanly.
        assert transcriber.calls[-1] == []

    def test_empty_candidate_list_applies_cleanly(self):
        transcriber = _FakeMoonshineTranscriber()
        result = apply_safe_keyterms(transcriber, [])
        assert result.accepted == []
        assert result.rejected == []
        assert result.apply_succeeded is True

    def test_fall_i_unexpected_native_error_falls_back_to_previous_terms(self):
        """Fall I: an error that cannot be attributed to a specific term
        (e.g. the transcriber itself is in a bad state) must not be
        misreported as a per-term rejection -- and must restore a
        known-good fallback rather than leaving biasing in a broken state.

        Validation succeeds (call 1: the whole candidate list is fine), but
        the final authoritative apply itself (call 2, identical content)
        fails structurally; the subsequent restore call (call 3, the
        caller-supplied fallback) succeeds.
        """

        class _FailOnSecondCall:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def set_keyterms(self, terms: list[str]) -> None:
                self.calls.append(list(terms))
                if len(self.calls) == 2:
                    raise RuntimeError("native crash")

        transcriber = _FailOnSecondCall()

        result = apply_safe_keyterms(
            transcriber, ["Wohnzimmer", "Küche"], fallback_terms=["Wohnzimmer"]
        )

        assert result.apply_succeeded is False
        assert result.accepted == ["Wohnzimmer"]
        assert transcriber.calls[-1] == ["Wohnzimmer"]
        assert len(transcriber.calls) == 3

    def test_unexpected_error_without_fallback_disables_biasing(self):
        transcriber = _FakeMoonshineTranscriber(always_raise=RuntimeError("native crash"))

        result = apply_safe_keyterms(transcriber, ["Wohnzimmer"], fallback_terms=None)

        assert result.apply_succeeded is False
        assert result.accepted == []
        assert transcriber.calls[-1] == []

    def test_never_raises_even_if_restore_also_fails(self):
        """If even the fallback restore call fails, apply_safe_keyterms()
        must still not raise -- the caller (startup or refresh loop) must
        keep running regardless."""

        class _AlwaysBroken:
            def set_keyterms(self, terms: list[str]) -> None:
                raise RuntimeError("everything is broken")

        result = apply_safe_keyterms(_AlwaysBroken(), ["Wohnzimmer"], fallback_terms=["Küche"])
        assert result.apply_succeeded is False

    def test_duplicate_terms_are_deduped(self):
        transcriber = _FakeMoonshineTranscriber()
        result = apply_safe_keyterms(transcriber, ["Küche", "Küche", "Wohnzimmer"])
        assert result.accepted == ["Küche", "Wohnzimmer"]

    def test_whitespace_only_and_empty_candidates_rejected_without_a_native_call(self):
        transcriber = _FakeMoonshineTranscriber()
        result = apply_safe_keyterms(transcriber, ["Wohnzimmer", "   ", "", "Küche"])
        assert result.accepted == ["Wohnzimmer", "Küche"]
        assert {r.term for r in result.rejected} == {"   ", ""}
        assert all(r.reason == REASON_EMPTY for r in result.rejected)

    def test_bisection_isolates_multiple_bad_terms_spread_across_the_list(self):
        """Several unrelated incompatible terms in a longer list must all
        be found and skipped, not just the first one encountered. Their
        speech-fallback variants ("Büro", "Keller") are also marked
        incompatible here so this test genuinely exercises full rejection,
        not fallback recovery (see the fallback-specific tests for that)."""
        transcriber = _FakeMoonshineTranscriber(
            incompatible_terms={"/Büro", "\\Keller", "Büro", "Keller"}
        )
        candidates = ["Wohnzimmer", "/Büro", "Küche", "Garage", "\\Keller", "Bad", "Flur"]

        result = apply_safe_keyterms(transcriber, candidates)

        assert set(result.accepted) == {"Wohnzimmer", "Küche", "Garage", "Bad", "Flur"}
        assert {r.term for r in result.rejected} == {"/Büro", "\\Keller"}

    def test_separator_speech_fallback_recovers_slash_term(self):
        """Fall: 'Treppe/Büro' is rejected as-is by the loaded model but the
        speech-safe fallback 'Treppe Büro' is accepted -- the composite
        term survives instead of being dropped entirely."""
        transcriber = _FakeMoonshineTranscriber(incompatible_terms={"Treppe/Büro"})

        result = apply_safe_keyterms(transcriber, ["Treppe/Büro"])

        assert result.accepted == ["Treppe Büro"]
        assert result.rejected == []
        assert result.applied[0].status == STATUS_SPEECH_FALLBACK
        assert result.applied[0].original == "Treppe/Büro"
        assert result.applied[0].final == "Treppe Büro"

    def test_original_kept_when_model_accepts_it_despite_separator(self):
        """If the loaded model actually accepts a separator character, the
        original form is kept unchanged -- '/' is never stripped
        pre-emptively before the real model has had a chance to accept it."""
        transcriber = _FakeMoonshineTranscriber()  # nothing is incompatible

        result = apply_safe_keyterms(transcriber, ["Treppe/Büro"])

        assert result.accepted == ["Treppe/Büro"]
        assert result.applied[0].status == STATUS_UNCHANGED

    def test_soft_hyphen_is_normalized_before_any_native_call_and_deduped(self):
        """Fall (soft hyphen): 'Wasch\\xadmaschine' is fixed at stage 1
        (lossless normalization), so it never even reaches the tokenizer as
        the broken form -- and duplicates with a clean 'Waschmaschine'
        candidate collapse into a single keyterm."""
        transcriber = _FakeMoonshineTranscriber()
        candidates = ["Wasch­maschine", "Waschmaschine"]

        result = apply_safe_keyterms(transcriber, candidates)

        assert result.accepted == ["Waschmaschine"]
        assert result.rejected == []
        # The native call never saw the soft hyphen -- normalization
        # happens purely in Python, before any set_keyterms() call.
        assert all("­" not in term for call in transcriber.calls for term in call)
        normalized_entries = [a for a in result.applied if a.status == STATUS_NORMALIZED]
        assert len(normalized_entries) == 1
        assert normalized_entries[0].original == "Wasch­maschine"
        assert normalized_entries[0].final == "Waschmaschine"

    def test_emoji_with_variation_selector_recovers_via_speech_fallback(self):
        """Fall: 'Familie ⚠️' (WARNING SIGN + VARIATION
        SELECTOR-16) is rejected as-is, but the speech-fallback 'Familie'
        (emoji and variation selector both stripped) is accepted."""
        transcriber = _FakeMoonshineTranscriber(incompatible_terms={"Familie ⚠️"})

        result = apply_safe_keyterms(transcriber, ["Familie ⚠️"])

        assert result.accepted == ["Familie"]
        assert result.applied[0].status == STATUS_SPEECH_FALLBACK
        assert result.applied[0].final == "Familie"

    def test_emoji_only_term_with_nothing_left_is_rejected_not_sent_empty(self):
        """A term that is nothing but an emoji collapses to an empty speech
        fallback -- it must be rejected, never sent to the model as ''."""
        transcriber = _FakeMoonshineTranscriber(incompatible_terms={"⚠️"})

        result = apply_safe_keyterms(transcriber, ["⚠️"])

        assert result.accepted == []
        assert [r.term for r in result.rejected] == ["⚠️"]
        # No call was ever made with an empty string.
        assert all("" not in call for call in transcriber.calls)


class TestBasicValidation:
    def test_empty_string_rejected(self):
        assert _basic_validate("") == REASON_EMPTY

    def test_whitespace_only_rejected_after_normalization(self):
        """_basic_validate() itself does not strip -- normalize_keyterm()
        does that in the apply_safe_keyterms() pipeline before validation
        runs, so a whitespace-only term is empty by the time it gets here."""
        assert _basic_validate(normalize_keyterm("   ")) == REASON_EMPTY

    def test_control_character_rejected(self):
        assert _basic_validate("Büro\x00Licht") == REASON_CONTROL_CHARACTER

    def test_newline_rejected(self):
        assert _basic_validate("Büro\nLicht") == REASON_CONTROL_CHARACTER

    def test_tab_rejected(self):
        assert _basic_validate("Büro\tLicht") == REASON_CONTROL_CHARACTER

    def test_comma_rejected_as_delimiter(self):
        assert _basic_validate("Büro,Licht") == REASON_DELIMITER

    def test_slash_is_not_rejected_by_basic_validation(self):
        """'/' is not universally invalid across Moonshine models -- only
        the real, loaded model's tokenizer decides that (see module
        docstring); basic validation must let it through."""
        assert _basic_validate("/Büro") is None

    def test_normal_german_terms_pass(self):
        for term in ["Büro", "Küche", "Außenlicht", "Gäste-WC", "Rollladen Süd", "Wohnzimmer TV"]:
            assert _basic_validate(term) is None


class TestNormalizeKeyterm:
    def test_strips_surrounding_whitespace(self):
        assert normalize_keyterm("  Büro  ") == "Büro"

    def test_nfc_normalizes_combining_diaeresis(self):
        decomposed = "üro"  # "u" + combining diaeresis + "ro" ~= "üro"
        assert normalize_keyterm(decomposed) == "üro"

    def test_does_not_transliterate_umlauts(self):
        assert normalize_keyterm("Büro") == "Büro"
        assert normalize_keyterm("Büro") != "Buro"


class TestNormalizeKeytermFormatCharacters:
    def test_soft_hyphen_removed_real_production_case(self):
        """Real production incident: the entity name contained U+00AD SOFT
        HYPHEN between 'h' and 'm', invisible in normal display. Built with
        the actual codepoint (\u00ad), not a visually copy-pasted string,
        so the test is unambiguous about what character is present."""
        assert normalize_keyterm("Wasch\u00admaschine") == "Waschmaschine"

    def test_zero_width_space_removed(self):
        assert normalize_keyterm("Wohn\u200bzimmer") == "Wohnzimmer"

    def test_zero_width_non_joiner_and_joiner_removed(self):
        assert normalize_keyterm("Test\u200c\u200dWort") == "TestWort"

    def test_word_joiner_removed(self):
        assert normalize_keyterm("Test\u2060Wort") == "TestWort"

    def test_byte_order_mark_removed(self):
        assert normalize_keyterm("K\ufeffuche".replace("uche", "\u00fcche")) == "K\u00fcche"

    def test_soft_hyphen_removal_does_not_merge_separate_words(self):
        """Only the invisible format character is removed -- an actual
        space between two words is never touched."""
        assert normalize_keyterm("Wohn\u00ad zimmer") == "Wohn zimmer"

    def test_internal_whitespace_is_collapsed(self):
        assert normalize_keyterm("Rollladen   S\u00fcd") == "Rollladen S\u00fcd"

    def test_normal_german_terms_are_never_touched(self):
        for term in [
            "B\u00fcro",
            "K\u00fcche",
            "Au\u00dfenlicht",
            "G\u00e4ste-WC",
            "J\u00fcrgens Lampe",
            "Temperatur 2",
            "Rollladen S\u00fcd",
        ]:
            assert normalize_keyterm(term) == term


class TestGenerateSpeechFallback:
    def test_slash_becomes_space(self):
        assert _generate_speech_fallback("Treppe/B\u00fcro") == "Treppe B\u00fcro"

    def test_backslash_and_pipe_become_space(self):
        assert _generate_speech_fallback("A\\B") == "A B"
        assert _generate_speech_fallback("A|B") == "A B"

    def test_emoji_stripped(self):
        assert _generate_speech_fallback("Garage \U0001f697") == "Garage"

    def test_emoji_with_variation_selector_fully_stripped(self):
        """The variation selector (U+FE0F) must not survive on its own once
        the base emoji character is removed."""
        result = _generate_speech_fallback("Familie \u26a0\ufe0f")
        assert result == "Familie"
        assert "\ufe0f" not in result

    def test_emoji_only_input_becomes_empty(self):
        assert _generate_speech_fallback("\u26a0\ufe0f") == ""

    def test_german_characters_never_touched(self):
        for term in [
            "B\u00fcro",
            "K\u00fcche",
            "Au\u00dfenlicht",
            "G\u00e4ste-WC",
            "J\u00fcrgens Lampe",
        ]:
            assert _generate_speech_fallback(term) == term

    def test_digits_and_hyphen_preserved(self):
        assert _generate_speech_fallback("Temperatur 2") == "Temperatur 2"
        assert _generate_speech_fallback("G\u00e4ste-WC") == "G\u00e4ste-WC"
