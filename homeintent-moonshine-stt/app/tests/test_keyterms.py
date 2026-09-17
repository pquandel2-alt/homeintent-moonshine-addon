"""Tests for extra_keyterms parsing and keyterm list merging."""

from app.keyterms import merge_keyterms, parse_extra_keyterms


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
