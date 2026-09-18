"""Tests for the original German text normalizer (app/german_text_normalizer.py).

Every "before" value in these tests was verified to actually mispronounce
with kokoro-onnx==0.6.1 + phonemizer==3.4.0 + espeakng-loader==0.2.4's real
German espeak-ng phonemization before this module existed (see its module
docstring) -- these are not hypothetical problems.
"""

from app.german_text_normalizer import normalize_german_text


class TestTimes:
    def test_time_with_uhr(self):
        assert normalize_german_text("18:20 Uhr") == "18 Uhr 20"

    def test_time_on_the_hour_omits_minute_word(self):
        assert normalize_german_text("14:00 Uhr") == "14 Uhr"

    def test_time_without_uhr_suffix(self):
        assert normalize_german_text("9:05") == "9 Uhr 5"

    def test_time_in_sentence(self):
        assert (
            normalize_german_text("Die Waschmaschine ist um 18:20 Uhr fertig.")
            == "Die Waschmaschine ist um 18 Uhr 20 fertig."
        )


class TestTemperatures:
    def test_celsius_with_decimal(self):
        assert normalize_german_text("21,5 °C") == "21,5 Grad Celsius"

    def test_celsius_no_space_before_c(self):
        assert normalize_german_text("21,5°C") == "21,5 Grad Celsius"

    def test_fahrenheit(self):
        assert normalize_german_text("70 °F") == "70 Grad Fahrenheit"

    def test_bare_degree_symbol(self):
        assert normalize_german_text("180°") == "180 Grad"

    def test_temperature_in_sentence(self):
        assert (
            normalize_german_text("Im Wohnzimmer sind 21,5 °C.")
            == "Im Wohnzimmer sind 21,5 Grad Celsius."
        )


class TestUnits:
    def test_kilowatt_hours(self):
        assert normalize_german_text("2,5 kWh") == "2,5 Kilowattstunden"

    def test_grams(self):
        assert normalize_german_text("500 g") == "500 Gramm"

    def test_milligrams_not_confused_with_bare_meter(self):
        """Regression: bare "m" (Meter) is a prefix of "mg" -- must not
        win the match and leave a stray "g" behind."""
        assert normalize_german_text("5 mg") == "5 Milligramm"

    def test_milliliters_not_confused_with_bare_meter(self):
        assert normalize_german_text("5 ml") == "5 Milliliter"

    def test_kilograms(self):
        assert normalize_german_text("2 kg") == "2 Kilogramm"

    def test_liters(self):
        assert normalize_german_text("1 l") == "1 Liter"

    def test_kilometers(self):
        assert normalize_german_text("3 km") == "3 Kilometer"

    def test_kilometers_per_hour(self):
        assert normalize_german_text("50 km/h") == "50 Stundenkilometer"

    def test_meters(self):
        assert normalize_german_text("10 m") == "10 Meter"

    def test_centimeters(self):
        assert normalize_german_text("30 cm") == "30 Zentimeter"

    def test_percent(self):
        assert normalize_german_text("45%") == "45 Prozent"

    def test_milliamps(self):
        assert normalize_german_text("500 mAh") == "500 Milliamperestunden"

    def test_hours_bare_h(self):
        assert normalize_german_text("3 h") == "3 Stunden"


class TestCurrency:
    def test_eur_word(self):
        assert normalize_german_text("49,99 EUR") == "49,99 Euro"

    def test_euro_symbol(self):
        assert normalize_german_text("49,99 €") == "49,99 Euro"

    def test_euro_symbol_at_end_of_string(self):
        """Regression: a plain word-boundary check after a non-word
        symbol like "€" can silently fail to match at all when nothing
        (or punctuation) follows -- see module docstring."""
        assert normalize_german_text("Das kostet 10€") == "Das kostet 10 Euro"


class TestWordAbbreviations:
    def test_min_mid_sentence_no_period_inserted(self):
        assert (
            normalize_german_text("Das dauert 45 Min. später.") == "Das dauert 45 Minuten später."
        )

    def test_std_at_end_of_sentence_keeps_period(self):
        assert normalize_german_text("Das dauert 2 Std.") == "Das dauert 2 Stunden."

    def test_zzgl(self):
        assert normalize_german_text("zzgl. Versand") == "zuzüglich Versand"

    def test_ggf(self):
        assert normalize_german_text("ggf. später") == "gegebenenfalls später"

    def test_bzw(self):
        assert normalize_german_text("Küche bzw. Wohnzimmer") == "Küche beziehungsweise Wohnzimmer"


class TestGermanOrthographyPreserved:
    """Core correctness requirement (item 11 of the spec): umlauts and ß
    must never be transliterated or otherwise altered."""

    def test_words_with_umlauts_and_eszett_untouched(self):
        for word in [
            "Küche",
            "Büro",
            "Außenlicht",
            "Gäste-WC",
            "Rollladen",
            "Waschmaschine",
            "Jürgen",
        ]:
            assert normalize_german_text(word) == word

    def test_ordinary_sentence_untouched(self):
        text = "Das Küchenfenster und das Schlafzimmerfenster sind geöffnet."
        assert normalize_german_text(text) == text

    def test_empty_string(self):
        assert normalize_german_text("") == ""

    def test_no_transliteration_of_umlauts_even_near_numbers(self):
        assert normalize_german_text("Büro 21,5 Grad") == "Büro 21,5 Grad"


class TestIdempotence:
    def test_normalizing_twice_is_a_no_op(self):
        text = "Die Waschmaschine ist um 18:20 Uhr fertig, 2,5 kWh verbraucht."
        once = normalize_german_text(text)
        twice = normalize_german_text(once)
        assert once == twice


class TestCombinedRealWorldExamples:
    """Realistic HomeIntent-style sentences combining several patterns."""

    def test_window_status_with_temperature(self):
        text = "Im Wohnzimmer sind 21,5 Grad Celsius und das Fenster ist offen."
        assert normalize_german_text(text) == text  # already-normalized form is untouched

    def test_washing_machine_finish_time(self):
        assert (
            normalize_german_text("Die Waschmaschine ist um 18:20 Uhr fertig.")
            == "Die Waschmaschine ist um 18 Uhr 20 fertig."
        )

    def test_energy_usage_with_currency(self):
        assert (
            normalize_german_text("Heute wurden 2,5 kWh für 0,80 EUR verbraucht.")
            == "Heute wurden 2,5 Kilowattstunden für 0,80 Euro verbraucht."
        )
