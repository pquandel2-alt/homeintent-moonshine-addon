# Test audio fixtures

## `schalte_lampe_wohnzimmer.wav`

Used by `app/tests/test_e2e_transcribe.py`'s real Moonshine STT end-to-end
test as a fixed, deterministic input -- replacing the previous approach of
synthesizing a fresh test utterance with a TTS voice on every CI run, which
made the STT test's outcome depend on two independently-behaving models at
once and was observed to be flaky in practice (a real GitHub Actions run
for the v0.2.2 tag mis-transcribed the previous on-the-fly-synthesized
audio; see ABSCHLUSSBERICHT_V0.2.3.md).

- **Text**: "Schalte die Lampe im Wohnzimmer ein."
- **Format**: WAV, 16000 Hz, mono, 16-bit PCM.
- **Duration**: a few seconds (see the file itself; regenerate and inspect
  with `scripts/generate_stt_fixture.py` if this needs re-verifying).
- **Generation method**: synthesized with the Piper `de_DE-thorsten-medium`
  voice via `moonshine_voice`'s own `TextToSpeech` helper (the same voice
  this repository's e2e tests already used for on-the-fly synthesis before
  this version), resampled to 16kHz with `scipy.signal.resample_poly`
  (proper anti-aliasing, unlike a naive linear interpolation) -- see
  `scripts/generate_stt_fixture.py` for the exact, reproducible generation
  code. The script synthesizes repeatedly and only keeps a candidate that a
  real Moonshine transcription round trip actually recognizes correctly,
  so the committed fixture is empirically verified, not merely generated.
  Regenerated only manually, via the `generate-stt-fixture`
  `workflow_dispatch`-only CI job (real network access to
  download.moonshine.ai is required; this sandbox's own network policy
  blocks it, so the fixture cannot be regenerated in a purely local
  session without that CI job).
- **Source / license**: the Piper `de_DE-thorsten-medium` model is trained
  on the [Thorsten-Voice](https://github.com/thorstenMueller/Thorsten-Voice)
  dataset by Thorsten Müller, whose project page states it is explicitly
  released to be usable "without any license struggling"; the trained
  Piper model itself (`Thorsten-Voice/Piper` on Hugging Face) is published
  under the **MIT license**. Both the dataset and the resulting model
  therefore permit generating and redistributing synthesized speech output
  such as this fixture. No third-party recording is used or committed --
  this file is wholly machine-generated from written German text.
- **Date generated**: 2026-09-18.

## Why "Lampe" and not "Licht"

The original test sentence was "Schalte das Licht im Wohnzimmer ein." --
also a natural smart-home command, but this Piper voice's rendering of
"Licht" turned out to be systematically misheard by Moonshine's
tiny-streaming-de model across several real CI runs (misheard as "jetzt",
then "Ditting", then "Setting" in separate runs -- never correctly, though
consistently within a single process). Retrying synthesis within one
process did not help since the wrong result was fully reproducible there;
only changing the word did. "Lampe" was verified to transcribe correctly
by the same real Moonshine model before being committed.

If this fixture is ever regenerated, verify with a fresh manual run of
`scripts/generate_stt_fixture.py` (or the CI job) that the resulting audio
still transcribes correctly with the current `tiny` Moonshine model before
committing it, and update this README's date. If a chosen sentence turns
out to be as unreliable as "Licht" was, prefer changing the sentence over
trying to force a marginal word through.
