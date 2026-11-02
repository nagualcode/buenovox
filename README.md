# BuenoVOX

**Natural vocal pitch correction (auto + MIDI), written in Rust with the
[truce](https://github.com/truce-audio/truce) framework.** AU, VST3 and CLAP for macOS
(Apple Silicon).

A research project for personal, non-commercial use and the study of the craft. The full plan
(state of the art, patents, roadmap) lives in [PLANO.md](PLANO.md) (Portuguese).

![Interface](docs/gui.png)

> Listeners are moved by the voice, not by the tuning. Getting the tuning right only removes
> the distraction.

BuenoVOX **corrects only the perceived center of each note and gives everything else back
untouched**: vibrato, attack, drift, micro-fluctuations and the singer's cycle-to-cycle jitter.

---

## Installation (from source)

```bash
cargo install cargo-truce                 # once
cargo truce install --au2 --vst3 --clap   # build, sign and install
```

Restart your DAW. In Live: *Plug-Ins → Audio Units → Jardim Sonoro → BuenoVOX*.

## Usage

1. Put **BuenoVOX** on the vocal track.
2. Pick a **style**:
   * **NATURAL** (default): moves each whole note to its target as a block. Vibrato, drift
     and attack stay the singer's own. Between legato notes the offset migrates in
     proportion to the singer's own glide.
   * **CLASSIC**: traditional correction, with *Speed* (2nd-order retune), *Flex* (only
     corrects near the note) and *Humanize* (longer notes are left looser).
   * **HARD**: instant retune, the effect. The note is chosen from the note's center, not
     from the instantaneous f0, so an out-of-tune singer with vibrato does not warble between
     two notes.
3. Pick the **target source**:
   * **SCALE**: key and scale (12 scales).
   * **MIDI TARGET**: each held MIDI note is the target. The DAW's piano roll becomes the
     graphical editor: notes are the targets and **pitch bend** draws the fine curve (range
     set by *Bend*). With no notes held, the key/scale applies. In Live, on a MIDI track,
     set *MIDI To* to the vocal track and then *BuenoVOX*.
   * **MIDI SCALE**: the notes you play define the allowed notes, latched until the next
     chord.

   The singer may anticipate or lag a MIDI note change by up to 60 ms.
4. **Correction:**
   * **Amount**: how much of the error to correct. 0% is transparent.
   * **Speed**: in Natural, smoothing of the note offset; in Classic, the retune time.
   * **Tighten** (Natural): how much to flatten slow drift inside a note.
   * **Scoop** (Natural): shortens scoops and falls at onsets after silence.
5. **Expression:**
   * **Vibrato**: 0–200% of the singer's vibrato.
   * **Create Vib**: adds vibrato to long straight notes, fading in slowly.
   * **Throat**: longer throat (darker voice) or shorter (brighter).
6. **Tuning:**
   * **Pure Tuning**: just intonation relative to the lowest held MIDI note (or the tonic).
   * **A4**: reference from 415 to 466 Hz.
7. **Latency**, reported to the host (Live compensates on playback):
   * **STUDIO** (~270 ms): the brain sees ~190 ms ahead, almost the whole note, before
     correcting it. For mixing.
   * **TRACK** (~95 ms): for recording with monitoring.
   * **LIVE** (18 ms at any sample rate): for singing on stage. The audio does not wait for
     the pitch detector:
     * epoch marks are predicted from the real spacing of the previous cycle and confirmed by
       correlation, which needs only ~0.7 period of signal ahead;
     * grains are capped at 5 ms half-windows, so low voices get grains shorter than a period;
     * note decisions arrive ~30 ms after the sound, so the first ~30 ms of a note after
       silence pass uncorrected;
     * Hard divides by each cycle's real period, so its staircase stays sharp.

   A mode change takes effect when playback restarts or the plugin reloads.

---

## How it works

```
input ─► YIN (f0 every 4 ms) ─► epoch marks (one per glottal cycle)
                     │
                     ▼
               BRAIN (4 ms frames, with lookahead)
   segmentation by movement ─► local center (360 ms Hann, cancels vibrato)
   ─► note center (perceived pitch) ─► target (MIDI > scale) ─► policy
                     │ shift in semitones
                     ▼
   ratio PSOLA: synthesis period = real spacing between marks / ratio
```

* **Detector:** the YIN tracker inherited from [Heaven](https://github.com/nagualcode/heaven),
  with continuity-based candidates, octave tests, a register gate and a 3-frame median.
* **Segmentation by movement, not by rounding.** In a pitch corrector the singer is off center
  by definition, and with vibrato the peaks would cross the semitone boundary every cycle. A
  new note starts:
  * after silence;
  * on a leap (the voice is about to land far from the anchor);
  * or on a small step, detected by a *short center*: a 180 ms average, about one vibrato
    cycle, that cancels the vibrato and rises exactly at the step. So a legato half step with
    vibrato is still detected.
* **Perceived pitch:** the average of the local center over stable frames, with the attack
  (scoop, overshoot) weighted down. This is the *stationary pitch* of BERT-APC.
* **Natural:** `y = p + amount·(T − C) − tighten·drift + (vibrato − 1)·vib + created vibrato`.
  With the defaults this is a uniform shift of the whole note.
* **Ratio PSOLA:** grains are re-placed at the real spacing between analysis marks divided by
  the correction ratio. The singer's natural jitter reaches the output, and without correction
  the synthesis marks stay locked to the analysis marks: the output is the input itself, cycle
  by cycle.

## The ruler

Whoever writes the DSP cannot listen, so everything is measured against a **synthetic singer
with known f0** (`src/dsp/testsig.rs`). It is an additive source-filter synthesis with the
formants of the vowel "a" and a Saitou-model contour:
* 2nd-order overshoot at transitions;
* onset scoop;
* vibrato with delay and growth;
* fine fluctuation.

Each note can be sung out of tune, so the right answer is known.

```bash
cargo test --profile regua --no-default-features -- --nocapture
```

Current figures, on a 7-note phrase with the singer off by up to ±35 cents, 40-cent vibrato,
legato, a legato half step and rests:

| | Note-center error | Shift variation within the note* |
|---|---|---|
| Natural / Studio | ≤ 2.1 cents | ≤ 1.5 cents |
| Natural / Track | ≤ 4.0 cents | ≤ 3.6 cents |
| Natural / Live (18 ms) | ≤ 3.7 cents | ≤ 4.9 cents |
| Classic (120 ms retune) | ≤ 2.9 cents | ≤ 6.8 cents |
| Hard (Studio and Live) | ≤ 2.0 cents | output held still: ≤ 0.3 cent deviation |

\* 0 = vibrato, drift and fine fluctuation passed through untouched.

Other figures:
* **Detector:** **0.27 cent** median error, 0% gross errors.
* **PSOLA with no correction:** **55 dB** SNR against the dry signal in Studio and **62 dB** in
  Live (the waveform, not just the timbre). Across voiced/unvoiced boundaries the grain
  half-windows equal the distance to the neighboring grain on each side, so the windows sum to
  exactly 1.
* **Low voice in Live** (A2, 110 Hz, capped grains): corrected within 8 cents, energy within
  1.2 dB.
* **Bypass:** identical to the delayed dry signal.
* **Block size:** output is unchanged between 64 and 1024 samples.
* **MIDI Target:** 60 → 59.998 and 62 → 61.999.
* **Pitch bend:** +0.5 st → 60.499.
* **Robustness:** no data starvation at 44.1, 48 or 96 kHz in any mode; survives silence, noise,
  impulses, clipping and NaN.
* **Performance:** ~100× real time at 48 kHz and ~40× at 96 kHz in Studio.
* **Validators:** passes **auval** (AU v2 and v3) and **pluginval** (VST3).

## Development

```bash
cargo test --profile regua --no-default-features            # the ruler
cargo build --release --no-default-features --bin bvx_cli
target/release/bvx_cli gen phrase.wav detune=35 vibrato=40  # synthetic singer
target/release/bvx_cli vocal.wav out.wav key=C scale=major style=natural dump=curves.csv
target/release/bvx_cli vocal.wav out.wav source=miditarget midi="0-2:60;2-4:62"
cargo truce screenshot --out docs/gui.png
```

`bvx_cli` runs the plugin's own engine offline. With `dump=` it writes, every 4 ms frame, the
input pitch, target, output, shift, vibrato, note center, local center and anchor.

License: MIT or Apache-2.0.
