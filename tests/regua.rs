//! A regua: o BuenoVOX medido contra um cantor sintetico de f0 conhecido.
//!
//! Quem escreve o DSP nao pode ouvir, entao tudo que deve soar bem precisa antes medir bem
//! (PLANO.md, secao 4). Cada teste imprime as medidas (`cargo test --release -- --nocapture`).

use buenovox::dsp::pitch::PitchTracker;
use buenovox::dsp::testsig::{render, Rendered, SingerSpec, SungNote};
use buenovox::dsp::{Engine, EngineParams, LatencyMode, Source, Style};

const SR: f32 = 48_000.0;

fn hz_to_midi(f: f32) -> f32 { 69.0 + 12.0 * (f / 440.0).log2() }

/// Passa `x` (mono) pelo motor e devolve a saida alinhada (latencia removida).
fn run(x: &[f32], sr: f32, mode: LatencyMode, p: &EngineParams, block: usize, midi: &dyn Fn(&mut Engine, usize, usize)) -> (Vec<f32>, u64) {
    let mut e = Engine::new(sr, mode);
    let lat = e.latency();
    let total = x.len() + lat;
    let mut out = vec![0.0f32; total];
    let mut scratch = vec![0.0f32; total];
    let mut pos = 0;
    while pos < total {
        let m = block.min(total - pos);
        e.set_params(p);
        midi(&mut e, pos, m);
        let mut inp = vec![0.0f32; m];
        for (k, v) in inp.iter_mut().enumerate() {
            *v = x.get(pos + k).copied().unwrap_or(0.0);
        }
        let (a, b) = (&mut out[pos..pos + m], &mut scratch[pos..pos + m]);
        e.process(&inp, &inp, a, b);
        pos += m;
    }
    (out[lat..].to_vec(), e.starved())
}

fn no_midi(_: &mut Engine, _: usize, _: usize) {}

/// Altura medida (MIDI) a cada 4 ms; NaN sem vozeamento.
fn measure(x: &[f32], sr: f32) -> Vec<f32> {
    let mut pt = PitchTracker::new(sr);
    // em blocos: o anel interno do rastreador nao guarda o sinal inteiro
    for chunk in x.chunks(512) {
        pt.push(chunk);
    }
    let hop = (sr * 0.004) as usize;
    (0..x.len() / hop)
        .map(|k| {
            let fr = pt.at((k * hop) as f64);
            if fr.voiced() { hz_to_midi(fr.f0) } else { f32::NAN }
        })
        .collect()
}

/// f0 verdadeiro (MIDI) na mesma grade de 4 ms.
fn truth(r: &Rendered, sr: f32) -> Vec<f32> {
    let hop = (sr * 0.004) as usize;
    (0..r.f0.len() / hop).map(|k| if r.f0[k * hop] > 0.0 { hz_to_midi(r.f0[k * hop]) } else { f32::NAN }).collect()
}

fn mean_std(v: &[f32]) -> (f32, f32) {
    let n = v.len().max(1) as f32;
    let m = v.iter().sum::<f32>() / n;
    let s = (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n).sqrt();
    (m, s)
}

/// Frase de teste: notas desafinadas (ate `detune` cents, para os dois lados), com vibrato,
/// legato, um meio tom ligado e silencios. Vizinhos de meio tom nunca erram um em direcao ao
/// outro a ponto de virar a mesma altura (isso seria ambiguo ate para o ouvido).
fn phrase(detune: f32, vib: f32) -> Rendered {
    let signs = [1.0, -0.85, 0.7, -1.0, 0.85, 0.55, -0.7];
    let notes = [60.0, 62.0, 64.0, 65.0, 67.0, 65.0, 64.0]
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let n = SungNote::new(m, signs[i] * detune, 0.85).vibrato(5.3 + 0.1 * i as f32, vib);
            if i == 2 || i == 4 { n.gap(0.2) } else { n }
        })
        .collect();
    render(&SingerSpec::new(SR, notes))
}

struct NoteReport {
    intended: f32,
    /// Erro do centro da nota em relacao ao alvo (cents).
    center_err: f32,
    /// Variacao do deslocamento aplicado dentro da nota (cents): 0 = expressao intacta.
    expr_err: f32,
    /// Variacao da saida em torno do proprio centro (cents).
    out_std: f32,
}

/// Mede cada nota no corpo (sem o ataque e o fim): centro, preservacao da expressao.
fn report(r: &Rendered, out: &[f32], sr: f32) -> Vec<NoteReport> {
    let hop = (sr * 0.004) as usize;
    let tin = truth(r, sr);
    let tout = measure(out, sr);
    r.spans
        .iter()
        .map(|sp| {
            let k0 = (sp.start + (0.15 * sr) as usize) / hop;
            let k1 = (sp.end - (0.1 * sr) as usize) / hop;
            let mut d = Vec::new();
            let mut o = Vec::new();
            for k in k0..k1.min(tout.len()) {
                if tout[k].is_finite() && tin[k].is_finite() {
                    d.push(tout[k] - tin[k]);
                    o.push(tout[k]);
                }
            }
            let (dm, ds) = mean_std(&d);
            let (_, os) = mean_std(&o);
            // a saida ideal e a entrada deslocada de (pretendida - cantada)
            NoteReport { intended: sp.intended, center_err: 100.0 * (dm - (sp.intended - sp.sung)), expr_err: 100.0 * ds, out_std: 100.0 * os }
        })
        .collect()
}

fn print(name: &str, rep: &[NoteReport]) {
    println!("--- {name}");
    for n in rep {
        println!("  nota {:>5.1}: centro {:>+6.2} c   expressao {:>5.2} c   desvio da saida {:>5.2} c", n.intended, n.center_err, n.expr_err, n.out_std);
    }
}

fn c_major() -> EngineParams {
    EngineParams { key: 0, scale: 1, ..Default::default() }
}

// ---------------------------------------------------------------------------------- detector

#[test]
fn detector_accuracy_on_synthetic_voice() {
    let r = phrase(0.0, 40.0);
    let est = measure(&r.audio, SR);
    let tru = truth(&r, SR);
    let mut errs: Vec<f32> = Vec::new();
    let mut gross = 0;
    let mut voiced = 0;
    for (k, (&e, &t)) in est.iter().zip(&tru).enumerate() {
        // so o corpo das notas (o vozeamento comeca/acaba com o envelope)
        let inside = r.spans.iter().any(|s| k * 192 > s.start + 2400 && k * 192 + 2400 < s.end);
        if !inside || !t.is_finite() {
            continue;
        }
        voiced += 1;
        if !e.is_finite() {
            gross += 1;
            continue;
        }
        let c = 100.0 * (e - t).abs();
        if c > 50.0 {
            gross += 1;
        } else {
            errs.push(c);
        }
    }
    errs.sort_by(f32::total_cmp);
    let med = errs[errs.len() / 2];
    let rate = gross as f32 / voiced as f32;
    println!("detector: erro mediano {med:.2} cents, erro grosseiro {:.2}% de {voiced} quadros", 100.0 * rate);
    assert!(med < 3.0, "erro mediano {med}");
    assert!(rate < 0.01, "erro grosseiro {rate}");
}

// ---------------------------------------------------------------------------------- correcao

#[test]
fn natural_studio_corrects_center_and_keeps_expression() {
    let r = phrase(35.0, 40.0);
    let (out, starved) = run(&r.audio, SR, LatencyMode::Studio, &c_major(), 512, &no_midi);
    assert_eq!(starved, 0);
    let rep = report(&r, &out, SR);
    print("Natural / Studio, cantor +-35 cents, vibrato 40 cents", &rep);
    for n in &rep {
        assert!(n.center_err.abs() < 5.0, "nota {} centro {}", n.intended, n.center_err);
        assert!(n.expr_err < 5.0, "nota {} expressao {}", n.intended, n.expr_err);
        // o vibrato continua la (RMS de 40 cents de pico ~ 28 cents, menos a entrada gradual)
        assert!(n.out_std > 12.0, "nota {} vibrato achatado: {}", n.intended, n.out_std);
    }
}

#[test]
fn natural_track_mode() {
    let r = phrase(35.0, 40.0);
    let (out, starved) = run(&r.audio, SR, LatencyMode::Track, &c_major(), 256, &no_midi);
    assert_eq!(starved, 0);
    let rep = report(&r, &out, SR);
    print("Natural / Track", &rep);
    for n in &rep {
        assert!(n.center_err.abs() < 8.0, "nota {} centro {}", n.intended, n.center_err);
        assert!(n.expr_err < 8.0, "nota {} expressao {}", n.intended, n.expr_err);
    }
}

#[test]
fn natural_live_mode() {
    let r = phrase(35.0, 40.0);
    let (out, starved) = run(&r.audio, SR, LatencyMode::Live, &c_major(), 128, &no_midi);
    assert_eq!(starved, 0);
    let rep = report(&r, &out, SR);
    print("Natural / Live", &rep);
    for n in &rep {
        assert!(n.center_err.abs() < 8.0, "nota {} centro {}", n.intended, n.center_err);
        assert!(n.expr_err < 8.0, "nota {} expressao {}", n.intended, n.expr_err);
    }
}

#[test]
fn hard_live_mode() {
    let r = phrase(35.0, 40.0);
    let mut p = c_major();
    p.brain.style = Style::Hard;
    let (out, _) = run(&r.audio, SR, LatencyMode::Live, &p, 128, &no_midi);
    let rep = report(&r, &out, SR);
    print("Hard / Live", &rep);
    for n in &rep {
        assert!(n.center_err.abs() < 6.0, "nota {} centro {}", n.intended, n.center_err);
        assert!(n.out_std < 6.0, "nota {} ainda oscila {}", n.intended, n.out_std);
    }
}

#[test]
fn live_latency_under_20ms() {
    for &sr in &[44_100.0f32, 48_000.0, 96_000.0] {
        let e = Engine::new(sr, LatencyMode::Live);
        let ms = e.latency() as f32 * 1000.0 / sr;
        println!("ao vivo a {sr} Hz: {ms:.1} ms");
        assert!(ms < 20.0);
    }
}

#[test]
fn live_identity_and_low_voices() {
    // sem correcao, o modo ao vivo continua transparente; e uma voz grave (A2, 110 Hz, grao
    // limitado a 5 ms) sai afinada e sem perder energia
    let r = phrase(0.0, 40.0);
    let mut p = c_major();
    p.brain.amount = 0.0;
    let (out, _) = run(&r.audio, SR, LatencyMode::Live, &p, 128, &no_midi);
    let (mut se, mut ss) = (0.0f64, 0.0f64);
    for (o, i) in out.iter().zip(&r.audio).skip(12000) {
        se += ((o - i) as f64).powi(2);
        ss += (*i as f64).powi(2);
    }
    let snr = 10.0 * (ss / se.max(1e-20)).log10();
    println!("ao vivo sem correcao: SNR {snr:.1} dB");
    assert!(snr > 30.0, "{snr}");

    let notes = vec![SungNote::new(45.0, 30.0, 1.2).vibrato(5.5, 30.0), SungNote::new(47.0, -30.0, 1.2)];
    let r = render(&SingerSpec::new(SR, notes));
    let (out, starved) = run(&r.audio, SR, LatencyMode::Live, &c_major(), 128, &no_midi);
    assert_eq!(starved, 0);
    let rep = report(&r, &out, SR);
    print("Natural / Live, voz grave", &rep);
    let e_in: f32 = r.audio.iter().map(|x| x * x).sum();
    let e_out: f32 = out.iter().map(|x| x * x).sum();
    let ratio_db = 10.0 * (e_out / e_in).log10();
    println!("energia saida/entrada: {ratio_db:+.2} dB");
    assert!(ratio_db.abs() < 1.5);
    for n in &rep {
        assert!(n.center_err.abs() < 8.0, "nota {} centro {}", n.intended, n.center_err);
    }
}

#[test]
fn hard_flattens_everything() {
    let r = phrase(35.0, 40.0);
    let mut p = c_major();
    p.brain.style = Style::Hard;
    let (out, _) = run(&r.audio, SR, LatencyMode::Studio, &p, 512, &no_midi);
    let rep = report(&r, &out, SR);
    print("Hard", &rep);
    for n in &rep {
        // a saida fica parada na nota: o centro do desvio vira o proprio alvo
        assert!(n.out_std < 6.0, "nota {} ainda oscila {}", n.intended, n.out_std);
    }
}

#[test]
fn classic_slow_retune_keeps_vibrato() {
    let r = phrase(35.0, 40.0);
    let mut p = c_major();
    p.brain.style = Style::Classic;
    p.brain.speed_ms = 120.0;
    p.brain.humanize = 0.0;
    let (out, _) = run(&r.audio, SR, LatencyMode::Studio, &p, 512, &no_midi);
    let rep = report(&r, &out, SR);
    print("Classic, retune 120 ms", &rep);
    for n in &rep {
        // retune lento e correcao parcial por definicao: o centro chega perto, o vibrato fica
        assert!(n.center_err.abs() < 12.0, "nota {} centro {}", n.intended, n.center_err);
        assert!(n.out_std > 8.0, "nota {} vibrato achatado {}", n.intended, n.out_std);
    }
}

#[test]
fn vibrato_knob_scales_vibrato() {
    let r = phrase(0.0, 40.0);
    let mut p = c_major();
    p.brain.vibrato = 0.0;
    let (flat, _) = run(&r.audio, SR, LatencyMode::Studio, &p, 512, &no_midi);
    p.brain.vibrato = 1.0;
    let (keep, _) = run(&r.audio, SR, LatencyMode::Studio, &p, 512, &no_midi);
    let a = report(&r, &flat, SR);
    let b = report(&r, &keep, SR);
    let ma = a.iter().map(|n| n.out_std).sum::<f32>() / a.len() as f32;
    let mb = b.iter().map(|n| n.out_std).sum::<f32>() / b.len() as f32;
    println!("vibrato 0%: {ma:.1} cents RMS, 100%: {mb:.1} cents RMS");
    assert!(ma < 0.4 * mb, "vibrato 0% deveria quase sumir: {ma} vs {mb}");
}

// ---------------------------------------------------------------------------------- MIDI

#[test]
fn midi_target_overrides_scale() {
    // canta 60.3 o tempo todo; o MIDI pede 62 na segunda metade
    let notes = vec![SungNote::new(60.0, 30.0, 2.0)];
    let r = render(&SingerSpec::new(SR, notes));
    let mut p = c_major();
    p.source = Source::MidiTarget;
    let half = r.spans[0].start + (1.0 * SR) as usize;
    let midi = move |e: &mut Engine, pos: usize, m: usize| {
        if pos == 0 {
            e.note_on(0, 60);
        }
        if (pos..pos + m).contains(&half) {
            e.note_off(half - pos, 60);
            e.note_on(half - pos, 62);
        }
    };
    let (out, _) = run(&r.audio, SR, LatencyMode::Studio, &p, 512, &midi);
    let tout = measure(&out, SR);
    let hop = (SR * 0.004) as usize;
    let seg = |a: usize, b: usize| mean_std(&tout[a / hop..b / hop].iter().copied().filter(|x| x.is_finite()).collect::<Vec<_>>()).0;
    let first = seg(r.spans[0].start + 12000, half - 9600);
    let second = seg(half + 14400, r.spans[0].end - 4800);
    println!("MIDI 60 -> {first:.3}, MIDI 62 -> {second:.3}");
    assert!((first - 60.0).abs() < 0.06);
    assert!((second - 62.0).abs() < 0.06);
}

#[test]
fn pitch_bend_moves_target() {
    let notes = vec![SungNote::new(60.0, 0.0, 1.5)];
    let r = render(&SingerSpec::new(SR, notes));
    let mut p = c_major();
    p.source = Source::MidiTarget;
    p.bend_range = 2.0;
    let midi = |e: &mut Engine, pos: usize, _m: usize| {
        if pos == 0 {
            e.note_on(0, 60);
            e.pitch_bend(0, 0.25);
        }
    };
    let (out, _) = run(&r.audio, SR, LatencyMode::Studio, &p, 512, &midi);
    let tout = measure(&out, SR);
    let hop = (SR * 0.004) as usize;
    let s = &r.spans[0];
    let (m, _) = mean_std(&tout[(s.start + 14400) / hop..(s.end - 4800) / hop].iter().copied().filter(|x| x.is_finite()).collect::<Vec<_>>());
    println!("bend +0.5 st -> {m:.3}");
    assert!((m - 60.5).abs() < 0.06);
}

// ---------------------------------------------------------------------------------- engenharia

#[test]
fn bypass_is_exact_delayed_dry() {
    let r = phrase(35.0, 40.0);
    let mut p = c_major();
    p.bypass = true;
    let (out, _) = run(&r.audio, SR, LatencyMode::Studio, &p, 512, &no_midi);
    // o cruzamento de 10 ms no inicio ja terminou antes do sinal (lead-in de 250 ms)
    let err = out.iter().zip(&r.audio).skip(4800).fold(0.0f32, |a, (o, i)| a.max((o - i).abs()));
    assert!(err < 1e-6, "bypass diferente do seco: {err}");
}

#[test]
fn psola_identity_when_no_correction() {
    let r = phrase(0.0, 40.0);
    let mut p = c_major();
    p.brain.amount = 0.0;
    let (out, _) = run(&r.audio, SR, LatencyMode::Studio, &p, 512, &no_midi);
    let (mut se, mut ss) = (0.0f64, 0.0f64);
    for (o, i) in out.iter().zip(&r.audio).skip(12000) {
        se += ((o - i) as f64).powi(2);
        ss += (*i as f64).powi(2);
    }
    let snr = 10.0 * (ss / se.max(1e-20)).log10();
    println!("PSOLA sem correcao: SNR em relacao ao seco {snr:.1} dB");
    assert!(snr > 40.0, "PSOLA sem correcao deveria ser transparente: {snr} dB");
}

#[test]
fn block_size_does_not_change_output() {
    let r = phrase(35.0, 40.0);
    let (a, _) = run(&r.audio, SR, LatencyMode::Studio, &c_major(), 64, &no_midi);
    let (b, _) = run(&r.audio, SR, LatencyMode::Studio, &c_major(), 1024, &no_midi);
    let err = a.iter().zip(&b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(err < 1e-5, "saida depende do tamanho do bloco: {err}");
}

#[test]
fn never_starves_at_common_rates() {
    for &sr in &[44_100.0f32, 48_000.0, 96_000.0] {
        let notes = vec![SungNote::new(45.0, 20.0, 0.8).vibrato(5.5, 50.0), SungNote::new(76.0, -20.0, 0.8)];
        let r = render(&SingerSpec::new(sr, notes));
        for mode in [LatencyMode::Live, LatencyMode::Track, LatencyMode::Studio] {
            let (_, starved) = run(&r.audio, sr, mode, &c_major(), 333, &no_midi);
            assert_eq!(starved, 0, "fome em {sr} Hz, {mode:?}");
        }
    }
}

#[test]
fn survives_hostile_input() {
    let n = 48_000;
    let mut rng = 12345u64;
    let mut noise = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0; n],
        (0..n).map(|_| noise()).collect(),
        (0..n).map(|i| if i % 4800 == 0 { 1.0 } else { 0.0 }).collect(),
        (0..n).map(|i| if (i / 100) % 2 == 0 { 4.0 } else { -4.0 }).collect(),
        (0..n).map(|i| if i % 7 == 0 { f32::NAN } else { (i as f32 * 0.03).sin() }).collect(),
    ];
    for x in &inputs {
        for style in [Style::Natural, Style::Classic, Style::Hard] {
            let mut p = c_major();
            p.brain.style = style;
            let (out, _) = run(x, SR, LatencyMode::Studio, &p, 512, &no_midi);
            assert!(out.iter().all(|v| v.is_finite() && v.abs() < 16.0));
        }
    }
}

// ---------------------------------------------------------------------------------- respiracao

/// Separa cada parte com as marcas e as receitas (alinhamento, ganhos) da mistura `mix`: com a
/// receita fixa a separacao e linear, e cada componente verdadeira pode ser medida sozinha.
fn split_with_marks(mix: &[f32], parts: &[&[f32]], past: f64, fut: f64, live: bool) -> Vec<(Vec<f32>, Vec<f32>)> {
    use buenovox::dsp::breath::BreathSplit;
    use buenovox::dsp::marks::MarkTracker;
    use buenovox::dsp::util::Ring;
    let mut pt = PitchTracker::new(SR);
    let mut marks = MarkTracker::new(SR);
    marks.live = live;
    let mut rings: Vec<Ring> = parts.iter().map(|_| Ring::new(1 << 17)).collect();
    let mut guide = Ring::new(1 << 17);
    let mut splits: Vec<BreathSplit> = parts.iter().map(|_| BreathSplit::new(SR, past, fut)).collect();
    let mut outs: Vec<(Vec<f32>, Vec<f32>)> = parts.iter().map(|_| (vec![0.0; mix.len()], vec![0.0; mix.len()])).collect();
    let mut n = 0usize;
    let mut taken = 0usize;
    for chunk in mix.chunks(256) {
        for (i, v) in chunk.iter().enumerate() {
            guide.set((n + i) as i64, *v);
        }
        for (ri, r) in rings.iter_mut().enumerate() {
            for i in 0..chunk.len() {
                r.set((n + i) as i64, parts[ri][n + i]);
            }
        }
        pt.push(chunk);
        n += chunk.len();
        marks.update(&pt, n as i64);
        for (i, sp) in splits.iter_mut().enumerate() {
            sp.advance_guided(&rings[i], &guide, &marks);
            if live {
                sp.ensure_guided(&rings[i], &guide, &marks, n as i64 - (0.012 * SR) as i64, n as i64);
            }
        }
        let done = splits.iter().map(|s| s.done()).min().unwrap().max(0) as usize;
        for t in taken..done {
            for (i, sp) in splits.iter().enumerate() {
                outs[i].0[t] = sp.harm.at(t as i64);
                outs[i].1[t] = sp.noise.at(t as i64);
            }
        }
        taken = done;
    }
    outs
}

fn energy_hp(x: &[f32], f: f64) -> f64 {
    let mut a = buenovox::dsp::util::Biquad::highpass(SR as f64, f, 0.707);
    let mut b = buenovox::dsp::util::Biquad::highpass(SR as f64, f, 0.707);
    x.iter().map(|&v| (b.process(a.process(v)) as f64).powi(2)).sum()
}

#[test]
fn breath_split_separates_air() {
    // voz muito soprosa (HNR ~6 dB), com vibrato: o ar deve sair da parte harmonica sem levar
    // harmonicos junto (o "fantasma" da nota antiga, que passaria sem correcao)
    let notes = vec![
        SungNote::new(57.0, 0.0, 1.0).vibrato(5.5, 40.0),
        SungNote::new(64.0, 0.0, 1.0).vibrato(5.5, 40.0).gap(0.2),
        SungNote::new(69.0, 0.0, 1.0).vibrato(5.5, 40.0),
        SungNote::new(72.0, 0.0, 1.0),
    ];
    let mut spec = SingerSpec::new(SR, notes);
    spec.breath = 0.15;
    let r = render(&spec);
    let hnr = 10.0 * (energy_hp(&r.harmonic, 50.0) / energy_hp(&r.aspiration, 50.0)).log10();
    println!("HNR da entrada: {hnr:.1} dB");
    // alcances da media de ciclos de cada modo (como no motor)
    for (name, past, fut, live, max_air) in [("Studio", 0.035, 0.035, false, -12.0), ("Track", 0.035, 0.010, false, -10.0), ("Live", 0.030, 0.0, true, -8.0)] {
        // a separacao e linear com as marcas fixas: cada componente verdadeira e separada sozinha
        let o = split_with_marks(&r.audio, &[&r.harmonic, &r.aspiration], past, fut, live);
        let (_, ghost) = &o[0];
        let (air_left, _) = &o[1];
        let ghost_db = 10.0 * (energy_hp(ghost, 50.0) / energy_hp(&r.harmonic, 50.0)).log10();
        let air_db = 10.0 * (energy_hp(air_left, 2000.0) / energy_hp(&r.aspiration, 2000.0)).log10();
        println!("{name}: harmonico no ruido (fantasma) {ghost_db:.1} dB, ar que fica na parte corrigida (> 2 kHz) {air_db:.1} dB");
        assert!(ghost_db < -22.0, "{name}: fantasma {ghost_db}");
        assert!(air_db < max_air, "{name}: ar {air_db}");
    }
}

/// Relacao harmonico/ruido acima de 2 kHz, medida pela propria media de ciclos (Studio).
fn hnr_hf(x: &[f32]) -> f64 {
    let o = split_with_marks(x, &[x], 0.035, 0.035, false);
    let (h, n) = &o[0];
    10.0 * (energy_hp(h, 2000.0) / energy_hp(n, 2000.0)).log10()
}

#[test]
fn breath_stays_air_when_shifted() {
    // voz soprosa cantando 60, MIDI pede 64: o PSOLA repete ciclos e, sem a separacao, o ar
    // repetido ganha a periodicidade da nota nova (zumbido)
    let notes = vec![SungNote::new(60.0, 0.0, 1.6).vibrato(5.5, 30.0)];
    let mut spec = SingerSpec::new(SR, notes);
    spec.breath = 0.08;
    let r = render(&spec);
    let mut p = c_major();
    p.source = Source::MidiTarget;
    let midi = |e: &mut Engine, pos: usize, _m: usize| {
        if pos == 0 {
            e.note_on(0, 64);
        }
    };
    let h_in = hnr_hf(&r.audio);
    p.breath_split = false;
    let (off, _) = run(&r.audio, SR, LatencyMode::Studio, &p, 512, &midi);
    p.breath_split = true;
    let (on, starved) = run(&r.audio, SR, LatencyMode::Studio, &p, 512, &midi);
    assert_eq!(starved, 0);
    let (h_off, h_on) = (hnr_hf(&off), hnr_hf(&on));
    println!("HNR > 2 kHz: entrada {h_in:.1} dB, saida sem separacao {h_off:.1} dB, com separacao {h_on:.1} dB");
    // o ar continua ar: a saida separada fica perto da entrada, e mais perto que a sem separacao
    assert!((h_on - h_in).abs() + 1.0 < (h_off - h_in).abs(), "separacao nao ajudou");
}

#[test]
fn breath_knob_sets_the_air() {
    // sem correcao: Breath 1 = a propria voz; 0 tira o ar, 2 dobra
    let notes = vec![SungNote::new(62.0, 0.0, 1.2).vibrato(5.5, 30.0)];
    let mut spec = SingerSpec::new(SR, notes);
    spec.breath = 0.08;
    let r = render(&spec);
    let mut p = c_major();
    p.brain.amount = 0.0;
    let air = |b: f32| {
        let mut p = p;
        p.breath = b;
        let (out, _) = run(&r.audio, SR, LatencyMode::Studio, &p, 512, &no_midi);
        // o ar que sobra = saida - harmonica verdadeira, acima de 2 kHz
        let d: Vec<f32> = out.iter().zip(&r.harmonic).map(|(o, h)| o - h).collect();
        10.0 * (energy_hp(&d, 2000.0) / energy_hp(&r.aspiration, 2000.0)).log10()
    };
    let (a0, a1, a2) = (air(0.0), air(1.0), air(2.0));
    println!("ar acima de 2 kHz em relacao ao original: Breath 0 {a0:+.1} dB, 1 {a1:+.1} dB, 2 {a2:+.1} dB");
    assert!(a1.abs() < 0.2);
    assert!(a0 < -8.0);
    assert!(a2 > 4.0 && a2 < 7.0);
}

#[test]
fn live_stays_aligned_with_large_blocks() {
    // as marcas do inicio de nota chegam atrasadas ao vivo; com blocos grandes o plano de graos
    // ja passou da borda e precisa reencontrar as marcas (senao a nota inteira sai deslocada)
    let r = phrase(0.0, 40.0);
    let mut p = c_major();
    p.brain.amount = 0.0;
    for block in [512usize, 1024] {
        let (out, _) = run(&r.audio, SR, LatencyMode::Live, &p, block, &no_midi);
        let (mut se, mut ss) = (0.0f64, 0.0f64);
        for (o, i) in out.iter().zip(&r.audio).skip(12000) {
            se += ((o - i) as f64).powi(2);
            ss += (*i as f64).powi(2);
        }
        let snr = 10.0 * (ss / se.max(1e-20)).log10();
        println!("ao vivo, blocos de {block}: SNR sem correcao {snr:.1} dB");
        assert!(snr > 40.0, "{snr}");
    }
}
