//! O detector neural de apoio: a inferencia em Rust e a mesma rede treinada em PyTorch, e a
//! fusao com o YIN melhora os casos dificeis sem piorar os faceis.

use buenovox::dsp::neural::{PitchNet, BINS, FEAT_LEN};

/// Saidas do PyTorch (train/check.py) para entradas reais, conferidas com a inferencia daqui.
#[test]
fn rust_inference_matches_pytorch() {
    let data = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/models/pitchnet_check.bin")).expect("models/pitchnet_check.bin");
    assert_eq!(&data[..4], b"BVXC");
    let n = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let fl = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    assert_eq!(fl, FEAT_LEN);
    let f32s: Vec<f32> = data[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let rec = FEAT_LEN + BINS + 1;
    let mut net = PitchNet::embedded().expect("pesos embutidos");
    let mut worst_p = 0.0f32;
    let mut worst_v = 0.0f32;
    for i in 0..n {
        let r = &f32s[i * rec..(i + 1) * rec];
        let mut feat = [0.0f32; FEAT_LEN];
        feat.copy_from_slice(&r[..FEAT_LEN]);
        let logits = &r[FEAT_LEN..FEAT_LEN + BINS];
        let mut probs = [0.0f32; BINS];
        let v = net.run(&feat, &mut probs);
        // softmax de referencia
        let mx = logits.iter().fold(f32::MIN, |m, &x| m.max(x));
        let sum: f32 = logits.iter().map(|l| (l - mx).exp()).sum();
        for (p, l) in probs.iter().zip(logits) {
            worst_p = worst_p.max((p - (l - mx).exp() / sum).abs());
        }
        worst_v = worst_v.max((v - r[FEAT_LEN + BINS]).abs());
    }
    println!("{n} quadros: maior diferenca de probabilidade {worst_p:.2e}, de logit de voz {worst_v:.2e}");
    assert!(worst_p < 1e-4 && worst_v < 1e-3);
}

/// Erros da decisao quadro a quadro (antes da mediana) contra o f0 verdadeiro.
struct Score {
    gross: f32,
    octave: f32,
    recall: f32,
    false_voice: f32,
}

fn score(x: &[f32], f0: &[f32], sr: f32, neural: bool) -> Score {
    use buenovox::dsp::pitch::PitchTracker;
    let mut pt = PitchTracker::new(sr);
    if neural {
        assert!(pt.set_neural(true));
    }
    let (mut both, mut gross, mut oct, mut ref_v, mut found, mut unv, mut false_v) = (0, 0, 0, 0, 0, 0, 0);
    let mut seen = 0;
    for chunk in x.chunks(pt.hop) {
        pt.push(chunk);
        if pt.analyzed == seen {
            continue;
        }
        seen = pt.analyzed;
        let r = f0.get(pt.span_center.round() as usize).copied().unwrap_or(0.0);
        let e = pt.last_raw.f0;
        if r > 0.0 {
            ref_v += 1;
        } else {
            unv += 1;
            false_v += (e > 0.0) as u32;
        }
        if r > 0.0 && e > 0.0 {
            both += 1;
            found += 1;
            let c = 1200.0 * (e / r).log2();
            if c.abs() > 50.0 {
                gross += 1;
                let k = (c.abs() / 1200.0).round();
                oct += (k >= 1.0 && (c.abs() - 1200.0 * k).abs() < 50.0) as u32;
            }
        }
    }
    let d = |a: u32, b: u32| a as f32 / b.max(1) as f32;
    Score { gross: d(gross, both), octave: d(oct, both), recall: d(found, ref_v), false_voice: d(false_v, unv) }
}

/// Vozes de treino com sementes que o treino nunca viu: rouca, soprosa, belting, vogais
/// fechadas, microfone fino, reverb, ruido. A rede tem que diminuir os erros grosseiros sem
/// perder voz.
#[test]
fn neural_assist_reduces_gross_errors_on_hard_voices() {
    use buenovox::dsp::voicegen::render_training;
    let (mut y, mut n) = ([0.0f32; 4], [0.0f32; 4]);
    let seeds = 900_000..900_012u64;
    let count = seeds.clone().count() as f32;
    for seed in seeds {
        let sr = if seed % 2 == 0 { 48_000.0 } else { 44_100.0 };
        let tv = render_training(seed, 12.0, sr);
        let a = score(&tv.audio, &tv.f0, sr, false);
        let b = score(&tv.audio, &tv.f0, sr, true);
        for (acc, s) in [(&mut y, a), (&mut n, b)] {
            acc[0] += s.gross / count;
            acc[1] += s.octave / count;
            acc[2] += s.recall / count;
            acc[3] += s.false_voice / count;
        }
    }
    for (name, s) in [("YIN", y), ("YIN + rede", n)] {
        println!(
            "{name:>10}: erro grosseiro {:.2}%, de oitava {:.2}%, voz achada {:.1}%, voz falsa {:.1}%",
            100.0 * s[0],
            100.0 * s[1],
            100.0 * s[2],
            100.0 * s[3]
        );
    }
    assert!(n[0] < 0.7 * y[0], "a rede deveria cortar os erros grosseiros");
    assert!(n[2] > y[2] - 0.01, "sem perder voz");
    assert!(n[3] < y[3] + 0.01, "sem inventar voz");
}
