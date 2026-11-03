//! Render offline do BuenoVOX (o mesmo motor do plugin), dump das curvas e cantor sintetico.
//!
//! bvx_cli <entrada.wav> <saida.wav> [chave=valor ...]
//!   style=natural|classic|hard  source=scale|miditarget|midiscale  key=C..B  scale=major|minor|...
//!   amount=0..1  speed=ms  flex=0..1  humanize=0..1  tighten=0..1  vibrato=0..2  create=0..1
//!   scoop=0..1  throat=-1..1  breath=0..2  split=0|1  neural=0|1  pure=0|1  a4=Hz  latency=live|track|studio
//!   block=512
//!   midi="0-2:60;2-4:62,65"   notas MIDI seguradas por intervalo de tempo (segundos)
//!   dump=curvas.csv           tempo, altura de entrada, alvo, saida, deslocamento (por quadro)
//!   compensate=1              remove a latencia do arquivo de saida (padrao 1)
//!
//! bvx_cli gen <saida.wav> [detune=cents] [vibrato=cents] [breath=0.01] [sr=48000]
//!   gera uma frase sintetica (C D E F G, desafinada alternadamente) com f0 conhecido
//!
//! Detector neural (train/):
//! bvx_cli synth <saida.wav> [seed=1] [seconds=20] [sr=48000]
//!   voz de treino variada (voicegen) e <saida.wav>.f0.csv com o f0 verdadeiro a cada 1 ms
//! bvx_cli dataset <saida.bin> [seed=1] [files=10] [seconds=20] [stride=1]
//!   vozes de treino -> quadros com entradas da rede, rotulo e decisao do YIN
//! bvx_cli features <entrada.wav> <saida.bin> [net=pesos.bin] [fusion=a,b,c,d,e,f] [stride=1]
//!   o mesmo para audio real (rotulo -1); com net=, a decisao gravada e a da fusao

use buenovox::dsp::testsig::{render, SingerSpec, SungNote};
use buenovox::dsp::tonal::{NOTE_NAMES, SCALE_NAMES};
use buenovox::dsp::{Engine, EngineParams, LatencyMode, Source, Style};
use std::io::Write;
use std::time::Instant;

fn read_wav(path: &str) -> (Vec<f32>, Vec<f32>, u32) {
    let mut r = hound::WavReader::open(path).expect("nao abriu o wav de entrada");
    let spec = r.spec();
    let ch = spec.channels as usize;
    let data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.unwrap() as f32 * scale).collect()
        }
    };
    let n = data.len() / ch;
    let l: Vec<f32> = (0..n).map(|i| data[i * ch]).collect();
    let r: Vec<f32> = (0..n).map(|i| data[i * ch + (ch > 1) as usize]).collect();
    (l, r, spec.sample_rate)
}

fn write_wav(path: &str, l: &[f32], r: &[f32], sr: u32) {
    let spec = hound::WavSpec { channels: 2, sample_rate: sr, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let mut w = hound::WavWriter::create(path, spec).expect("nao criou o wav de saida");
    for (a, b) in l.iter().zip(r) {
        w.write_sample(*a).unwrap();
        w.write_sample(*b).unwrap();
    }
    w.finalize().unwrap();
}

fn parse_key(s: &str) -> Option<u8> {
    NOTE_NAMES.iter().position(|n| n.eq_ignore_ascii_case(s)).map(|i| i as u8).or(match s {
        "Db" | "db" => Some(1),
        "D#" | "d#" => Some(3),
        "Gb" | "gb" => Some(6),
        "G#" | "g#" => Some(8),
        "A#" | "a#" => Some(10),
        _ => None,
    })
}

fn parse_scale(s: &str) -> Option<u8> {
    let norm = |x: &str| x.to_ascii_lowercase().replace(['.', ' ', '_', '-'], "");
    SCALE_NAMES.iter().position(|n| norm(n).starts_with(&norm(s))).map(|i| i as u8)
}

/// "0-2:60;2-4:62,65" -> [(t0, t1, [notas])]
fn parse_midi(s: &str) -> Vec<(f64, f64, Vec<u8>)> {
    s.split(';')
        .filter(|x| !x.trim().is_empty())
        .map(|seg| {
            let (range, notes) = seg.split_once(':').expect("midi: use t0-t1:nota,nota");
            let (a, b) = range.split_once('-').expect("midi: use t0-t1");
            let notes = notes.split(',').map(|n| n.trim().parse().expect("nota MIDI")).collect();
            (a.trim().parse().unwrap(), b.trim().parse().unwrap(), notes)
        })
        .collect()
}

fn gen(args: &[String]) {
    let out = &args[0];
    let mut detune = 35.0f32;
    let mut vib = 40.0f32;
    let mut sr = 48000.0f32;
    let mut breath = 0.01f32;
    for a in &args[1..] {
        let (k, v) = a.split_once('=').expect("use chave=valor");
        match k {
            "detune" => detune = v.parse().unwrap(),
            "vibrato" => vib = v.parse().unwrap(),
            "sr" => sr = v.parse().unwrap(),
            "breath" => breath = v.parse().unwrap(),
            _ => panic!("chave desconhecida: {k}"),
        }
    }
    let notes = [60.0, 62.0, 64.0, 65.0, 67.0]
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let n = SungNote::new(m, sign * detune, 0.9).vibrato(5.5, vib);
            if i == 2 { n.gap(0.25) } else { n }
        })
        .collect();
    let mut spec = SingerSpec::new(sr, notes);
    spec.breath = breath;
    let r = render(&spec);
    write_wav(out, &r.audio, &r.audio, sr as u32);
    for s in &r.spans {
        eprintln!("nota {:>5.1} cantada {:>7.2} de {:.2}s a {:.2}s", s.intended, s.sung, s.start as f32 / sr, s.end as f32 / sr);
    }
}

/// Quadros de analise de `x`: [tempo (s), rotulo (Hz, -1 = desconhecido), f0 decidido (Hz),
/// aperiodicidade, nivel (dB), f0 da rede, voz da rede, entradas da rede...].
fn frames(x: &[f32], sr: f32, label: Option<&[f32]>, net: Option<&[u8]>, fusion: Option<buenovox::dsp::pitch::Fusion>, stride: u64, out: &mut Vec<f32>) {
    use buenovox::dsp::neural::PitchNet;
    use buenovox::dsp::pitch::PitchTracker;
    let mut pt = PitchTracker::new(sr);
    pt.want_features = true;
    if let Some(bytes) = net {
        pt.set_net(PitchNet::from_bytes(bytes).expect("pesos invalidos"));
    }
    if let Some(f) = fusion {
        pt.fusion = f;
    }
    let mut seen = 0u64;
    for chunk in x.chunks(pt.hop) {
        pt.push(chunk);
        if pt.analyzed > seen {
            seen = pt.analyzed;
            if seen % stride != 0 {
                continue;
            }
            let c = pt.span_center;
            let lab = match label {
                Some(l) => l.get(c.round() as usize).copied().unwrap_or(0.0),
                None => -1.0,
            };
            let fr = pt.last_raw;
            out.extend_from_slice(&[(c / sr as f64) as f32, lab, fr.f0, fr.aper, fr.db, pt.net_out.f0, pt.net_out.voiced]);
            out.extend_from_slice(&pt.feat[..]);
        }
    }
}

fn write_frames(path: &str, data: &[f32]) {
    use buenovox::dsp::neural::FEAT_LEN;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("nao criou o arquivo"));
    f.write_all(b"BVXF").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&7u32.to_le_bytes()).unwrap();
    f.write_all(&(FEAT_LEN as u32).to_le_bytes()).unwrap();
    for v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn kv(args: &[String]) -> std::collections::HashMap<String, String> {
    args.iter().filter_map(|a| a.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn synth(args: &[String]) {
    use buenovox::dsp::voicegen::render_training;
    let o = kv(&args[1..]);
    let seed: u64 = o.get("seed").map_or(1, |v| v.parse().unwrap());
    let secs: f32 = o.get("seconds").map_or(20.0, |v| v.parse().unwrap());
    let sr: f32 = o.get("sr").map_or(48000.0, |v| v.parse().unwrap());
    let tv = render_training(seed, secs, sr);
    write_wav(&args[0], &tv.audio, &tv.audio, sr as u32);
    let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{}.f0.csv", args[0])).unwrap());
    let step = (sr / 1000.0) as usize;
    for (k, v) in tv.f0.iter().step_by(step).enumerate() {
        writeln!(f, "{:.3},{:.3}", k as f32 * 0.001, v).unwrap();
    }
}

fn dataset(args: &[String]) {
    use buenovox::dsp::voicegen::render_training;
    let o = kv(&args[1..]);
    let seed: u64 = o.get("seed").map_or(1, |v| v.parse().unwrap());
    let files: u64 = o.get("files").map_or(10, |v| v.parse().unwrap());
    let secs: f32 = o.get("seconds").map_or(20.0, |v| v.parse().unwrap());
    let stride: u64 = o.get("stride").map_or(1, |v| v.parse().unwrap());
    let mut data = Vec::new();
    for k in 0..files {
        let s = seed * 1_000_003 + k;
        let sr = [44_100.0f32, 48_000.0, 48_000.0, 44_100.0, 22_050.0, 96_000.0][(s % 6) as usize];
        let tv = render_training(s, secs, sr);
        frames(&tv.audio, sr, Some(&tv.f0), None, None, stride, &mut data);
    }
    write_frames(&args[0], &data);
}

fn features(args: &[String]) {
    let (l, r, sr) = read_wav(&args[0]);
    let x: Vec<f32> = l.iter().zip(&r).map(|(a, b)| 0.5 * (a + b)).collect();
    let o = kv(&args[2..]);
    let net = o.get("net").map(|p| std::fs::read(p).expect("nao leu os pesos"));
    let fusion = o.get("fusion").map(|v| {
        let f: Vec<f32> = v.split(',').map(|x| x.parse().unwrap()).collect();
        buenovox::dsp::pitch::Fusion { voiced: f[0], conf: f[1], max_aper: f[2], voice_on: f[3], voice_off: f[4], min_aper_off: f[5], agree_st: f.get(6).copied().unwrap_or(1.0) }
    });
    let stride: u64 = o.get("stride").map_or(1, |v| v.parse().unwrap());
    let mut data = Vec::new();
    frames(&x, sr as f32, None, net.as_deref(), fusion, stride, &mut data);
    write_frames(&args[1], &data);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("synth") if args.len() >= 2 => return synth(&args[1..]),
        Some("dataset") if args.len() >= 2 => return dataset(&args[1..]),
        Some("features") if args.len() >= 3 => return features(&args[1..]),
        _ => {}
    }
    if args.first().map(String::as_str) == Some("gen") && args.len() >= 2 {
        gen(&args[1..]);
        return;
    }
    if args.len() < 2 {
        eprintln!("uso: bvx_cli <entrada.wav> <saida.wav> [chave=valor ...]   |   bvx_cli gen <saida.wav>");
        std::process::exit(2);
    }
    let (in_l, in_r, sr) = read_wav(&args[0]);
    let mut p = EngineParams::default();
    let mut mode = LatencyMode::Studio;
    let mut block = 512usize;
    let mut midi = Vec::new();
    let mut dump: Option<String> = None;
    let mut compensate = true;
    for a in &args[2..] {
        let (k, v) = a.split_once('=').expect("use chave=valor");
        let f = || v.parse::<f32>().expect("numero");
        match k {
            "style" => {
                p.brain.style = match v {
                    "natural" => Style::Natural,
                    "classic" => Style::Classic,
                    "hard" => Style::Hard,
                    _ => panic!("style: natural|classic|hard"),
                }
            }
            "source" => {
                p.source = match v {
                    "scale" => Source::Scale,
                    "miditarget" => Source::MidiTarget,
                    "midiscale" => Source::MidiScale,
                    _ => panic!("source: scale|miditarget|midiscale"),
                }
            }
            "key" => p.key = parse_key(v).expect("tonalidade"),
            "scale" => p.scale = parse_scale(v).expect("escala"),
            "amount" => p.brain.amount = f(),
            "speed" => p.brain.speed_ms = f(),
            "flex" => p.brain.flex = f(),
            "humanize" => p.brain.humanize = f(),
            "tighten" => p.brain.tighten = f(),
            "vibrato" => p.brain.vibrato = f(),
            "create" => p.brain.vib_create = f(),
            "scoop" => p.brain.scoop = f(),
            "throat" => p.throat = f(),
            "breath" => p.breath = f(),
            "split" => p.breath_split = v != "0",
            "neural" => p.neural = v != "0",
            "pure" => p.pure = v == "1",
            "a4" => p.brain.a4 = f(),
            "latency" => {
                mode = match v {
                    "live" => LatencyMode::Live,
                    "track" => LatencyMode::Track,
                    _ => LatencyMode::Studio,
                }
            }
            "block" => block = v.parse().unwrap(),
            "midi" => midi = parse_midi(v),
            "dump" => dump = Some(v.to_string()),
            "compensate" => compensate = v != "0",
            _ => panic!("chave desconhecida: {k}"),
        }
    }

    let mut e = Engine::new(sr as f32, mode);
    let lat = e.latency();
    let n = in_l.len();
    let total = n + if compensate { lat } else { 0 };
    let (mut out_l, mut out_r) = (vec![0.0f32; total], vec![0.0f32; total]);
    let zeros = vec![0.0f32; block];
    let mut held = [false; 128];
    let mut csv = dump.as_ref().map(|path| {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("dump"));
        writeln!(f, "t,input,target,output,shift,vib_cents,center,local,note").unwrap();
        f
    });
    let t0 = Instant::now();
    let mut pos = 0;
    let mut dumped = 0u64;
    while pos < total {
        let m = block.min(total - pos);
        e.set_params(&p);
        // notas MIDI: liga/desliga conforme a linha do tempo, com deslocamento exato no bloco
        for k in 0..m {
            let t = (pos + k) as f64 / sr as f64;
            let mut want = [false; 128];
            for (a, b, notes) in &midi {
                if t >= *a && t < *b {
                    for &nn in notes {
                        want[nn as usize & 127] = true;
                    }
                }
            }
            for nn in 0..128 {
                if want[nn] != held[nn] {
                    if want[nn] { e.note_on(k, nn as u8) } else { e.note_off(k, nn as u8) }
                    held[nn] = want[nn];
                }
            }
        }
        let (il, ir) = if pos + m <= n { (&in_l[pos..pos + m], &in_r[pos..pos + m]) } else { (&zeros[..m], &zeros[..m]) };
        let (il, ir) = if pos < n && pos + m > n {
            // ultimo bloco parcial: completa com zeros
            let mut a = in_l[pos..n].to_vec();
            a.resize(m, 0.0);
            let mut b = in_r[pos..n].to_vec();
            b.resize(m, 0.0);
            (a, b)
        } else {
            (il.to_vec(), ir.to_vec())
        };
        e.process(&il, &ir, &mut out_l[pos..pos + m], &mut out_r[pos..pos + m]);
        if let Some(f) = csv.as_mut() {
            for (t, mt) in e.frames_from(dumped) {
                let t = t / sr as f64;
                let d = mt.dbg;
                writeln!(f, "{t:.4},{:.4},{:.4},{:.4},{:.4},{:.1},{:.4},{:.4},{}", mt.input, mt.target, mt.output, mt.shift, mt.vib_cents, d[0], d[1], d[2])
                    .unwrap();
            }
            dumped = e.frame_count();
        }
        pos += m;
    }
    let el = t0.elapsed().as_secs_f64();
    let audio_s = n as f64 / sr as f64;
    let (ol, or) = if compensate { (&out_l[lat..], &out_r[lat..]) } else { (&out_l[..], &out_r[..]) };
    write_wav(&args[1], ol, or, sr);
    eprintln!(
        "{:.2}s de audio em {:.3}s ({:.0}x tempo real), latencia {} amostras ({:.1} ms), fome {}",
        audio_s,
        el,
        audio_s / el,
        lat,
        lat as f64 * 1000.0 / sr as f64,
        e.starved()
    );
}
