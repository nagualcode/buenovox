//! Render offline do BuenoVOX (o mesmo motor do plugin), dump das curvas e cantor sintetico.
//!
//! bvx_cli <entrada.wav> <saida.wav> [chave=valor ...]
//!   style=natural|classic|hard  source=scale|miditarget|midiscale  key=C..B  scale=major|minor|...
//!   amount=0..1  speed=ms  flex=0..1  humanize=0..1  tighten=0..1  vibrato=0..2  create=0..1
//!   scoop=0..1  throat=-1..1  pure=0|1  a4=Hz  latency=track|studio  block=512
//!   midi="0-2:60;2-4:62,65"   notas MIDI seguradas por intervalo de tempo (segundos)
//!   dump=curvas.csv           tempo, altura de entrada, alvo, saida, deslocamento (por quadro)
//!   compensate=1              remove a latencia do arquivo de saida (padrao 1)
//!
//! bvx_cli gen <saida.wav> [detune=cents] [vibrato=cents] [sr=48000]
//!   gera uma frase sintetica (C D E F G, desafinada alternadamente) com f0 conhecido

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
    for a in &args[1..] {
        let (k, v) = a.split_once('=').expect("use chave=valor");
        match k {
            "detune" => detune = v.parse().unwrap(),
            "vibrato" => vib = v.parse().unwrap(),
            "sr" => sr = v.parse().unwrap(),
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
    let r = render(&SingerSpec::new(sr, notes));
    write_wav(out, &r.audio, &r.audio, sr as u32);
    for s in &r.spans {
        eprintln!("nota {:>5.1} cantada {:>7.2} de {:.2}s a {:.2}s", s.intended, s.sung, s.start as f32 / sr, s.end as f32 / sr);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
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
            "pure" => p.pure = v == "1",
            "a4" => p.brain.a4 = f(),
            "latency" => mode = if v == "track" { LatencyMode::Track } else { LatencyMode::Studio },
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
    eprintln!("psola (inicios, travados, passos estimados): {:?}", e.psola_debug());
}
