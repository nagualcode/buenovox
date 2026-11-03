//! Vozes sinteticas variadas, com f0 conhecido, para treinar e medir o detector neural.
//!
//! Diferente do cantor da regua (`testsig`, aditivo e limpo), este e fonte-filtro pulso a pulso:
//! cada ciclo glotal e um impulso de banda limitada com jitter e shimmer, passando por uma
//! inclinacao espectral e por formantes que mudam de vogal a cada nota. Assim e facil produzir os
//! casos em que detectores classicos erram:
//! * voz rouca (fry/diplofonia): pulsos alternadamente fortes e fracos, periodo aparente dobrado;
//! * belting com sub-harmonico: alternancia leve;
//! * voz soprosa: ruido de aspiracao pulsado com o ciclo;
//! * vogais fechadas ("u", "i"), com o 1o formante perto do 2o harmonico;
//! * microfone fino (sem fundamental), distorcao, reverb e ruido de fundo;
//! * consoantes, respiracoes e silencios (o rotulo e "sem voz").
//!
//! O rotulo e a taxa de pulsos (a nota que o cantor quer cantar), inclusive na voz rouca.

use super::util::{Biquad, Drift, Rng};

pub struct TrainVoice {
    pub audio: Vec<f32>,
    /// f0 verdadeiro por amostra (Hz); 0 = sem voz.
    pub f0: Vec<f32>,
}

/// Formantes de vogais (Hz), voz adulta media.
const VOWELS: [[f32; 5]; 7] = [
    [730.0, 1090.0, 2440.0, 3400.0, 4500.0], // a
    [530.0, 1840.0, 2480.0, 3400.0, 4500.0], // e
    [270.0, 2290.0, 3010.0, 3400.0, 4500.0], // i
    [570.0, 840.0, 2410.0, 3400.0, 4500.0],  // o
    [300.0, 870.0, 2240.0, 3400.0, 4500.0],  // u
    [500.0, 1500.0, 2500.0, 3400.0, 4500.0], // schwa
    [660.0, 1720.0, 2410.0, 3400.0, 4500.0], // ae
];
const BANDS: [f32; 5] = [70.0, 95.0, 120.0, 160.0, 200.0];

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Normal,
    Fry,
    Belt,
    Breathy,
}

struct Note {
    start: usize,
    len: usize,
    midi: f32,
    legato: bool,
    vib_rate: f32,
    vib_cents: f32,
    scoop: f32,
    vowel: [f32; 5],
    mode: Mode,
    alt: f32,
    att: f32,
    rel: f32,
    amp: f32,
}

/// Ressonador de 2 polos com ganho 1 em DC (sintese de formantes em cascata, como no Klatt).
#[derive(Clone, Copy, Default)]
struct Reson {
    a: f32,
    b: f32,
    c: f32,
    y1: f32,
    y2: f32,
}

impl Reson {
    fn set(&mut self, f: f32, bw: f32, sr: f32) {
        let r = (-std::f32::consts::PI * bw / sr).exp();
        let th = std::f32::consts::TAU * (f / sr).min(0.49);
        self.b = 2.0 * r * th.cos();
        self.c = -r * r;
        self.a = 1.0 - self.b - self.c;
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.a * x + self.b * self.y1 + self.c * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Reverb de Schroeder (4 pentes em paralelo + 2 passa-tudo em serie).
struct Reverb {
    combs: Vec<(Vec<f32>, usize, f32)>,
    aps: Vec<(Vec<f32>, usize)>,
}

impl Reverb {
    fn new(sr: f32, rt60: f32, rng: &mut Rng) -> Self {
        let combs = [0.0297f32, 0.0371, 0.0411, 0.0437]
            .iter()
            .map(|&d| {
                let n = ((d * (0.85 + 0.3 * rng.uniform())) * sr) as usize;
                let g = 10f32.powf(-3.0 * n as f32 / (sr * rt60));
                (vec![0.0; n.max(1)], 0, g)
            })
            .collect();
        let aps = [0.005f32, 0.0017].iter().map(|&d| (vec![0.0; ((d * sr) as usize).max(1)], 0)).collect();
        Self { combs, aps }
    }

    fn process(&mut self, x: f32) -> f32 {
        let mut y = 0.0;
        for (buf, i, g) in &mut self.combs {
            let v = buf[*i];
            buf[*i] = x + v * *g;
            *i = (*i + 1) % buf.len();
            y += v;
        }
        y *= 0.25;
        for (buf, i) in &mut self.aps {
            let v = buf[*i];
            let w = y + 0.5 * v;
            buf[*i] = w;
            *i = (*i + 1) % buf.len();
            y = v - 0.5 * w;
        }
        y
    }
}

fn pick<T: Copy>(rng: &mut Rng, v: &[T]) -> T { v[(rng.uniform() * v.len() as f32) as usize % v.len()] }

fn range(rng: &mut Rng, a: f32, b: f32) -> f32 { a + (b - a) * rng.uniform() }

pub fn render_training(seed: u64, seconds: f32, sr: f32) -> TrainVoice {
    let mut rng = Rng::new(seed);
    let len = (seconds * sr) as usize;
    let mut f0v = vec![0.0f32; len];

    // o cantor
    let center = range(&mut rng, 40.0, 76.0);
    let spread = range(&mut rng, 3.0, 10.0);
    let fscale = range(&mut rng, 0.85, 1.25);
    let base_breath = range(&mut rng, 0.0, 1.0).powi(2) * 0.3;
    let tilt_hz = range(&mut rng, 700.0, 4000.0);
    let jitter = range(&mut rng, 0.002, 0.012);
    let shimmer = range(&mut rng, 0.02, 0.12);

    // linha do tempo: notas, consoantes, respiracoes, silencios
    let mut notes: Vec<Note> = Vec::new();
    let mut events: Vec<(usize, usize, u8, f32)> = Vec::new(); // inicio, duracao, tipo, nivel
    let mut t = (range(&mut rng, 0.05, 0.4) * sr) as usize;
    let mut prev_note = false;
    while t < len {
        let r = rng.uniform();
        if r < 0.72 {
            let legato = prev_note && rng.uniform() < 0.5;
            let mode = match rng.uniform() {
                x if x < 0.12 => Mode::Fry,
                x if x < 0.22 => Mode::Belt,
                x if x < 0.32 => Mode::Breathy,
                _ => Mode::Normal,
            };
            let alt = match mode {
                Mode::Fry => range(&mut rng, 0.1, 0.7),
                Mode::Belt => range(&mut rng, 0.7, 0.93),
                _ => 1.0,
            };
            let mut vowel = pick(&mut rng, &VOWELS);
            for f in &mut vowel {
                *f *= fscale * range(&mut rng, 0.93, 1.07);
            }
            let n = Note {
                start: t,
                len: (range(&mut rng, 0.12, 1.6) * sr) as usize,
                midi: (center + range(&mut rng, -spread, spread)).clamp(36.0, 84.0),
                legato,
                vib_rate: range(&mut rng, 4.5, 7.0),
                vib_cents: if rng.uniform() < 0.6 { range(&mut rng, 10.0, 90.0) } else { 0.0 },
                scoop: if rng.uniform() < 0.4 { range(&mut rng, -1.5, 0.5) } else { 0.0 },
                vowel,
                mode,
                alt,
                att: range(&mut rng, 0.01, 0.08),
                rel: range(&mut rng, 0.02, 0.15),
                amp: range(&mut rng, 0.3, 1.0),
            };
            t += n.len;
            notes.push(n);
            prev_note = true;
        } else {
            let kind = (rng.uniform() * 4.0) as u8;
            let dur = match kind {
                0 => range(&mut rng, 0.05, 0.6), // silencio
                1 => range(&mut rng, 0.04, 0.2), // sibilante
                2 => range(&mut rng, 0.2, 0.5),  // respiracao
                _ => range(&mut rng, 0.01, 0.03), // plosiva
            };
            let n = (dur * sr) as usize;
            if kind > 0 {
                events.push((t, n, kind, range(&mut rng, 0.02, 0.3)));
            }
            t += n;
            prev_note = false;
        }
    }

    // fonte glotal pulso a pulso + aspiracao, por nota
    let mut src = vec![0.0f32; len + 64];
    let mut asp = vec![0.0f32; len];
    let mut formant = vec![[0.0f32; 5]; 0];
    formant.resize(len, VOWELS[5]);
    let (zeta, wn) = (0.6f32, std::f32::consts::TAU * 6.0);
    let mut x = 0.0f32;
    let mut v = 0.0f32;
    let mut drift = Drift::new(0.08, 0.6);
    let mut phase = 0.0f64;
    let mut pulse_k = 0u64;
    let mut prev_vowel = VOWELS[5];
    let mut next_period_scale = 1.0f32;
    for n in &notes {
        if !n.legato {
            x = n.midi + n.scoop;
            v = 0.0;
        }
        let breath = match n.mode {
            Mode::Breathy => (base_breath + 0.15) * 2.0,
            _ => base_breath,
        };
        for j in 0..n.len {
            let i = n.start + j;
            if i >= len {
                break;
            }
            let tn = j as f32 / sr;
            let acc = wn * wn * (n.midi - x) - 2.0 * zeta * wn * v;
            v += acc / sr;
            x += v / sr;
            let env_v = ((tn - 0.2) / 0.3).clamp(0.0, 1.0);
            let vib = env_v * n.vib_cents / 100.0 * (std::f32::consts::TAU * n.vib_rate * tn).sin();
            let m = x + vib + drift.step(1.0 / sr, &mut rng);
            let f0 = 440.0 * ((m - 69.0) / 12.0).exp2();
            let att = if n.legato { 1.0 } else { (tn / n.att).min(1.0) };
            let rel = ((n.len as f32 / sr - tn) / n.rel).clamp(0.0, 1.0);
            let env = n.amp * att * rel;
            if env > 0.05 * n.amp {
                f0v[i] = f0;
            }
            // vogal: transicao de 50 ms a partir da anterior
            let a = (tn / 0.05).min(1.0);
            let mut fv = [0.0; 5];
            for k in 0..5 {
                fv[k] = prev_vowel[k] + (n.vowel[k] - prev_vowel[k]) * a;
            }
            formant[i] = fv;
            // pulsos
            let inc = f0 as f64 / sr as f64 * next_period_scale as f64;
            let before = phase;
            phase += inc;
            if phase >= 1.0 {
                phase -= 1.0;
                let frac = (phase / inc).min(1.0) as f32; // quanto do passo ja passou do pulso
                let pos = i as f32 - frac;
                let alt = if pulse_k % 2 == 1 { n.alt } else { 1.0 };
                let shim = (1.0 + shimmer * rng.gauss()).max(0.2);
                let amp = env * alt * shim;
                // impulso de banda limitada: seno cardinal com janela de Hann, 8 amostras por lado
                let c = pos.floor() as i64;
                let fr = pos - c as f32;
                for k in -7i64..=8 {
                    let d = k as f32 - fr;
                    let w = 0.5 + 0.5 * (std::f32::consts::PI * d / 8.5).cos();
                    let sinc = if d.abs() < 1e-6 { 1.0 } else { (std::f32::consts::PI * d).sin() / (std::f32::consts::PI * d) };
                    let idx = c + k;
                    if idx >= 0 && (idx as usize) < src.len() {
                        src[idx as usize] += amp * w * sinc;
                    }
                }
                pulse_k += 1;
                let jit = if n.mode == Mode::Fry { 3.0 * jitter } else { jitter };
                next_period_scale = 1.0 / (1.0 + jit * rng.gauss()).clamp(0.8, 1.25);
            }
            let _ = before;
            // aspiracao pulsada com o ciclo (mais forte na fase aberta)
            let open = 0.5 + 0.5 * (std::f64::consts::TAU * (phase - 0.3)).cos() as f32;
            asp[i] = breath * env * (0.4 + 0.6 * open) * rng.gauss() * 0.15;
        }
        prev_vowel = n.vowel;
    }

    // inclinacao espectral da fonte + formantes
    let mut tilt = Biquad::lowpass(sr as f64, tilt_hz as f64, 0.6);
    let mut dc = Biquad::highpass(sr as f64, 40.0, 0.7);
    let mut res = [Reson::default(); 5];
    let mut bands = BANDS;
    for b in &mut bands {
        *b *= range(&mut rng, 0.7, 1.4);
    }
    let mut audio = vec![0.0f32; len];
    for i in 0..len {
        if i % 32 == 0 {
            for k in 0..5 {
                res[k].set(formant[i][k].min(0.45 * sr), bands[k], sr);
            }
        }
        let mut y = dc.process(tilt.process(src[i])) * 4.0 + asp[i];
        for r in &mut res {
            y = r.process(y);
        }
        audio[i] = y;
    }
    // normaliza a voz antes dos ruidos
    let peak = audio.iter().fold(0.0f32, |a, &s| a.max(s.abs())).max(1e-9);
    for s in &mut audio {
        *s *= 0.5 / peak;
    }

    // consoantes e respiracoes (sem voz)
    for &(start, n, kind, level) in &events {
        let mut f = match kind {
            1 => Biquad::highpass(sr as f64, range(&mut rng, 3500.0, 6000.0) as f64, 0.7),
            2 => Biquad::lowpass(sr as f64, range(&mut rng, 900.0, 2500.0) as f64, 0.7),
            _ => Biquad::highpass(sr as f64, 300.0, 0.7),
        };
        for j in 0..n {
            let i = start + j;
            if i >= len {
                break;
            }
            let e = (std::f32::consts::PI * j as f32 / n as f32).sin();
            audio[i] += level * e * f.process(rng.gauss()) * 0.5;
        }
    }

    // pos-processamento: microfone fino, distorcao, reverb, ruido de fundo, ganho
    if rng.uniform() < 0.15 {
        let fc = range(&mut rng, 120.0, 400.0) as f64;
        let (mut a, mut b) = (Biquad::highpass(sr as f64, fc, 0.7), Biquad::highpass(sr as f64, fc, 0.7));
        for s in &mut audio {
            *s = b.process(a.process(*s));
        }
    }
    if rng.uniform() < 0.1 {
        let drive = range(&mut rng, 2.0, 10.0);
        let norm = drive.tanh();
        for s in &mut audio {
            *s = (drive * *s).tanh() / norm * 0.5;
        }
    }
    if rng.uniform() < 0.35 {
        let mut rv = Reverb::new(sr, range(&mut rng, 0.3, 2.0), &mut rng);
        let wet = range(&mut rng, 0.1, 0.5);
        for s in &mut audio {
            *s = (1.0 - wet) * *s + wet * rv.process(*s);
        }
    }
    if rng.uniform() < 0.5 {
        let rms = (audio.iter().map(|s| s * s).sum::<f32>() / len.max(1) as f32).sqrt();
        let snr = range(&mut rng, 10.0, 45.0);
        let g = rms * 10f32.powf(-snr / 20.0);
        let pink = rng.uniform() < 0.5;
        let mut lp = Biquad::lowpass(sr as f64, 1500.0, 0.5);
        for s in &mut audio {
            let n = rng.gauss();
            *s += g * if pink { 2.0 * lp.process(n) } else { n };
        }
    }
    let gain = 10f32.powf(range(&mut rng, -30.0, -1.0) / 20.0);
    let peak = audio.iter().fold(0.0f32, |a, &s| a.max(s.abs())).max(1e-9);
    for s in &mut audio {
        *s *= gain / peak;
    }
    TrainVoice { audio, f0: f0v }
}
