//! Cantor sintetico com f0 conhecido, para a regua (testes e CLI).
//!
//! Sintese aditiva fonte-filtro: harmonicos com inclinacao de -6 dB/oitava (pulso glotal +
//! radiacao), moldados pelo envelope de formantes de uma vogal. O contorno de f0 segue o modelo
//! de Saitou et al.: cada nota e um alvo, a transicao entre notas ligadas e a resposta de um
//! sistema de 2a ordem subamortecido (overshoot), o ataque depois de silencio comeca abaixo
//! (scoop), o vibrato entra depois de um atraso e cresce, e ha flutuacao fina aleatoria.
//! Cada nota pode ser cantada desafinada (`detune_cents`): a resposta certa e conhecida.

use super::util::{Biquad, Rng};

#[derive(Clone, Copy, Debug)]
pub struct SungNote {
    /// Nota pretendida (MIDI afinado com A4 = 440).
    pub midi: f32,
    /// Erro de afinacao do cantor nesta nota (cents).
    pub detune_cents: f32,
    /// Duracao (s).
    pub dur: f32,
    /// Silencio depois da nota (s); 0 = ligada a proxima.
    pub gap: f32,
    /// Vibrato: taxa (Hz) e extensao de pico (cents).
    pub vib_rate: f32,
    pub vib_cents: f32,
}

impl SungNote {
    pub fn new(midi: f32, detune_cents: f32, dur: f32) -> Self {
        Self { midi, detune_cents, dur, gap: 0.0, vib_rate: 5.5, vib_cents: 0.0 }
    }

    pub fn gap(mut self, g: f32) -> Self {
        self.gap = g;
        self
    }

    pub fn vibrato(mut self, rate: f32, cents: f32) -> Self {
        self.vib_rate = rate;
        self.vib_cents = cents;
        self
    }
}

#[derive(Clone, Debug)]
pub struct SingerSpec {
    pub sr: f32,
    pub notes: Vec<SungNote>,
    /// Scoop de ataque depois de silencio (semitons abaixo do alvo).
    pub scoop_st: f32,
    /// Flutuacao fina (cents RMS).
    pub fine_cents: f32,
    /// Formantes (Hz, largura de banda Hz).
    pub formants: [(f32, f32); 5],
    /// Ruido de aspiracao relativo.
    pub breath: f32,
    pub seed: u64,
}

impl SingerSpec {
    pub fn new(sr: f32, notes: Vec<SungNote>) -> Self {
        Self {
            sr,
            notes,
            scoop_st: 0.6,
            fine_cents: 3.0,
            // vogal "a" (voz masculina/feminina media)
            formants: [(800.0, 80.0), (1150.0, 90.0), (2900.0, 120.0), (3900.0, 130.0), (4950.0, 140.0)],
            breath: 0.01,
            seed: 7,
        }
    }
}

/// Trecho de nota no sinal gerado (amostras).
#[derive(Clone, Copy, Debug)]
pub struct NoteSpan {
    pub start: usize,
    pub end: usize,
    pub intended: f32,
    pub sung: f32,
    pub vib_cents: f32,
}

pub struct Rendered {
    pub audio: Vec<f32>,
    /// f0 verdadeiro por amostra (Hz, 0 = silencio).
    pub f0: Vec<f32>,
    /// Componente de vibrato verdadeira por amostra (semitons).
    pub vib: Vec<f32>,
    pub spans: Vec<NoteSpan>,
}

pub fn render(spec: &SingerSpec) -> Rendered {
    let sr = spec.sr;
    let dt = 1.0 / sr;
    let mut rng = Rng::new(spec.seed);
    let lead_in = (0.25 * sr) as usize;
    let total: f32 = spec.notes.iter().map(|n| n.dur + n.gap).sum::<f32>() + 0.6;
    let len = lead_in + (total * sr) as usize;
    let mut audio = vec![0.0f32; len];
    let mut f0v = vec![0.0f32; len];
    let mut vibv = vec![0.0f32; len];
    let mut spans = Vec::new();

    // dinamica de 2a ordem (Saitou): overshoot nas transicoes ligadas
    let (zeta, wn) = (0.55f32, std::f32::consts::TAU * 7.0);
    let mut x = 0.0f32;
    let mut v = 0.0f32;
    let mut phase = 0.0f64;
    let mut fine = 0.0f32;
    let mut fine_lp = Biquad::lowpass(sr as f64, 12.0, 0.707);
    let mut i = lead_in;
    let mut prev_gap = true;
    let nmax = 64usize;
    for note in &spec.notes {
        let target = note.midi + note.detune_cents / 100.0;
        let n_len = (note.dur * sr) as usize;
        if prev_gap {
            x = target - spec.scoop_st;
            v = 0.0;
        }
        spans.push(NoteSpan { start: i, end: i + n_len, intended: note.midi, sung: target, vib_cents: note.vib_cents });
        for j in 0..n_len {
            let tn = j as f32 * dt;
            // dinamica
            let acc = wn * wn * (target - x) - 2.0 * zeta * wn * v;
            v += acc * dt;
            x += v * dt;
            // vibrato com atraso e crescimento
            let env_v = ((tn - 0.25) / 0.35).clamp(0.0, 1.0);
            let vib = env_v * note.vib_cents / 100.0 * (std::f32::consts::TAU * note.vib_rate * tn).sin();
            // flutuacao fina
            fine = fine_lp.process(rng.gauss());
            let fine_st = fine * spec.fine_cents / 100.0 * 6.0;
            let m = x + vib + fine_st;
            let f0 = 440.0 * ((m - 69.0) / 12.0).exp2();
            if i >= len {
                break;
            }
            f0v[i] = f0;
            vibv[i] = vib;
            phase += std::f64::consts::TAU * f0 as f64 / sr as f64;
            // envelope de amplitude
            let att = (tn / 0.03).min(1.0);
            let rel = if note.gap > 0.0 { ((note.dur - tn) / 0.04).clamp(0.0, 1.0) } else { 1.0 };
            let amp = 0.25 * att * rel;
            let mut s = 0.0f32;
            for k in 1..=nmax {
                let fk = f0 * k as f32;
                if fk > 0.45 * sr || fk > 9000.0 {
                    break;
                }
                let mut env = 0.0f32;
                for &(ff, bw) in &spec.formants {
                    let r = fk / ff;
                    env += 1.0 / (((1.0 - r * r).powi(2) + (fk * bw / (ff * ff)).powi(2)).sqrt());
                }
                s += env / k as f32 * ((k as f64 * phase).sin() as f32);
            }
            audio[i] = amp * 0.05 * s + amp * spec.breath * rng.gauss();
            i += 1;
        }
        let gap = (note.gap * sr) as usize;
        i += gap;
        prev_gap = note.gap > 0.0;
    }
    let _ = fine;
    // normaliza o pico em -6 dBFS
    let peak = audio.iter().fold(0.0f32, |a, &s| a.max(s.abs())).max(1e-9);
    for s in &mut audio {
        *s *= 0.5 / peak;
    }
    Rendered { audio, f0: f0v, vib: vibv, spans }
}
