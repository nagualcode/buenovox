//! Shifter PSOLA (Pitch-Synchronous Overlap-Add) da voz corrigida.
//!
//! Herdado do Heaven, com uma diferenca que importa para a naturalidade: o periodo de sintese
//! nao e o periodo absoluto da nota alvo, e o espacamento REAL entre as marcas de analise
//! dividido pela razao de correcao. Assim o jitter ciclo a ciclo do cantor (a micro-irregularidade
//! que faz uma voz soar humana) passa para a saida, e so a altura media muda.
//!
//! Cada grao e um ciclo glotal (janela de Hann de 2 periodos centrada numa marca de epoca). Os
//! graos sao recolocados no novo periodo: a altura muda e o envelope espectral (formantes) fica,
//! porque cada grao carrega a resposta do trato vocal. Ler o grao mais rapido ou mais devagar
//! (`formant`) encurta ou alonga a "garganta".
//!
//! Trechos sem vozeamento (consoantes, respiracao) passam sem mudanca de altura, em graos de
//! 5 ms sobre a grade original: a reconstrucao e exata.

use super::marks::MarkTracker;
use super::util::{HannTable, Ring};

/// O que a voz deve fazer num instante de origem.
#[derive(Clone, Copy, Debug)]
pub struct VoiceState {
    /// Razao de frequencia (saida / entrada). 1 = sem correcao.
    pub ratio: f32,
    /// Leitura do grao: 1 = formantes preservadas, >1 = garganta curta, <1 = garganta longa.
    pub formant: f32,
}

impl Default for VoiceState {
    fn default() -> Self { Self { ratio: 1.0, formant: 1.0 } }
}

pub trait VoiceControl {
    /// Estado da voz no instante de origem `s` (o material que o grao vai ler).
    fn state(&self, s: f64) -> VoiceState;
}

/// Limites da leitura do grao (garganta).
pub const FORMANT_MIN: f32 = 0.75;
pub const FORMANT_MAX: f32 = 1.35;

pub struct GrainVoice {
    acc: Ring,
    next_center: f64,
    started: bool,
    /// Ultima posicao ja entregue (nada pode ser somado antes disso).
    emitted: i64,
    /// Graos vozeados planejados sem marca proxima (o motor esta "com fome": latencia curta demais).
    pub starved: u64,
    in_voiced: bool,
    pub dbg_onsets: u64,
    pub dbg_snaps: u64,
    pub dbg_unknown: u64,
}

/// Maior meia-janela de saida (amostras), usada para planejar os graos com antecedencia.
pub fn max_half_len(sr: f32) -> f64 { (sr / super::pitch::F0_MIN) as f64 / FORMANT_MIN as f64 + 4.0 }

impl GrainVoice {
    pub fn new() -> Self { Self { acc: Ring::new(1 << 15), next_center: 0.0, started: false, emitted: 0, starved: 0, in_voiced: false, dbg_onsets: 0, dbg_snaps: 0, dbg_unknown: 0 } }

    /// Planeja e soma no acumulador todos os graos com centro antes de `until`.
    pub fn plan(&mut self, until: f64, input: &Ring, marks: &MarkTracker, ctl: &impl VoiceControl, hann: &HannTable) {
        if !self.started {
            self.next_center = self.emitted as f64;
            self.started = true;
        }
        while self.next_center < until {
            let mut c = self.next_center;
            let (m, half, fwd, known) = marks.nearest_cycle(c);
            if m.voiced && (m.pos - c).abs() < 1.5 * m.period as f64 {
                let st = ctl.state(c);
                let ratio = st.ratio.clamp(0.25, 4.0) as f64;
                let phi = st.formant.clamp(FORMANT_MIN, FORMANT_MAX) as f64;
                let h_src = half as f64;
                // inicio de trecho vozeado: a primeira marca de sintese cai sobre a de analise
                // (sem correcao, a saida e a propria entrada, ciclo por ciclo)
                // (para frente ate uma meia janela: o grao vozeado cobre o fim do ultimo grao sem
                // vozeamento, sem buraco; para tras so um pouco, para nao somar demais)
                let lead = m.pos - c;
                if !self.in_voiced && m.pos - h_src / phi >= self.emitted as f64 && lead > -0.5 * h_src && lead <= h_src / phi {
                    c = m.pos;
                }
                if !self.in_voiced {
                    self.dbg_onsets += 1;
                    if c == m.pos {
                        self.dbg_snaps += 1;
                    }
                }
                if !known {
                    self.dbg_unknown += 1;
                }
                self.in_voiced = true;
                let ps = (fwd as f64 / ratio).max(8.0);
                // normalizacao de potencia: densidade de graos e duracao do grao
                let norm = ((ps * phi) / fwd as f64).sqrt().clamp(0.4, 2.0) as f32;
                self.add_grain(c, m.pos, h_src, phi, norm, input, hann);
                self.next_center = c + ps;
            } else {
                // fome = as marcas ainda nao chegaram ate aqui (uma marca vozeada longe e so uma
                // lacuna real, como um salto grande de registro)
                if marks.latest_pos() < c {
                    self.starved += 1;
                }
                self.in_voiced = false;
                let hop = marks.uv_hop as f64;
                self.add_grain(c, c, hop, 1.0, 1.0, input, hann);
                self.next_center = c + hop;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn add_grain(&mut self, c: f64, src: f64, h_src: f64, phi: f64, g: f32, input: &Ring, hann: &HannTable) {
        let h_out = h_src / phi;
        let n0 = ((c - h_out).ceil() as i64).max(self.emitted);
        let n1 = (c + h_out).floor() as i64;
        let inv = (1.0 / h_out) as f32;
        for n in n0..=n1 {
            let dn = n as f64 - c;
            let w = hann.w(dn as f32 * inv);
            let x = input.interp(src + dn * phi);
            self.acc.add(n, g * w * x);
        }
    }

    /// Entrega a amostra de saida `n`.
    #[inline]
    pub fn take(&mut self, n: i64) -> f32 {
        self.emitted = n + 1;
        self.acc.take(n)
    }
}

impl Default for GrainVoice {
    fn default() -> Self { Self::new() }
}
