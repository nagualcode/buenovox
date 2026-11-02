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
    /// Estado da voz no instante de origem `s` (o material que o grao vai ler). `src_period` e
    /// o periodo real do ciclo naquele ponto (amostras), para alvos absolutos (Hard).
    fn state(&self, s: f64, src_period: f32) -> VoiceState;
}

/// Limites da leitura do grao (garganta).
pub const FORMANT_MIN: f32 = 0.75;
pub const FORMANT_MAX: f32 = 1.35;

pub struct GrainVoice {
    acc: Ring,
    next_center: f64,
    prev_center: f64,
    started: bool,
    /// Ultima posicao ja entregue (nada pode ser somado antes disso).
    emitted: i64,
    /// Graos planejados alem das marcas ja calculadas (o motor esta "com fome": latencia curta
    /// demais).
    pub starved: u64,
    in_voiced: bool,
    /// Maior meia janela de analise (amostras): no modo ao vivo limita o futuro que cada grao
    /// precisa ler (vozes graves ganham graos mais curtos que o periodo).
    pub max_half: f64,
}

/// Maior meia-janela de saida (amostras), usada para planejar os graos com antecedencia.
pub fn max_half_len(sr: f32) -> f64 { (sr / super::pitch::F0_MIN) as f64 / FORMANT_MIN as f64 + 4.0 }

/// Meia janela maxima do grao no modo ao vivo (s).
pub const LIVE_MAX_HALF: f64 = 0.005;

impl GrainVoice {
    pub fn new() -> Self {
        Self {
            acc: Ring::new(1 << 15),
            next_center: 0.0,
            prev_center: 0.0,
            started: false,
            emitted: 0,
            starved: 0,
            in_voiced: false,
            max_half: f64::INFINITY,
        }
    }

    /// Ha uma marca vozeada utilizavel perto de `c`? Devolve a marca e a meia janela de analise.
    fn voiced_mark(&self, marks: &MarkTracker, c: f64) -> Option<(super::marks::Mark, f64, f32)> {
        let (m, half, fwd, _) = marks.nearest_cycle(c);
        (m.voiced && (m.pos - c).abs() < 1.5 * m.period as f64).then(|| (m, (half as f64).min(self.max_half), fwd))
    }

    /// Planeja e soma no acumulador todos os graos com centro antes de `until`.
    ///
    /// Entre dois graos vozeados as janelas sao as do PSOLA (meia janela = periodo de analise).
    /// Nas passagens entre trecho vozeado e sem vozeamento as meias janelas viram a distancia ate
    /// o grao vizinho, dos dois lados: as metades de Hann se complementam e a soma das janelas e
    /// exatamente 1 na fronteira (sem bolha nem buraco de nivel no inicio e no fim das notas).
    pub fn plan(&mut self, until: f64, input: &Ring, marks: &MarkTracker, ctl: &impl VoiceControl, hann: &HannTable) {
        let hop = marks.uv_hop as f64;
        if !self.started {
            self.next_center = self.emitted as f64;
            self.prev_center = self.next_center - hop;
            self.started = true;
        }
        while self.next_center < until {
            let c = self.next_center;
            if let Some((m, h_src, fwd)) = self.voiced_mark(marks, c) {
                let st = ctl.state(c, fwd);
                let ratio = st.ratio.clamp(0.25, 4.0) as f64;
                let phi = st.formant.clamp(FORMANT_MIN, FORMANT_MAX) as f64;
                let h_out = h_src / phi;
                let ps = (fwd as f64 / ratio).max(8.0);
                let next = c + ps;
                let left = if self.in_voiced { h_out } else { c - self.prev_center };
                let right = if self.voiced_mark(marks, next).is_some() { h_out } else { ps };
                // normalizacao de potencia: densidade de graos e duracao do grao
                let norm = ((ps * phi) / fwd as f64).sqrt().clamp(0.4, 2.0) as f32;
                self.add_grain(c, m.pos, left, right, phi, norm, input, hann);
                self.in_voiced = true;
                self.prev_center = c;
                self.next_center = next;
            } else {
                // fome = as marcas ainda nao chegaram ate aqui (uma marca vozeada longe e so uma
                // lacuna real, como um salto grande de registro)
                if marks.latest_pos() < c {
                    self.starved += 1;
                }
                let mut next = c + hop;
                // vai comecar um trecho vozeado: o proximo grao cai sobre a marca de analise
                // (sem correcao, a saida e a propria entrada, ciclo por ciclo)
                if let Some((m, h_src, _)) = self.voiced_mark(marks, next) {
                    let lead = m.pos - next;
                    if lead > -0.5 * h_src && lead <= h_src && m.pos > c + 8.0 {
                        next = m.pos;
                    }
                }
                let left = if self.in_voiced { c - self.prev_center } else { hop };
                self.add_grain(c, c, left, next - c, 1.0, 1.0, input, hann);
                self.in_voiced = false;
                self.prev_center = c;
                self.next_center = next;
            }
        }
    }

    /// Soma um grao centrado em `c` (saida) lendo a origem em torno de `src`, com meias janelas
    /// de saida `left` e `right` (amostras).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn add_grain(&mut self, c: f64, src: f64, left: f64, right: f64, phi: f64, g: f32, input: &Ring, hann: &HannTable) {
        let (left, right) = (left.max(1.0), right.max(1.0));
        let n0 = ((c - left).ceil() as i64).max(self.emitted);
        let n1 = (c + right).floor() as i64;
        let (il, ir) = ((1.0 / left) as f32, (1.0 / right) as f32);
        for n in n0..=n1 {
            let dn = n as f64 - c;
            let w = hann.w(dn as f32 * if dn < 0.0 { il } else { ir });
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
