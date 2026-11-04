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

/// De onde os graos leem: a voz inteira ou so a parte harmonica (separada sob demanda).
pub trait GrainSource {
    /// Garante o sinal ate `upto` (exclusivo).
    fn prepare(&mut self, upto: i64);
    fn ring(&self) -> &Ring;
}

/// A propria entrada.
pub struct Plain<'a>(pub &'a Ring);

impl GrainSource for Plain<'_> {
    fn prepare(&mut self, _upto: i64) {}
    fn ring(&self) -> &Ring { self.0 }
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
    /// Origem - saida do grao atual (amostras): zero sem correcao.
    delta: f64,
    /// Lado direito do grao anterior: comprimento, descida e se foi encurtado pelo limite.
    prev_right: f64,
    prev_fall: f64,
    prev_capped: bool,
    too_low: bool,
    /// Distancia alem da saida ate onde os graos sao planejados (ao vivo).
    pub reach: f64,
    /// Ate onde a origem pode ser lida agora (ao vivo; infinito fora dele).
    pub read_limit: f64,
    /// Maior periodo corrigido: acima dele a voz passa sem correcao (ao vivo).
    pub max_period: f64,
    sr_per_s: f64,
}

/// Maior meia-janela de saida (amostras), usada para planejar os graos com antecedencia.
pub fn max_half_len(sr: f32) -> f64 { (sr / super::pitch::F0_MIN) as f64 / FORMANT_MIN as f64 + 4.0 }

/// Descida curta do grao no modo ao vivo (s).
pub const LIVE_FADE: f64 = 0.0015;

impl GrainVoice {
    pub fn with_rate(sr: f32) -> Self { Self { sr_per_s: sr as f64, ..Self::new() } }

    pub fn new() -> Self {
        Self {
            acc: Ring::new(1 << 15),
            next_center: 0.0,
            prev_center: 0.0,
            started: false,
            emitted: 0,
            starved: 0,
            in_voiced: false,
            delta: 0.0,
            prev_right: 0.0,
            prev_fall: 0.0,
            prev_capped: false,
            too_low: false,
            reach: f64::INFINITY,
            read_limit: f64::INFINITY,
            max_period: f64::INFINITY,
            sr_per_s: 48_000.0,
        }
    }

    /// Ha uma marca vozeada utilizavel perto de `c`? Devolve a marca, as meias janelas de
    /// analise para tras e para a frente (distancias reais ate as marcas vizinhas) e o passo ate
    /// a proxima marca.
    fn voiced_mark(&self, marks: &MarkTracker, c: f64) -> Option<(super::marks::Mark, f64, f64, f32)> {
        self.voiced_mark_k(marks, c).map(|(m, b, f, fwd, _)| (m, b, f, fwd))
    }

    /// Como `voiced_mark`, dizendo tambem se a marca seguinte ja existia (senao o passo e uma
    /// estimativa).
    fn voiced_mark_k(&self, marks: &MarkTracker, c: f64) -> Option<(super::marks::Mark, f64, f64, f32, bool)> {
        let (m, half, fwd, known) = marks.nearest_cycle(c);
        let back = 2.0 * half as f64 - fwd as f64;
        (m.voiced && (m.pos - c).abs() < 1.5 * m.period as f64).then_some((m, back, fwd as f64, fwd, known))
    }

    /// Planeja e soma no acumulador todos os graos com centro antes de `until`.
    ///
    /// Entre dois graos vozeados as meias janelas sao as distancias reais ate as marcas vizinhas
    /// (assimetricas quando o periodo muda, como no vibrato): sem correcao as metades de Hann de
    /// graos vizinhos se complementam e a reconstrucao e exata. Nas passagens entre trecho vozeado e sem vozeamento as meias janelas viram a distancia ate
    /// o grao vizinho, dos dois lados: as metades de Hann se complementam e a soma das janelas e
    /// exatamente 1 na fronteira (sem bolha nem buraco de nivel no inicio e no fim das notas).
    pub fn plan(&mut self, until: f64, input: &mut impl GrainSource, marks: &MarkTracker, ctl: &impl VoiceControl, hann: &HannTable) {
        let hop = marks.uv_hop as f64;
        if !self.started {
            self.next_center = self.emitted as f64;
            self.prev_center = self.next_center - hop;
            self.prev_right = hop;
            self.prev_fall = hop;
            self.started = true;
        }
        while self.next_center < until {
            let c = self.next_center;
            // leitura contínua: a origem acompanha a saida com um deslocamento que so muda pelo
            // que a correcao pede. Sem correcao ele e zero e a saida e a propria entrada, com
            // quaisquer marcas. Os graos ficam sincronos entre si porque andam um periodo real.
            let s = c + self.delta;
            let found = self.voiced_mark_k(marks, s);
            // voz grave demais para a latencia: passa limpa, sem correcao (com histerese, para nao
            // alternar a cada ciclo perto do limite)
            if let Some(v) = found {
                let p = v.3 as f64;
                if p > self.max_period {
                    self.too_low = true;
                } else if p < 0.92 * self.max_period {
                    self.too_low = false;
                }
            }
            if let Some((m, _, _, fwd, _)) = found.filter(|_| !self.too_low) {
                let st = ctl.state(c, fwd);
                let ratio = st.ratio.clamp(0.25, 4.0) as f64;
                let phi = st.formant.clamp(FORMANT_MIN, FORMANT_MAX) as f64;
                let period = fwd as f64;
                let ps = (period / ratio).max(8.0);
                let next = c + ps;
                let mut d = self.delta + period - ps;
                // com correcao, puxa os graos devagar para as marcas (pulso no centro da janela)
                if (ratio - 1.0).abs() > 1e-3 {
                    let e = m.pos - s;
                    if e.abs() < 0.5 * period {
                        d += 0.1 * e;
                    }
                }
                // passou de um periodo: o grao se repete ou se pula (e assim que a altura muda)
                let hi = if self.read_limit.is_finite() { 0.0 } else { 0.5 * period };
                if d > hi {
                    d -= period;
                } else if d < hi - period {
                    d += period;
                }
                let next_voiced = self.voiced_mark(marks, next + d).is_some();
                let want = if next_voiced { period / phi } else { ps };
                let (right, fall, capped) = self.right_side(s, ps, want, phi);
                let (left, taper) = self.rise(c);
                // normalizacao de potencia: densidade de graos e duracao do grao
                let norm = if capped || self.prev_capped { 1.0 } else { ((ps * phi) / period).sqrt().clamp(0.4, 2.0) as f32 };
                self.add_grain_t(c, s, left, taper, right, fall, phi, norm, input, hann);
                self.prev_capped = capped;
                self.prev_right = right;
                self.prev_fall = fall;
                self.in_voiced = true;
                self.delta = d;
                self.prev_center = c;
                self.next_center = next;
            } else {
                // fome = as marcas ainda nao chegaram ate aqui (uma marca vozeada longe e so uma
                // lacuna real, como um salto grande de registro)
                // (no fim de um trecho sem voz a grade de 5 ms pode ficar um pouco atras: ali o grao
                // le a propria entrada e nao precisa de marca)
                if marks.latest_pos() < c && marks.count() > 0 && marks.get(marks.count() - 1).voiced {
                    self.starved += 1;
                }
                let mut next = c + hop;
                // vai comecar um trecho vozeado: o proximo grao cai sobre a marca de analise
                if let Some((m, h_back, h_fwd, _)) = self.voiced_mark(marks, next) {
                    let h_src = 0.5 * (h_back + h_fwd);
                    let lead = m.pos - next;
                    if lead > -0.5 * h_src && lead <= h_src && m.pos > c + 8.0 {
                        next = m.pos;
                    }
                }
                let (left, taper) = self.rise(c);
                let (right, fall, capped) = self.right_side(c, next - c, next - c, 1.0);
                self.add_grain_t(c, c, left, taper, right, fall, 1.0, 1.0, input, hann);
                self.prev_capped = capped;
                self.prev_right = right;
                self.prev_fall = fall;
                self.in_voiced = false;
                self.delta = 0.0;
                self.prev_center = c;
                self.next_center = next;
            }
        }
    }

    /// Lado direito da janela de um grao que le a origem em `s`, com o proximo grao a `step` e o
    /// comprimento desejado `want`: (comprimento, descida, encurtado). Ao vivo a janela nao pode
    /// passar do sinal que ja existe, e o proximo grao so sera somado quando estiver a `reach` da
    /// saida: a subida dele tem que comecar depois disso, senao nasceria cortada (um degrau, que
    /// se ouve como chiado). Por isso, com o proximo grao longe, a passagem entre os dois acontece
    /// numa descida curta, mais adiante, e cada um fica em 1 ate ela.
    fn right_side(&self, s: f64, step: f64, want: f64, phi: f64) -> (f64, f64, bool) {
        if !self.read_limit.is_finite() {
            return (want, want, false);
        }
        let room = ((self.read_limit - s) / phi).max(8.0);
        let fade = LIVE_FADE * self.sr_per_s;
        if step > self.reach {
            let r = room.min(step).max(fade);
            (r, fade.min(r), true)
        } else {
            let r = want.min(room).min(self.reach).max(8.0);
            (r, r, false)
        }
    }

    /// Subida da janela do grao em `c`: complementa exatamente a descida do grao anterior.
    fn rise(&self, c: f64) -> (f64, f64) {
        if self.prev_capped {
            (c - (self.prev_center + self.prev_right - self.prev_fall), self.prev_fall)
        } else {
            (self.prev_right, self.prev_right)
        }
    }

    /// Soma um grao centrado em `c` (saida) lendo a origem em torno de `src`. A janela sobe em
    /// `taper` amostras a partir de `c - left`, fica em 1 e desce em `fall` amostras terminando em
    /// `c + right` (sem trecho plano quando `taper == left` e `fall == right`: Hann).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn add_grain_t(&mut self, c: f64, src: f64, left: f64, taper: f64, right: f64, fall: f64, phi: f64, g: f32, input: &mut impl GrainSource, hann: &HannTable) {
        let (left, right) = (left.max(1.0), right.max(1.0));
        let (taper, fall) = (taper.clamp(1.0, left), fall.clamp(1.0, right));
        input.prepare((src + right * phi).ceil() as i64 + 3);
        let input = input.ring();
        let n0 = ((c - left).ceil() as i64).max(self.emitted);
        let n1 = (c + right).floor() as i64;
        let (it, ir) = ((1.0 / taper) as f32, (1.0 / fall) as f32);
        let (flat_l, flat_r) = (left - taper, right - fall);
        for n in n0..=n1 {
            let dn = n as f64 - c;
            let w = if dn < 0.0 { hann.w(((dn + flat_l).min(0.0)) as f32 * it) } else { hann.w(((dn - flat_r).max(0.0)) as f32 * ir) };
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
