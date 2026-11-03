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
    /// Passo dado com o periodo estimado (marca, passo usado, razao), a acertar.
    guess: Option<(f64, f64, f64)>,
    /// A janela da direita do grao anterior (amostras) e se ela foi encurtada pelo limite.
    prev_right: f64,
    prev_capped: bool,
    /// Maior meia janela de analise (amostras): no modo ao vivo limita o futuro que cada grao
    /// precisa ler (vozes graves ganham graos mais curtos que o periodo).
    pub max_half: f64,
}

/// Maior meia-janela de saida (amostras), usada para planejar os graos com antecedencia.
pub fn max_half_len(sr: f32) -> f64 { (sr / super::pitch::F0_MIN) as f64 / FORMANT_MIN as f64 + 4.0 }

/// Meia janela maxima do grao no modo ao vivo (s).
pub const LIVE_MAX_HALF: f64 = 0.0035;

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
            guess: None,
            prev_right: 0.0,
            prev_capped: false,
            max_half: f64::INFINITY,
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
        (m.voiced && (m.pos - c).abs() < 1.5 * m.period as f64).then(|| (m, back.min(self.max_half), (fwd as f64).min(self.max_half), fwd, known))
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
            self.started = true;
        }
        while self.next_center < until {
            // o passo anterior saiu da ultima marca conhecida, com o periodo estimado: agora que a
            // marca seguinte existe, o erro da estimativa e devolvido (sem isso a sintese perde a
            // trava das marcas de analise e a voz inteira sai deslocada no tempo, o que acontece
            // muito com latencia curta no modo ao vivo)
            if let Some((mp, used, ratio)) = self.guess.take() {
                if let Some(actual) = marks.next_voiced_after(mp + 1.0) {
                    let err = (actual - mp) - used;
                    if err.abs() < 0.4 * used {
                        self.next_center = (self.next_center + err / ratio).max(self.prev_center + 8.0);
                    }
                }
            }
            let c = self.next_center;
            if let Some((m, h_back, h_fwd, fwd, known)) = self.voiced_mark_k(marks, c) {
                let h_src = 0.5 * (h_back + h_fwd);
                // as marcas do inicio da nota chegaram depois que o plano ja tinha passado da
                // borda (modo ao vivo, blocos grandes): um grao sem correcao faz a ponte ate a
                // proxima marca, para que os graos vozeados caiam sobre as marcas de analise.
                // Sem isso o trecho todo sairia deslocado no tempo ate a proxima pausa.
                if !self.in_voiced && (m.pos - c).abs() > 1.0 {
                    if let Some(t) = marks.next_voiced_after(c + 8.0).filter(|&t| t - c <= hop + 2.0 * h_src) {
                        self.add_grain(c, c, c - self.prev_center, t - c, 1.0, 1.0, input, hann);
                        self.prev_capped = false;
                        self.prev_right = t - c;
                        self.prev_center = c;
                        self.next_center = t;
                        continue;
                    }
                }
                let st = ctl.state(c, fwd);
                let ratio = st.ratio.clamp(0.25, 4.0) as f64;
                let phi = st.formant.clamp(FORMANT_MIN, FORMANT_MAX) as f64;
                let ps = (fwd as f64 / ratio).max(8.0);
                let next = c + ps;
                let next_voiced = self.voiced_mark(marks, next).is_some();
                let right = if next_voiced { h_fwd / phi } else { ps };
                // grao limitado (ao vivo, voz grave): a janela da direita do grao anterior acabou
                // antes deste centro. O lado do passado nao custa latencia, entao a janela deste
                // grao vai ate o grao anterior, sobe junto com a descida dele e fica em 1 ate o
                // centro: as janelas somam 1 e nao ha buracos (nem perda de energia).
                let (left, taper) = if self.in_voiced && self.prev_capped {
                    (c - self.prev_center, self.prev_right)
                } else if self.in_voiced {
                    (h_back / phi, h_back / phi)
                } else {
                    (c - self.prev_center, c - self.prev_center)
                };
                // normalizacao de potencia: densidade de graos e duracao do grao
                let norm = if self.prev_capped || h_fwd < fwd as f64 - 0.5 { 1.0 } else { ((ps * phi) / fwd as f64).sqrt().clamp(0.4, 2.0) as f32 };
                self.add_grain_t(c, m.pos, left, taper, right, phi, norm, input, hann);
                self.prev_capped = next_voiced && h_fwd < fwd as f64 - 0.5;
                self.prev_right = right;
                if !known {
                    self.guess = Some((m.pos, fwd as f64, ratio));
                }
                self.in_voiced = true;
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
                // (sem correcao, a saida e a propria entrada, ciclo por ciclo)
                if let Some((m, h_back, h_fwd, _)) = self.voiced_mark(marks, next) {
                    let h_src = 0.5 * (h_back + h_fwd);
                    let lead = m.pos - next;
                    if lead > -0.5 * h_src && lead <= h_src && m.pos > c + 8.0 {
                        next = m.pos;
                    }
                }
                let left = if self.in_voiced { c - self.prev_center } else { hop };
                self.add_grain(c, c, left, next - c, 1.0, 1.0, input, hann);
                self.prev_capped = false;
                self.prev_right = next - c;
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
    fn add_grain(&mut self, c: f64, src: f64, left: f64, right: f64, phi: f64, g: f32, input: &mut impl GrainSource, hann: &HannTable) {
        self.add_grain_t(c, src, left, left, right, phi, g, input, hann);
    }

    /// Como `add_grain`, com a subida da esquerda mais curta que a meia janela (`taper <= left`):
    /// a janela sobe em `taper` amostras a partir de `c - left` e fica em 1 ate o centro.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn add_grain_t(&mut self, c: f64, src: f64, left: f64, taper: f64, right: f64, phi: f64, g: f32, input: &mut impl GrainSource, hann: &HannTable) {
        let (left, right) = (left.max(1.0), right.max(1.0));
        let taper = taper.clamp(1.0, left);
        input.prepare((src + right * phi).ceil() as i64 + 3);
        let input = input.ring();
        let n0 = ((c - left).ceil() as i64).max(self.emitted);
        let n1 = (c + right).floor() as i64;
        let (it, ir) = ((1.0 / taper) as f32, (1.0 / right) as f32);
        // a subida termina em `c - left + taper`; dali ate o centro a janela e 1
        let flat = left - taper;
        for n in n0..=n1 {
            let dn = n as f64 - c;
            let w = if dn < 0.0 { hann.w(((dn + flat).min(0.0)) as f32 * it) } else { hann.w(dn as f32 * ir) };
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
