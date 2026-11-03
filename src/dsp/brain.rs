//! O cerebro do BuenoVOX: decide, quadro a quadro, para que altura a voz deve ir.
//!
//! O principio (PLANO.md, secao 0): corrigir so o centro percebido de cada nota e devolver todo o
//! resto intacto. Para cada quadro de controle (4 ms) o cerebro:
//!
//! 1. segmenta a voz em notas por MOVIMENTO, nao por arredondamento: uma nota nova comeca depois
//!    de silencio ou quando a voz sai da ancora da nota atual por mais que um limiar que cresce
//!    com o vibrato medido. (Arredondar para o semitom mais proximo nao serve num afinador: o
//!    cantor esta fora do centro por definicao, e os picos do vibrato cruzariam a fronteira.)
//! 2. mede o **centro local** `c` (media com janela de Hann de ~360 ms, restrita a nota atual,
//!    que remove o vibrato) e o **centro da nota** `C` (media acumulada de `c` nos quadros
//!    estaveis: a altura percebida, como o *stationary pitch* do BERT-APC);
//! 3. escolhe a nota alvo `T` (MIDI, escala ou acorde MIDI) a partir de `C`;
//! 4. decompoe a curva cantada `p` em: `C` (centro) + `drift = c - C` (deriva lenta) +
//!    `vib = p - c` (vibrato e flutuacao fina);
//! 5. aplica a politica:
//!    * **Natural**: `y = p + amount*(T - C) - tighten*drift + (vibrato-1)*vib + vibrato criado`.
//!      Com os valores padrao isso e um deslocamento uniforme da nota inteira: vibrato, deriva,
//!      ataque e micro-oscilacoes passam intactos. Entre notas ligadas o deslocamento migra da
//!      nota velha para a nova na proporcao do glide do proprio cantor (o overshoot e a
//!      preparacao de Saitou ficam como ele cantou). Scoops de ataque podem ser encurtados.
//!    * **Classic**: quantizacao do f0 instantaneo com retune de 2a ordem, Flex e Humanize.
//!    * **Hard**: retune instantaneo, o efeito.
//!
//! A saida e o deslocamento em semitons `y - p`; o PSOLA o aplica como razao sobre o periodo
//! real de cada ciclo.

use super::pitch::{PitchFrame, PitchTracker};
use super::tonal::{choose_target, Ctx};
use super::util::{smoothstep, Drift, Rng};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Style {
    #[default]
    Natural,
    Classic,
    Hard,
}

impl Style {
    pub fn from_index(i: usize) -> Self {
        match i {
            1 => Self::Classic,
            2 => Self::Hard,
            _ => Self::Natural,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrainParams {
    pub style: Style,
    /// Fracao do erro de afinacao corrigida, 0..1.
    pub amount: f32,
    /// Natural: suavizacao do deslocamento da nota. Classic: tempo de retune. Em ms.
    pub speed_ms: f32,
    /// Classic: so corrige perto da nota (0 = sempre, 1 = so muito perto).
    pub flex: f32,
    /// Classic: notas longas com retune mais lento.
    pub humanize: f32,
    /// Natural: quanto da deriva lenta dentro da nota e achatada, 0..1.
    pub tighten: f32,
    /// Escala do vibrato do cantor, 0..2 (1 = preserva).
    pub vibrato: f32,
    /// Vibrato criado em notas retas, 0..1.
    pub vib_create: f32,
    /// Encurta scoops e quedas de ataque, 0..1.
    pub scoop: f32,
    /// Referencia de afinacao (Hz).
    pub a4: f32,
}

impl Default for BrainParams {
    fn default() -> Self {
        Self {
            style: Style::Natural,
            amount: 1.0,
            speed_ms: 25.0,
            flex: 0.0,
            humanize: 0.3,
            tighten: 0.2,
            vibrato: 1.0,
            vib_create: 0.0,
            scoop: 0.3,
            a4: 440.0,
        }
    }
}

/// Decisao do cerebro num quadro de controle.
#[derive(Clone, Copy, Debug)]
pub struct CtrlFrame {
    /// Deslocamento a aplicar (semitons).
    pub shift: f32,
    /// Altura da entrada (MIDI afinado; 0 = sem voz).
    pub input: f32,
    /// Altura de saida pretendida (0 = sem voz).
    pub output: f32,
    /// Nota alvo com ajuste fino (NaN = sem alvo).
    pub target: f32,
    /// Extensao do vibrato do cantor (cents RMS).
    pub vib_cents: f32,
    /// Alvo absoluto (Hz) do estilo Hard; 0 = deslocamento relativo. O PSOLA divide pelo
    /// periodo real de cada ciclo: a escada do Hard nao depende do atraso do detector.
    pub hard_hz: f32,
    /// Diagnostico: centro da nota, centro local, ancora.
    pub dbg: [f32; 3],
}

impl Default for CtrlFrame {
    fn default() -> Self { Self { shift: 0.0, input: 0.0, output: 0.0, target: f32::NAN, vib_cents: 0.0, hard_hz: 0.0, dbg: [0.0; 3] } }
}

/// Duracao da janela do centro local (s): ~2 ciclos de vibrato.
const CENTER_WIN: f64 = 0.36;
/// Janela do centro curto (s): ~1 ciclo de vibrato.
const SHORT_WIN: f64 = 0.18;
// Sem lookahead (ao vivo) as decisoes de troca de nota so olham o passado: janelas e transicoes
// mais curtas (medido no vocadito com as notas anotadas como alvo: erro mediano 16 -> 12 cents).
const LIVE_SHORT_WIN: f64 = 0.06;
const LIVE_TRANS: f32 = 0.05;
const LIVE_AWAY: f32 = 0.05;
const LIVE_HOLD: f32 = 0.12;
/// Duracao tipica do ataque de uma nota (s).
const ATTACK: f64 = 0.08;
/// Antecipacao/atraso tolerados nas trocas de nota MIDI e acorde (s).
const MIDI_SLACK: f64 = 0.06;

pub struct Brain {
    sr: f32,
    hop: f64,
    dt: f32,
    /// Lookahead disponivel (amostras).
    la: f64,
    /// Tempo segurando o alvo antigo depois de uma troca no meio da nota (s).
    target_hold: f32,
    /// Horizonte da "nota em que o cantor vai chegar" (amostras).
    fut: f64,

    /// Ancora da nota atual (MIDI afinado continuo); None = sem nota.
    anchor: Option<f32>,
    seg_start: f64,
    target: Option<i32>,
    seg_sum: f64,
    seg_w: f64,
    time_in_note: f32,
    unvoiced_time: f32,
    stability: f32,
    away_time: f32,
    octave_time: f32,
    onset_from_silence: bool,
    last_p: f32,
    last_shift: f32,

    // deslocamento da nota (Natural)
    o: f32,
    o1: f32,
    trans: bool,
    /// Transicao detectada pelo centro curto (passo pequeno): o progresso segue o centro, nao o
    /// f0 cru (que tem vibrato do tamanho do passo).
    trans_by_center: bool,
    trans_time: f32,
    p0: f32,
    o_a: f32,

    // retune (Classic/Hard)
    cl1: f32,
    cl2: f32,
    /// Amortecimento do retune do Classic (do ultimo intervalo).
    cl_zeta: f32,

    // vibrato
    vib_ms: f32,
    vib_phase: f32,
    rate_drift: Drift,
    rng: Rng,
}

impl Brain {
    pub fn new(sr: f32, hop: usize, lookahead: f64) -> Self {
        Self {
            sr,
            hop: hop as f64,
            dt: hop as f32 / sr,
            la: lookahead,
            target_hold: 0.0,
            fut: lookahead.min(0.03 * sr as f64),
            anchor: None,
            seg_start: 0.0,
            target: None,
            seg_sum: 0.0,
            seg_w: 0.0,
            time_in_note: 0.0,
            unvoiced_time: 1.0,
            stability: 0.0,
            away_time: 0.0,
            octave_time: 0.0,
            onset_from_silence: true,
            last_p: 60.0,
            last_shift: 0.0,
            o: 0.0,
            o1: 0.0,
            trans: false,
            trans_by_center: false,
            trans_time: 0.0,
            p0: 0.0,
            o_a: 0.0,
            cl1: 0.0,
            cl2: 0.0,
            cl_zeta: 0.55,
            vib_ms: 0.0,
            vib_phase: 0.0,
            rate_drift: Drift::new(1.0, 0.8),
            rng: Rng::new(0x42_56_4F_58),
        }
    }

    #[inline]
    fn midi(f0: f32, a4: f32) -> f32 { 69.0 + 12.0 * (f0 / a4).log2() }

    #[inline]
    fn voiced(fr: &PitchFrame) -> bool { fr.voiced() && fr.conf > 0.02 }

    /// Limiar de saida da nota (semitons): cresce com o vibrato medido, para os picos do
    /// vibrato nao parecerem uma nota nova.
    /// Sem lookahead (modo ao vivo): as decisoes so podem olhar o passado.
    fn live(&self) -> bool { self.la < 0.02 * self.sr as f64 }

    fn leave_threshold(&self) -> f32 { (0.6 + 1.3 * self.vib_ms.sqrt()).min(1.4) }

    /// Mediana das alturas vozeadas nos quadros `j0..=j1` (em hops a partir de `t`), se houver
    /// quadros vozeados suficientes e eles concordarem (dispersao <= `spread`).
    fn median_pitch(&self, t: f64, pt: &PitchTracker, a4: f32, j0: i32, j1: i32, spread: f32) -> Option<f32> {
        let mut vals = [0.0f32; 32];
        let mut c = 0;
        for j in j0..=j1 {
            let fr = pt.at(t + j as f64 * self.hop);
            if Self::voiced(&fr) && c < vals.len() {
                vals[c] = Self::midi(fr.f0, a4);
                c += 1;
            }
        }
        let total = (j1 - j0 + 1).max(1) as usize;
        if c < 3 || c * 5 < total * 4 {
            return None;
        }
        let v = &mut vals[..c];
        v.sort_by(f32::total_cmp);
        (v[c - 1] - v[0] <= spread).then(|| v[c / 2])
    }

    /// Altura em que a voz vai estar nos ultimos ~24 ms do horizonte (onde o cantor chega).
    fn landing(&self, t: f64, pt: &PitchTracker, a4: f32) -> Option<f32> {
        let n = (self.fut / self.hop).floor() as i32;
        self.median_pitch(t, pt, a4, (n - 6).max(-6), n, 1.0)
    }

    /// Centro curto: media simples da altura numa janela de ~1 ciclo de vibrato (180 ms) em
    /// torno de `tc`, sem passar do futuro disponivel (`t + la`). Uma media de um ciclo inteiro
    /// cancela o vibrato (resta ~15% da extensao entre 5 e 6.5 Hz) e responde rapido a um passo
    /// de nota: o meio da subida do centro cai no instante do passo.
    fn short_center(&self, t: f64, tc: f64, pt: &PitchTracker, a4: f32) -> Option<f32> {
        let win = if self.live() { LIVE_SHORT_WIN } else { SHORT_WIN };
        let half = (win * 0.5 * self.sr as f64 / self.hop).floor() as i32;
        let limit = t + self.la;
        let (mut s, mut n) = (0.0f32, 0u32);
        for j in -half..=half {
            let tj = tc + j as f64 * self.hop;
            if tj > limit {
                break;
            }
            let fr = pt.at(tj);
            if Self::voiced(&fr) {
                s += Self::midi(fr.f0, a4);
                n += 1;
            }
        }
        (n as i32 > half).then(|| s / n as f32)
    }

    /// Centro local: media ponderada (Hann) da altura na janela em torno de `t`, so com quadros
    /// desta nota (perto da ancora). Usa ate 180 ms de futuro quando o lookahead permite.
    fn local_center(&self, t: f64, pt: &PitchTracker, anchor: f32, a4: f32) -> Option<f32> {
        let win = CENTER_WIN * self.sr as f64;
        let wf = self.la.min(0.5 * win).max(self.hop);
        let wb = (win - wf).max(0.12 * self.sr as f64);
        let nb = (wb / self.hop).floor() as i32;
        let nf = (wf / self.hop).floor() as i32;
        let reach = self.leave_threshold() + 0.4;
        let (mut s, mut w) = (0.0f64, 0.0f64);
        for j in -nb..=nf {
            let tj = t + j as f64 * self.hop;
            if tj < self.seg_start {
                continue;
            }
            let fr = pt.at(tj);
            if !Self::voiced(&fr) {
                continue;
            }
            let m = Self::midi(fr.f0, a4);
            if (m - anchor).abs() > reach {
                continue;
            }
            let u = if j < 0 { j as f64 * self.hop / wb } else { j as f64 * self.hop / wf };
            // o ataque (scoop, overshoot) pesa pouco: a altura percebida e a do corpo da nota
            let age = ((tj - self.seg_start) / (ATTACK * self.sr as f64)) as f32;
            let wj = (0.5 + 0.5 * (std::f64::consts::PI * u).cos()) * (0.2 + fr.conf as f64) * (0.05 + 0.95 * smoothstep(age)) as f64;
            s += m as f64 * wj;
            w += wj;
        }
        (w > 1e-3).then(|| (s / w) as f32)
    }

    fn start_segment(&mut self, t: f64, anchor: f32) {
        self.anchor = Some(anchor);
        self.seg_start = t;
        self.seg_sum = 0.0;
        self.seg_w = 0.0;
        self.time_in_note = 0.0;
        self.stability = 0.0;
        self.away_time = 0.0;
        self.target = None;
        self.target_hold = 0.0;
        self.vib_phase = 0.0;
    }

    /// Decide o quadro de controle do instante `t` (amostra de origem). `ctx_at` devolve o
    /// contexto tonal (parametros + MIDI) valido num instante.
    pub fn frame(&mut self, t: f64, pt: &PitchTracker, ctx_at: &dyn Fn(f64) -> Ctx, p: &BrainParams) -> CtrlFrame {
        let dt = self.dt;
        let a4 = p.a4;
        let fr = pt.at(t);
        let voiced = Self::voiced(&fr);
        let mut pm = if voiced { Self::midi(fr.f0, a4) } else { self.last_p };

        // Guarda de oitava (Heaven): excursao de ~12 semitons em relacao a nota atual e quase
        // sempre erro de deteccao. So vira salto real se persistir 120 ms.
        let mut octave_suspect = false;
        if let (true, Some(an), true) = (voiced, self.anchor, self.time_in_note > 0.08) {
            let d = pm - an;
            let folded = pm - 12.0 * (d / 12.0).round();
            if d.abs() > 9.0 && (folded - an).abs() < 2.5 {
                self.octave_time += dt;
                if self.octave_time < 0.12 {
                    pm = folded;
                    octave_suspect = true;
                }
            } else {
                self.octave_time = 0.0;
            }
        }

        // ------------------------------------------------------------ segmentacao
        if voiced {
            let gap = self.unvoiced_time;
            self.unvoiced_time = 0.0;
            let thr = self.leave_threshold();
            // a voz volta de uma pausa curta (respiracao, consoante) ja longe da nota anterior:
            // e outra nota, desde o primeiro quadro (sem isso o primeiro quadro herdaria o alvo
            // da nota anterior e daria um pulo de altura de alguns milissegundos)
            if gap >= 0.012 && self.anchor.is_some_and(|an| (pm - an).abs() > thr) && !octave_suspect {
                self.anchor = None;
            }
            match self.anchor {
                None => {
                    // inicio de frase: a altura em que a voz assenta nos proximos ~100 ms
                    let n = (self.la.min(0.1 * self.sr as f64) / self.hop).floor() as i32;
                    let start = self.median_pitch(t, pt, a4, 0, n.min(31), 24.0).unwrap_or(pm);
                    self.onset_from_silence = true;
                    self.trans = false;
                    self.cl1 = 0.0;
                    self.cl2 = 0.0;
                    self.start_segment(t, start);
                }
                Some(an) => {
                    let land = self.landing(t, pt, a4).filter(|&l| {
                        let octave = ((l - an).abs() - 12.0).abs() < 1.0;
                        !(octave_suspect || (octave && self.octave_time < 0.12))
                    });
                    let mut new_anchor = None;
                    let mut by_center = false;
                    if let Some(l) = land {
                        // salto: a voz vai pousar longe da ancora e ja esta indo para la
                        let d = pm - an;
                        let toward = d * (l - an) > 0.0;
                        if (l - an).abs() > thr && ((toward && d.abs() > 0.3) || (pm - l).abs() < 0.5) {
                            new_anchor = Some(l);
                        }
                    }
                    if new_anchor.is_none() && !octave_suspect && self.time_in_note > 0.06 {
                        // passo pequeno (meio tom ligado, mesmo com vibrato): o centro curto a
                        // frente mostra o novo patamar; troca quando o centro curto de agora passa
                        // da metade do caminho (o instante do passo)
                        let ahead = (self.la - 0.5 * SHORT_WIN * self.sr as f64).clamp(0.0, 0.5 * SHORT_WIN * self.sr as f64);
                        if let (Some(c_now), Some(c_ahead)) = (self.short_center(t, t, pt, a4), self.short_center(t, t + ahead, pt, a4)) {
                            let step = c_ahead - an;
                            let thr_s = 0.3 + 0.5 * self.vib_ms.sqrt();
                            if step.abs() > thr_s && step.abs() < 12.0 && (c_now - an) / step > 0.5 {
                                new_anchor = Some(c_ahead);
                                by_center = true;
                                self.p0 = an;
                            }
                        }
                    }
                    if new_anchor.is_none() {
                        // deriva longa para longe (passagens rapidas, sem pouso claro a frente)
                        if (pm - an).abs() > thr + 0.2 {
                            self.away_time += dt;
                            if self.away_time > if self.live() { LIVE_AWAY } else { 0.1 } {
                                new_anchor = Some(pm);
                            }
                        } else {
                            self.away_time = 0.0;
                        }
                    }
                    if let Some(na) = new_anchor {
                        self.onset_from_silence = false;
                        self.trans = true;
                        self.trans_by_center = by_center;
                        self.trans_time = 0.0;
                        if !by_center {
                            self.p0 = pm;
                        }
                        self.o_a = self.o;
                        self.start_segment(t, na);
                    }
                }
            }
        } else {
            self.unvoiced_time += dt;
            if self.unvoiced_time > 0.1 && self.anchor.is_some() {
                self.anchor = None;
                self.trans = false;
            }
        }

        let ctx_now = ctx_at(t);
        let Some(anchor) = self.anchor.filter(|_| voiced) else {
            // sem voz: o PSOLA nao desloca trechos sem vozeamento; caudas fracas seguem o
            // ultimo deslocamento
            return CtrlFrame { shift: self.last_shift, ..Default::default() };
        };
        self.last_p = pm;
        self.time_in_note += dt;

        // ------------------------------------------------------------ centros
        let c_loc = self.local_center(t, pt, anchor, a4).unwrap_or(pm);
        let after_attack = (t - self.seg_start) > ATTACK * self.sr as f64;
        let near = (c_loc - anchor).abs() < 0.8;
        if fr.conf > 0.1 && near && after_attack {
            let w = fr.conf.max(0.2) as f64;
            self.seg_sum += c_loc as f64 * w;
            self.seg_w += w;
        }
        // no comeco da nota o centro local (que ja olha o futuro) pesa mais que o acumulado
        let k = 10.0;
        let cn = ((self.seg_sum + c_loc as f64 * k) / (self.seg_w + k)) as f32;
        // a ancora acompanha o centro medido (devagar o bastante para nao seguir o vibrato)
        if after_attack {
            self.anchor = Some(anchor + (cn - anchor) * (1.0 - (-dt / 0.15).exp()));
        }

        let vib = pm - c_loc;
        self.vib_ms += (vib * vib - self.vib_ms) * (1.0 - (-dt / 0.3).exp());
        let st_target = if near && self.time_in_note > 0.08 && !self.trans { 1.0 } else { 0.0 };
        self.stability += (st_target - self.stability) * (1.0 - (-dt / 0.04).exp());
        let s = self.stability;

        // vibrato criado: so em notas retas e longas, entrando devagar
        let created = if p.vib_create > 0.0 {
            let rate = 5.5 + 0.35 * self.rate_drift.step(dt, &mut self.rng);
            self.vib_phase = (self.vib_phase + std::f32::consts::TAU * rate * dt) % std::f32::consts::TAU;
            let flat = (1.0 - self.vib_ms.sqrt() / 0.12).clamp(0.0, 1.0);
            let env = smoothstep((self.time_in_note - 0.25) / 0.5);
            p.vib_create * 0.5 * env * flat * self.vib_phase.sin()
        } else {
            0.0
        };
        let vib_mod = (p.vibrato - 1.0) * vib + created;

        // ------------------------------------------------------------ alvo e politica
        let slack = (MIDI_SLACK * self.sr as f64).min(self.la);
        let ctxs = [ctx_at(t - slack), ctx_now, ctx_at(t + slack)];

        // uma so decisao de nota para todos os estilos: o centro da nota (ataque com peso baixo,
        // lookahead), com histerese. Os estilos diferem so em COMO a voz chega ao alvo.
        let mut tn = choose_target(cn, &ctxs, self.target, 0.12);
        // o alvo mudou no meio da nota (MIDI trocou antes do cantor): enquanto o cantor ainda
        // estiver mais perto do alvo antigo, ele vale por ate `hold` (o tempo de a segmentacao
        // perceber a troca). Sem isso o novo intervalo seria aplicado sobre a nota antiga.
        if let (Some(old), Some(new)) = (self.target, tn) {
            let hold = if self.live() { LIVE_HOLD } else { 0.0 };
            let closer_old = (old as f32 - cn).abs() + 0.5 < (new as f32 - cn).abs();
            if new != old && closer_old && self.target_hold < hold {
                self.target_hold += dt;
                tn = Some(old);
            } else if new == old {
                self.target_hold = 0.0;
            }
        }
        if let (Some(a), Some(b)) = (self.target, tn) {
            if a != b {
                // amortecimento medido por intervalo (train/transitions.py): passos pequenos
                // passam mais do alvo, subindo mais que descendo
                let iv = b as f32 - a as f32;
                self.cl_zeta = if iv.abs() < 2.5 { if iv > 0.0 { 0.47 } else { 0.52 } } else { 0.63 };
            }
        }
        self.target = tn;
        let Some(tn) = tn else {
            return self.pass(pm);
        };
        let tf = tn as f32 + ctx_now.fine(tn);

        let y = match p.style {
            Style::Natural => {
                let o_target = tf - cn;
                if self.trans {
                    self.trans_time += dt;
                    let span = cn - self.p0;
                    let prog = if self.trans_by_center {
                        // o centro curto ja passou da metade no instante da troca: o deslocamento
                        // migra enquanto ele completa a subida (~90 ms)
                        let cs = self.short_center(t, t, pt, a4).unwrap_or(cn);
                        let pc = if span.abs() > 0.05 { ((cs - self.p0) / span).clamp(0.0, 1.0) } else { 1.0 };
                        ((pc - 0.5) * 2.0).max(self.trans_time / if self.live() { LIVE_TRANS } else { 0.15 })
                    } else {
                        let pitch_prog = if span.abs() > 0.05 { ((pm - self.p0) / span).clamp(0.0, 1.0) } else { 1.0 };
                        pitch_prog.max(self.trans_time / if self.live() { 1.33 * LIVE_TRANS } else { 0.2 })
                    };
                    let prog = prog.clamp(0.0, 1.0);
                    self.o = self.o_a + (o_target - self.o_a) * smoothstep(prog);
                    self.o1 = self.o;
                    if prog >= 1.0 {
                        self.trans = false;
                    }
                } else if self.time_in_note <= dt * 1.5 {
                    self.o = o_target;
                    self.o1 = o_target;
                } else {
                    let tc = (p.speed_ms / 1000.0).max(0.005);
                    let a = 1.0 - (-dt / (0.5 * tc)).exp();
                    self.o1 += (o_target - self.o1) * a;
                    self.o += (self.o1 - self.o) * a;
                }
                let drift = c_loc - cn;
                let mut y = pm + p.amount * (self.o + s * (vib_mod - p.tighten * drift));
                // scoop de ataque (so depois de silencio): aproxima o desvio grande do alvo
                if self.onset_from_silence && self.time_in_note < 0.3 && p.scoop > 0.0 {
                    let dev = pm - cn;
                    let w = p.scoop * (1.0 - smoothstep(self.time_in_note / 0.3)) * smoothstep((dev.abs() - 0.1) / 0.2);
                    y -= w * p.amount * dev;
                }
                y
            }
            Style::Classic | Style::Hard => {
                let hard = p.style == Style::Hard;
                let err = tf - pm;
                let w = if p.flex > 0.0 && !hard {
                    let lim = 0.6 - 0.5 * p.flex;
                    1.0 - smoothstep((err.abs() - lim) / lim)
                } else {
                    1.0
                };
                let c = err * w * p.amount;
                let tc = if hard {
                    0.0
                } else {
                    p.speed_ms / 1000.0 * (1.0 + 3.0 * p.humanize * smoothstep((self.time_in_note - 0.1) / 0.5))
                };
                if tc < 0.002 {
                    self.cl1 = 0.0;
                    self.cl2 = c;
                } else {
                    // retune de 2a ordem subamortecido (Saitou), com o amortecimento medido em
                    // 2454 transicoes reais do VocalSet: a voz passa um pouco do alvo e assenta,
                    // como um cantor, em vez de chegar numa curva exponencial. Speed = tempo de
                    // subida de 10% a 90%. cl2 = correcao, cl1 = sua velocidade.
                    let z = self.cl_zeta;
                    let wn = (1.0 - 0.4167 * z + 2.917 * z * z) / tc;
                    let n = ((wn * dt / 0.15).ceil() as usize).clamp(1, 64);
                    let h = dt / n as f32;
                    for _ in 0..n {
                        let acc = wn * wn * (c - self.cl2) - 2.0 * z * wn * self.cl1;
                        self.cl1 += acc * h;
                        self.cl2 += self.cl1 * h;
                    }
                }
                pm + self.cl2 + if hard { 0.0 } else { p.amount * s * vib_mod }
            }
        };

        let shift = (y - pm).clamp(-24.0, 24.0);
        self.last_shift = shift;
        CtrlFrame {
            shift,
            input: pm,
            output: pm + shift,
            target: tf,
            vib_cents: 100.0 * self.vib_ms.sqrt(),
            hard_hz: if p.style == Style::Hard && p.amount >= 1.0 { a4 * ((tf - 69.0) / 12.0).exp2() } else { 0.0 },
            dbg: [cn, c_loc, anchor],
        }
    }

    /// Sem alvo possivel: a voz passa sem correcao.
    fn pass(&mut self, pm: f32) -> CtrlFrame {
        self.last_shift = 0.0;
        CtrlFrame { shift: 0.0, input: pm, output: pm, target: f32::NAN, vib_cents: 100.0 * self.vib_ms.sqrt(), ..Default::default() }
    }
}
