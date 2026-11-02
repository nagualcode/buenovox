//! Motor do BuenoVOX: liga o rastreador de altura, as marcas de epoca, o cerebro e o shifter
//! PSOLA numa linha do tempo com latencia fixa.
//!
//! Linha do tempo: a amostra de entrada `n` sai em `n + L`. O cerebro decide o instante `t`
//! olhando a voz ate `t + lookahead`. Quanto maior o lookahead, mais o cerebro sabe da nota
//! antes de corrigi-la (modo Studio); o modo Track troca parte disso por latencia menor.
//!
//! Tonalidade, escala e MIDI chegam com o carimbo da amostra de ENTRADA em que valem e sao
//! consultados no instante do material analisado: uma nota MIDI no piano roll vale exatamente
//! sobre a silaba cantada naquele ponto.

use super::brain::{Brain, BrainParams, CtrlFrame};
use super::marks::MarkTracker;
use super::pitch::{PitchTracker, F0_MIN};
use super::psola::{max_half_len, GrainVoice, VoiceControl, VoiceState, FORMANT_MIN, LIVE_MAX_HALF};
use super::tonal::{Ctx, Source};
use super::util::{HannTable, Ring};

const CTRL_CAP: usize = 16384;
const CTX_CAP: usize = 1024;

/// Compromisso latencia x conhecimento da nota.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LatencyMode {
    /// Para cantar ao vivo: < 20 ms. O audio nao espera o detector (as decisoes de nota chegam
    /// ~30 ms depois do som; o ciclo a ciclo vem das marcas previstas).
    Live,
    /// Para gravar com monitoracao: lookahead curto.
    Track,
    /// Para mixar: o cerebro ve ~190 ms a frente (a nota quase inteira).
    #[default]
    Studio,
}

impl LatencyMode {
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Live,
            1 => Self::Track,
            _ => Self::Studio,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Live => 0,
            Self::Track => 1,
            Self::Studio => 2,
        }
    }

    fn lookahead_s(self) -> f64 {
        match self {
            Self::Live => 0.0,
            Self::Track => 0.008,
            Self::Studio => 0.19,
        }
    }
}

/// Latencia do modo ao vivo (s).
pub const LIVE_LATENCY: f64 = 0.018;

/// Parametros do motor (lidos do host a cada bloco).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineParams {
    pub brain: BrainParams,
    pub source: Source,
    pub key: u8,
    pub scale: u8,
    pub pure: bool,
    /// Alcance do pitch bend (semitons).
    pub bend_range: f32,
    /// Garganta -1..1 (negativo = mais longa, voz mais escura).
    pub throat: f32,
    pub bypass: bool,
}

impl Default for EngineParams {
    fn default() -> Self {
        Self {
            brain: BrainParams::default(),
            source: Source::Scale,
            key: 0,
            scale: 0,
            pure: false,
            bend_range: 2.0,
            throat: 0.0,
            bypass: false,
        }
    }
}

pub struct Engine {
    sr: f32,
    mode: LatencyMode,
    latency: usize,
    lookahead: f64,
    hop: usize,
    input: Ring,
    dry_l: Ring,
    dry_r: Ring,
    pitch: PitchTracker,
    marks: MarkTracker,
    hann: HannTable,
    n_in: i64,
    mono: Vec<f32>,

    params: EngineParams,
    ctx_events: Vec<(i64, Ctx)>,
    held: [u64; 2],
    latched: u16,
    latch_armed: bool,
    bend_raw: f32,

    frames: Vec<CtrlFrame>,
    frame_count: u64,
    brain: Brain,
    voice: GrainVoice,
    wet: f32,

    /// Quadros de controle pedidos antes de existirem (latencia curta demais).
    starved_ctrl: std::cell::Cell<u64>,
    /// Medidores (instante de saida mais recente).
    pub meter: CtrlFrame,
}

impl Engine {
    pub fn new(sr: f32, mode: LatencyMode) -> Self {
        let sr = sr.max(8000.0);
        let pitch = PitchTracker::new(sr);
        let hop = pitch.hop;
        let lookahead = (mode.lookahead_s() * sr as f64).round();
        // Latencia minima: o quadro de altura mais recente fica `analysis_lag` atras da entrada;
        // as marcas precisam de meio ciclo alem disso; os graos sao planejados `max_half_len` a
        // frente da saida; o cerebro precisa de `lookahead` alem do grao mais adiantado.
        let lag = (pitch.analysis_lag() as f64) - 0.5 * pitch.window() as f64 + 2.0 * hop as f64;
        let period = sr as f64 / F0_MIN as f64;
        let half = max_half_len(sr);
        let need_marks = lag + period + half;
        let need_ctrl = lag + half + lookahead + hop as f64;
        let live = mode == LatencyMode::Live;
        let latency = if live {
            (LIVE_LATENCY * sr as f64).round() as usize
        } else {
            (need_marks.max(need_ctrl) + 2.0 * hop as f64 + 0.5 * period + 64.0).ceil() as usize
        };
        let mut marks = MarkTracker::new(sr);
        marks.live = live;
        let mut voice = GrainVoice::new();
        if live {
            voice.max_half = LIVE_MAX_HALF * sr as f64;
        }
        Self {
            sr,
            mode,
            latency,
            lookahead,
            hop,
            input: Ring::new(1 << 17),
            dry_l: Ring::new(1 << 16),
            dry_r: Ring::new(1 << 16),
            pitch,
            marks,
            hann: HannTable::new(),
            n_in: 0,
            mono: vec![0.0; 1024],
            params: EngineParams::default(),
            ctx_events: Vec::with_capacity(CTX_CAP),
            held: [0; 2],
            latched: 0,
            latch_armed: true,
            bend_raw: 0.0,
            frames: vec![CtrlFrame::default(); CTRL_CAP],
            frame_count: 0,
            brain: Brain::new(sr, hop, lookahead),
            voice,
            wet: 1.0,
            starved_ctrl: std::cell::Cell::new(0),
            meter: CtrlFrame::default(),
        }
    }

    pub fn sample_rate(&self) -> f32 { self.sr }

    pub fn mode(&self) -> LatencyMode { self.mode }

    pub fn latency(&self) -> usize { self.latency }

    pub fn pitch_tracker(&self) -> &PitchTracker { &self.pitch }

    /// Quantas vezes o motor precisou de informacao que ainda nao existia (deve ser 0).
    pub fn starved(&self) -> u64 { self.starved_ctrl.get() + self.voice.starved }

    fn ctx_now(&self) -> Ctx {
        let p = &self.params;
        Ctx {
            source: p.source,
            key: p.key % 12,
            scale: p.scale,
            held: self.held,
            latched: self.latched,
            bend: self.bend_raw * p.bend_range,
            pure: p.pure,
        }
    }

    fn push_ctx(&mut self, at: i64) {
        let c = self.ctx_now();
        if let Some((t_last, last)) = self.ctx_events.last_mut() {
            if *last == c {
                return;
            }
            if *t_last == at {
                *last = c;
                return;
            }
        }
        if self.ctx_events.len() == CTX_CAP {
            self.ctx_events.remove(0);
        }
        self.ctx_events.push((at, c));
    }

    /// Atualiza os parametros no inicio do bloco.
    pub fn set_params(&mut self, p: &EngineParams) {
        self.params = *p;
        self.push_ctx(self.n_in);
    }

    /// Evento MIDI no deslocamento `offset` do proximo bloco.
    pub fn note_on(&mut self, offset: usize, note: u8) {
        let n = (note & 0x7F) as usize;
        if self.latch_armed {
            self.latched = 0;
            self.latch_armed = false;
        }
        self.held[n >> 6] |= 1 << (n & 63);
        self.latched |= 1 << (n % 12);
        self.push_ctx(self.n_in + offset as i64);
    }

    pub fn note_off(&mut self, offset: usize, note: u8) {
        let n = (note & 0x7F) as usize;
        self.held[n >> 6] &= !(1 << (n & 63));
        if self.held == [0, 0] {
            // o proximo acorde substitui a trava
            self.latch_armed = true;
        }
        self.push_ctx(self.n_in + offset as i64);
    }

    pub fn all_notes_off(&mut self, offset: usize) {
        self.held = [0; 2];
        self.latch_armed = true;
        self.push_ctx(self.n_in + offset as i64);
    }

    /// Pitch bend normalizado em -1..1.
    pub fn pitch_bend(&mut self, offset: usize, norm: f32) {
        self.bend_raw = norm.clamp(-1.0, 1.0);
        self.push_ctx(self.n_in + offset as i64);
    }

    /// Mascara de classes de altura das notas MIDI seguradas (para a interface).
    pub fn held_mask(&self) -> u16 { self.ctx_now().held_mask() }

    fn prune_ctx(&mut self, before: f64) {
        let keep_from = self.ctx_events.iter().rposition(|(at, _)| (*at as f64) <= before).unwrap_or(0);
        if keep_from > 0 {
            self.ctx_events.drain(..keep_from);
        }
    }

    /// Processa um bloco estereo. As fatias devem ter o mesmo tamanho.
    pub fn process(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
        let mut pos = 0;
        while pos < n {
            let m = (n - pos).min(self.mono.len());
            self.process_chunk(&in_l[pos..pos + m], &in_r[pos..pos + m], &mut out_l[pos..pos + m], &mut out_r[pos..pos + m]);
            pos += m;
        }
    }

    fn process_chunk(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        let n = in_l.len();
        let n0 = self.n_in;
        for i in 0..n {
            let (l, r) = (sane(in_l[i]), sane(in_r[i]));
            let m = 0.5 * (l + r);
            self.mono[i] = m;
            let t = n0 + i as i64;
            self.input.set(t, m);
            self.dry_l.set(t, l);
            self.dry_r.set(t, r);
        }
        let mono = std::mem::take(&mut self.mono);
        self.pitch.push(&mono[..n]);
        self.mono = mono;
        self.n_in += n as i64;
        self.marks.update(&self.pitch, self.n_in);

        // quadros de controle
        let hop = self.hop as f64;
        let bp = self.params.brain;
        while (self.frame_count as f64) * hop + self.lookahead <= self.pitch.latest_time() {
            let k = self.frame_count;
            let events = &self.ctx_events;
            let fallback = self.ctx_now();
            let ctx_at = |t: f64| ctx_at(events, t, fallback);
            let fr = self.brain.frame(k as f64 * hop, &self.pitch, &ctx_at, &bp);
            self.frames[(k % CTRL_CAP as u64) as usize] = fr;
            self.frame_count += 1;
        }
        let lat = self.latency as i64;
        self.prune_ctx((n0 - lat) as f64 - self.sr as f64);

        // graos
        let tau_end = (n0 + n as i64 - lat) as f64;
        let until = tau_end + self.plan_ahead();
        let mut voice = std::mem::take(&mut self.voice);
        voice.plan(until, &self.input, &self.marks, &Ctl { e: self }, &self.hann);
        self.voice = voice;

        // saida: voz corrigida (mono) nos dois canais; bypass cruza para o seco atrasado
        let wet_t = if self.params.bypass { 0.0 } else { 1.0 };
        let a = 1.0 - (-1.0 / (0.01 * self.sr)).exp();
        for i in 0..n {
            let tau = n0 + i as i64 - lat;
            let y = self.voice.take(tau);
            self.wet += (wet_t - self.wet) * a;
            let (wet, dry) = (self.wet, 1.0 - self.wet);
            out_l[i] = wet * y + dry * self.dry_l.at(tau);
            out_r[i] = wet * y + dry * self.dry_r.at(tau);
        }
        self.meter = self.frame_at(tau_end);
    }

    /// Quadros de controle ja decididos: (instante de origem, quadro), a partir do indice `k0`.
    pub fn frames_from(&self, k0: u64) -> impl Iterator<Item = (f64, CtrlFrame)> + '_ {
        let oldest = self.frame_count.saturating_sub(CTRL_CAP as u64 - 4);
        (k0.max(oldest)..self.frame_count).map(|k| (k as f64 * self.hop as f64, self.frames[(k % CTRL_CAP as u64) as usize]))
    }

    pub fn frame_count(&self) -> u64 { self.frame_count }

    /// Quanto alem da saida os graos precisam ser planejados (a maior meia janela de saida).
    fn plan_ahead(&self) -> f64 {
        if self.mode == LatencyMode::Live {
            self.voice.max_half / FORMANT_MIN as f64 + 4.0
        } else {
            max_half_len(self.sr)
        }
    }

    fn frame_at(&self, t: f64) -> CtrlFrame {
        if self.frame_count == 0 {
            return CtrlFrame::default();
        }
        let k = (t / self.hop as f64).floor().max(0.0) as u64;
        let last = self.frame_count - 1;
        let oldest = self.frame_count.saturating_sub(CTRL_CAP as u64 - 4);
        self.frames[(k.clamp(oldest, last) % CTRL_CAP as u64) as usize]
    }

    /// Deslocamento interpolado no instante `t`.
    fn shift_at(&self, t: f64) -> f32 {
        if self.frame_count == 0 {
            return 0.0;
        }
        let x = (t / self.hop as f64).max(0.0);
        let last = self.frame_count - 1;
        // ao vivo o audio passa a frente das decisoes: vale a ultima (nao e fome)
        if x > last as f64 + 1.0 && t > self.hop as f64 * 8.0 && self.mode != LatencyMode::Live {
            self.starved_ctrl.set(self.starved_ctrl.get() + 1);
        }
        let oldest = self.frame_count.saturating_sub(CTRL_CAP as u64 - 4);
        let k0 = (x.floor() as u64).clamp(oldest, last);
        let k1 = (k0 + 1).min(last);
        let f = (x - k0 as f64).clamp(0.0, 1.0) as f32;
        let a = self.frames[(k0 % CTRL_CAP as u64) as usize].shift;
        let b = self.frames[(k1 % CTRL_CAP as u64) as usize].shift;
        a + (b - a) * f
    }
}

/// Contexto tonal valido no instante `t` (ultimo evento ate `t`).
fn ctx_at(events: &[(i64, Ctx)], t: f64, fallback: Ctx) -> Ctx {
    for (at, c) in events.iter().rev() {
        if (*at as f64) <= t {
            return *c;
        }
    }
    events.first().map_or(fallback, |(_, c)| *c)
}

/// Visao do motor para o shifter.
struct Ctl<'a> {
    e: &'a Engine,
}

impl VoiceControl for Ctl<'_> {
    fn state(&self, s: f64, src_period: f32) -> VoiceState {
        let formant = (self.e.params.throat * 0.3).exp2();
        let hard = self.e.frame_at(s).hard_hz;
        if hard > 0.0 && src_period > 0.0 && !self.e.params.bypass {
            // alvo absoluto: o ciclo sai exatamente no periodo da nota
            return VoiceState { ratio: hard * src_period / self.e.sr, formant };
        }
        VoiceState { ratio: (self.e.shift_at(s) / 12.0).exp2(), formant }
    }
}

#[inline]
fn sane(x: f32) -> f32 { if x.is_finite() { x.clamp(-8.0, 8.0) } else { 0.0 } }
