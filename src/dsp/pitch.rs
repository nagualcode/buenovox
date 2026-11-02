//! Rastreador de altura (f0) da voz (herdado do Heaven): YIN (de Cheveigne & Kawahara, 2002) calculado
//! via FFT, com escolha de candidatos por continuidade (evita erros de oitava), histerese de
//! vozeamento e um detector de sibilancia (razao de energia acima de ~4 kHz).
//!
//! Cada quadro descreve a janela de analise centrada em `t` (indice absoluto de amostra).
//! O motor trabalha com latencia fixa, entao os quadros podem olhar para o "futuro" da saida.

use super::util::{Biquad, Ring};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const F0_MIN: f32 = 62.0;
pub const F0_MAX: f32 = 1100.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct PitchFrame {
    /// Centro do quadro (amostra absoluta).
    pub t: f64,
    /// f0 em Hz (0 se nao vozeado).
    pub f0: f32,
    /// Confianca de vozeamento 0..1.
    pub conf: f32,
    /// Aperiodicidade YIN (0 = periodico perfeito).
    pub aper: f32,
    /// Nivel RMS do quadro (dBFS).
    pub db: f32,
    /// Fracao de energia em alta frequencia (sibilancia), 0..1.
    pub hf: f32,
    /// Melhor estimativa de f0 mesmo em quadros fracamente periodicos (caudas sopradas);
    /// igual a `f0` nos vozeados, 0 quando nao ha periodicidade aproveitavel.
    pub f0_weak: f32,
}

impl PitchFrame {
    #[inline]
    pub fn voiced(&self) -> bool { self.f0 > 0.0 }
}

pub struct PitchTracker {
    sr: f32,
    w: usize,
    tau_min: usize,
    tau_max: usize,
    pub hop: usize,
    nfft: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    a: Vec<f32>,
    b: Vec<f32>,
    sa: Vec<Complex<f32>>,
    sb: Vec<Complex<f32>>,
    corr: Vec<f32>,
    scratch_f: Vec<Complex<f32>>,
    scratch_i: Vec<Complex<f32>>,
    sq_prefix: Vec<f64>,
    cmnd: Vec<f32>,

    hp: Biquad,
    lp: Biquad,
    lp2: Biquad,
    hf_hp: Biquad,
    /// Sinal filtrado para a analise (passa-faixa ~50..1500 Hz).
    pub filtered: Ring,
    /// Sinal acima de ~4 kHz (para sibilancia).
    hf_ring: Ring,
    /// Sinal bruto (para nivel).
    raw: Ring,
    written: i64,

    next_t0: i64,
    prev_f0: f32,
    prev_voiced_t: f64,
    voiced_state: bool,
    level_peak_db: f32,

    /// Quadros ja finalizados (com mediana de 3 aplicada).
    frames: Vec<PitchFrame>,
    frame_head: usize,
    frame_count: u64,
    pending: [PitchFrame; 2],
    pending_n: usize,
    /// Historico recente de f0 vozeado (tessitura do cantor).
    reg_hist: [f32; REG_N],
    reg_head: usize,
    reg_n: usize,
    /// Diagnostico: resultado cru da ultima analise (antes da mediana).
    pub last_raw: PitchFrame,
}

const REG_N: usize = 512;

const FRAME_CAP: usize = 16384;
const MAX_CANDS: usize = 32;

impl PitchTracker {
    pub fn new(sr: f32) -> Self {
        let w = (sr * 0.016).round() as usize;
        let tau_min = (sr / F0_MAX).floor() as usize;
        let tau_max = (sr / F0_MIN).ceil() as usize;
        let hop = (sr * 0.004).round() as usize;
        let nfft = (w + tau_max + 2).next_power_of_two();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(nfft);
        let ifft = planner.plan_fft_inverse(nfft);
        let scratch_f = fft.make_scratch_vec();
        let scratch_i = ifft.make_scratch_vec();
        let sr64 = sr as f64;
        Self {
            sr,
            w,
            tau_min,
            tau_max,
            hop,
            nfft,
            a: vec![0.0; nfft],
            b: vec![0.0; nfft],
            sa: fft.make_output_vec(),
            sb: fft.make_output_vec(),
            corr: vec![0.0; nfft],
            scratch_f,
            scratch_i,
            fft,
            ifft,
            sq_prefix: vec![0.0; w + tau_max + 2],
            cmnd: vec![1.0; tau_max + 2],
            hp: Biquad::highpass(sr64, 50.0, 0.707),
            lp: Biquad::lowpass(sr64, 1500.0, 0.707),
            lp2: Biquad::lowpass(sr64, 1500.0, 0.707),
            hf_hp: Biquad::highpass(sr64, 4000.0, 0.707),
            filtered: Ring::new(1 << 16),
            hf_ring: Ring::new(1 << 16),
            raw: Ring::new(1 << 16),
            written: 0,
            next_t0: 0,
            prev_f0: 0.0,
            prev_voiced_t: -1e9,
            voiced_state: false,
            level_peak_db: -60.0,
            frames: vec![PitchFrame::default(); FRAME_CAP],
            frame_head: 0,
            frame_count: 0,
            pending: [PitchFrame::default(); 2],
            pending_n: 0,
            reg_hist: [0.0; REG_N],
            reg_head: 0,
            reg_n: 0,
            last_raw: PitchFrame::default(),
        }
    }

    pub fn sr(&self) -> f32 { self.sr }

    /// Janela de analise (amostras).
    pub fn window(&self) -> usize { self.w }

    /// Diagnostico: minimos locais do CMND do ultimo quadro analisado (tau, valor).
    pub fn debug_cands(&self) -> Vec<(usize, f32)> {
        (self.tau_min.max(2)..self.tau_max)
            .filter(|&t| self.cmnd[t] < 0.6 && self.cmnd[t] < self.cmnd[t - 1] && self.cmnd[t] <= self.cmnd[t + 1])
            .map(|t| (t, self.cmnd[t]))
            .collect()
    }

    /// Distancia (amostras) entre o fim do sinal escrito e o centro do quadro mais recente.
    pub fn analysis_lag(&self) -> usize { self.w + self.tau_max + 2 * self.hop }

    /// Acrescenta amostras (mono) e analisa todos os quadros possiveis.
    pub fn push(&mut self, x: &[f32]) {
        for &s in x {
            let s = if s.is_finite() { s } else { 0.0 };
            let f = self.lp2.process(self.lp.process(self.hp.process(s)));
            self.filtered.set(self.written, f);
            self.hf_ring.set(self.written, self.hf_hp.process(s));
            self.raw.set(self.written, s);
            self.written += 1;
        }
        while self.next_t0 + (self.w + self.tau_max) as i64 <= self.written {
            let t0 = self.next_t0;
            let fr = self.analyze(t0);
            self.last_raw = fr;
            self.finalize(fr);
            self.next_t0 += self.hop as i64;
        }
    }

    fn analyze(&mut self, t0: i64) -> PitchFrame {
        let (w, tmax, n) = (self.w, self.tau_max, self.nfft);
        let len_b = w + tmax;
        for i in 0..n {
            self.a[i] = if i < w { self.filtered.at(t0 + i as i64) } else { 0.0 };
            self.b[i] = if i < len_b { self.filtered.at(t0 + i as i64) } else { 0.0 };
        }
        // prefixo dos quadrados de b
        self.sq_prefix[0] = 0.0;
        for i in 0..len_b {
            let v = self.b[i] as f64;
            self.sq_prefix[i + 1] = self.sq_prefix[i] + v * v;
        }
        let _ = self.fft.process_with_scratch(&mut self.a, &mut self.sa, &mut self.scratch_f);
        let _ = self.fft.process_with_scratch(&mut self.b, &mut self.sb, &mut self.scratch_f);
        for (x, y) in self.sa.iter_mut().zip(self.sb.iter()) {
            *x = x.conj() * y;
        }
        if let Some(first) = self.sa.first_mut() {
            first.im = 0.0;
        }
        if let Some(last) = self.sa.last_mut() {
            last.im = 0.0;
        }
        let _ = self.ifft.process_with_scratch(&mut self.sa, &mut self.corr, &mut self.scratch_i);
        let norm = 1.0 / n as f64;
        let e0 = self.sq_prefix[w];

        // Diferenca e CMND
        let mut running = 0.0f64;
        self.cmnd[0] = 1.0;
        for tau in 1..=tmax {
            let et = self.sq_prefix[tau + w] - self.sq_prefix[tau];
            let r = self.corr[tau] as f64 * norm;
            let d = (e0 + et - 2.0 * r).max(0.0);
            running += d;
            self.cmnd[tau] = if running > 1e-12 { (d * tau as f64 / running) as f32 } else { 1.0 };
        }

        // Nivel e sibilancia
        let mut e_raw = 0.0f64;
        let mut e_hf = 0.0f64;
        for i in 0..w {
            let r = self.raw.at(t0 + i as i64) as f64;
            let h = self.hf_ring.at(t0 + i as i64) as f64;
            e_raw += r * r;
            e_hf += h * h;
        }
        let db = (10.0 * (e_raw / w as f64 + 1e-12).log10()) as f32;
        let hf = (e_hf / (e_raw + 1e-12)).min(1.0) as f32;
        self.level_peak_db = if db > self.level_peak_db { db } else { self.level_peak_db - 0.02 };
        self.level_peak_db = self.level_peak_db.max(-60.0);

        // Candidatos: minimos locais do CMND, em ordem crescente de tau (os primeiros sao os
        // que importam: o periodo verdadeiro e o menor vale profundo).
        let mut cands: [(usize, f32); MAX_CANDS] = [(0, 1.0); MAX_CANDS];
        let mut nc = 0;
        for tau in self.tau_min.max(2)..tmax {
            let c = self.cmnd[tau];
            if c < 0.6 && c < self.cmnd[tau - 1] && c <= self.cmnd[tau + 1] {
                cands[nc] = (tau, c);
                nc += 1;
                if nc == MAX_CANDS {
                    break;
                }
            }
        }
        let cands = &cands[..nc];
        let t_center_base = t0 as f64 + w as f64 * 0.5;
        let recent = t_center_base - self.prev_voiced_t < 0.06 * self.sr as f64;

        let yin = Self::yin_rule(cands);
        let pick = match yin {
            Some((_, vy)) if recent && self.prev_f0 > 0.0 => {
                // Continuidade: entre os candidatos quase tao bons quanto a escolha do YIN,
                // fica o de menor salto em relacao ao quadro anterior.
                let prev_tau = self.sr / self.prev_f0;
                let mut best = yin;
                let mut best_cost = f32::MAX;
                for &(tau, v) in cands {
                    if v > (vy + 0.1).min(0.45) {
                        continue;
                    }
                    let jump = 12.0 * (tau as f32 / prev_tau).log2().abs();
                    let cost = v + 0.035 * jump;
                    if cost < best_cost {
                        best_cost = cost;
                        best = Some((tau, v));
                    }
                }
                best
            }
            other => other,
        };

        let mut fr = PitchFrame { t: t_center_base, f0: 0.0, conf: 0.0, aper: 1.0, db, hf, f0_weak: 0.0 };
        if let Some((tau, v)) = pick {
            let (mut tau, mut v) = (tau, v);
            // Teste de oitava: se o vale em 2*tau e claramente mais fundo, tau era o 2o harmonico
            // (vogais como "o" e "u" tem o 2o harmonico forte).
            // Com o quadro anterior vozeado, so desce se a continuidade apoiar a oitava de baixo
            // (voz "creaky" tem pulsos alternados e periodo aparente dobrado sem mudar a nota).
            if let Some(&(t2, v2)) = cands.iter().find(|c| (c.0 as f32 / tau as f32 - 2.0).abs() < 0.08) {
                let supported = !(recent && self.prev_f0 > 0.0)
                    || (12.0 * (self.sr / t2 as f32 / self.prev_f0).log2()).abs() < 4.0;
                // ...a menos que o vale de baixo seja muito mais fundo (erro de oitava evidente)
                let dramatic = v2 < 0.3 * v && v2 < 0.03;
                if (supported || dramatic) && v2 < v - 0.04 && v2 < 0.6 * v {
                    tau = t2;
                    v = v2;
                }
            }
            // Correcao de sub-harmonico: num sinal periodico todos os multiplos do periodo tambem
            // sao vales. Se tau/k (k = 2..5) e um vale tao bom quanto, o periodo real e o menor.
            let mut best_t2 = tau;
            for &(t2, v2) in cands {
                if t2 >= best_t2 {
                    continue;
                }
                let ratio = tau as f32 / t2 as f32;
                let k = ratio.round();
                if (2.0..=5.0).contains(&k) && (ratio - k).abs() < 0.04 * k && v2 <= v + 0.025 && v2 < 0.1 {
                    best_t2 = t2;
                }
            }
            if best_t2 != tau {
                tau = best_t2;
                v = cands.iter().find(|c| c.0 == tau).map_or(v, |c| c.1);
            }
            let frac = if tau > 1 && tau < tmax {
                let (c0, c1, c2) = (self.cmnd[tau - 1], self.cmnd[tau], self.cmnd[tau + 1]);
                let den = c0 - 2.0 * c1 + c2;
                if den.abs() > 1e-9 { (0.5 * (c0 - c2) / den).clamp(-0.5, 0.5) } else { 0.0 }
            } else {
                0.0
            };
            let tau_f = tau as f32 + frac;
            let f0 = self.sr / tau_f;
            fr.t = t0 as f64 + 0.5 * (w as f64 + tau_f as f64);
            fr.aper = v;
            let gate = (self.level_peak_db - 42.0).max(-62.0);
            let thr = if self.voiced_state { 0.32 } else { 0.22 };
            let sibilant = hf > 0.45;
            // Portao de registro: longe da regiao em que o cantor esta cantando (mais de 14
            // semitons abaixo ou 19 acima da mediana recente), so vale se for muito periodico.
            let in_register = match self.register() {
                Some(reg) => {
                    let st = 12.0 * (f0 / reg).log2();
                    (-14.0..=19.0).contains(&st) || v < 0.04
                }
                None => true,
            };
            let voiced = v < thr && db > gate && !sibilant && in_register && (F0_MIN..=F0_MAX).contains(&f0);
            self.voiced_state = voiced;
            let weak_ok = v < 0.6 && db > gate - 6.0 && hf < 0.25 && in_register && (F0_MIN..=F0_MAX).contains(&f0);
            if voiced || weak_ok {
                fr.f0_weak = f0;
            }
            if voiced {
                self.reg_hist[self.reg_head] = f0;
                self.reg_head = (self.reg_head + 1) % REG_N;
                self.reg_n = (self.reg_n + 1).min(REG_N);
                fr.f0 = f0;
                fr.conf = ((0.4 - v) / 0.3).clamp(0.0, 1.0) * ((db - gate) / 6.0).clamp(0.0, 1.0);
                self.prev_f0 = f0;
                self.prev_voiced_t = fr.t;
            }
        } else {
            self.voiced_state = false;
        }
        fr
    }

    /// Mediana do f0 vozeado recente (~2 s de canto), se houver historico suficiente.
    fn register(&self) -> Option<f32> {
        if self.reg_n < 60 {
            return None;
        }
        let mut v = [0.0f32; REG_N];
        v[..self.reg_n].copy_from_slice(&self.reg_hist[..self.reg_n]);
        let v = &mut v[..self.reg_n];
        let mid = v.len() / 2;
        v.select_nth_unstable_by(mid, f32::total_cmp);
        Some(v[mid])
    }

    /// Regra classica do YIN: primeiro vale abaixo do limiar; senao o melhor vale.
    fn yin_rule(c: &[(usize, f32)]) -> Option<(usize, f32)> {
        c.iter().copied().find(|x| x.1 < 0.15).or_else(|| c.iter().copied().min_by(|a, b| a.1.total_cmp(&b.1)))
    }

    /// Mediana de 3 em f0 (remove erros isolados de um quadro) e grava o quadro.
    fn finalize(&mut self, fr: PitchFrame) {
        if self.pending_n < 2 {
            self.pending[self.pending_n] = fr;
            self.pending_n += 1;
            if self.pending_n == 2 {
                let first = self.pending[0];
                self.store(first);
            }
            return;
        }
        let (a, mut b, c) = (self.pending[0], self.pending[1], fr);
        if a.voiced() && b.voiced() && c.voiced() {
            let mut v = [a.f0, b.f0, c.f0];
            v.sort_by(f32::total_cmp);
            b.f0 = v[1];
        } else if a.voiced() && c.voiced() && !b.voiced() && b.hf < 0.3 && b.aper < 0.5 {
            // buraco de um quadro dentro de nota vozeada
            b.f0 = (a.f0 * c.f0).sqrt();
            b.f0_weak = b.f0;
            b.conf = 0.5 * (a.conf + c.conf);
        } else if !a.voiced() && !c.voiced() && b.voiced() {
            // quadro vozeado isolado: provavelmente ruido
            b.f0 = 0.0;
            b.conf = 0.0;
        }
        self.store(b);
        self.pending[0] = self.pending[1];
        self.pending[1] = fr;
    }

    fn store(&mut self, fr: PitchFrame) {
        self.frames[self.frame_head] = fr;
        self.frame_head = (self.frame_head + 1) % FRAME_CAP;
        self.frame_count += 1;
    }

    fn frame(&self, k: u64) -> &PitchFrame { &self.frames[(k % FRAME_CAP as u64) as usize] }

    /// Centro do quadro mais recente disponivel (ou -inf).
    pub fn latest_time(&self) -> f64 {
        if self.frame_count == 0 { f64::NEG_INFINITY } else { self.frame(self.frame_count - 1).t }
    }

    /// Estado da voz no instante `t`: interpolacao entre quadros vizinhos (f0 em escala log).
    pub fn at(&self, t: f64) -> PitchFrame {
        if self.frame_count == 0 {
            return PitchFrame::default();
        }
        let oldest = self.frame_count.saturating_sub(FRAME_CAP as u64 - 8);
        let last = self.frame_count - 1;
        if t >= self.frame(last).t {
            return *self.frame(last);
        }
        if t <= self.frame(oldest).t {
            return *self.frame(oldest);
        }
        // busca binaria
        let (mut lo, mut hi) = (oldest, last);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.frame(mid).t <= t { lo = mid } else { hi = mid }
        }
        let (a, b) = (*self.frame(lo), *self.frame(hi));
        let span = (b.t - a.t).max(1e-9);
        let x = ((t - a.t) / span) as f32;
        let mut out = if x < 0.5 { a } else { b };
        out.t = t;
        if a.voiced() && b.voiced() {
            out.f0 = (a.f0.ln() * (1.0 - x) + b.f0.ln() * x).exp();
        }
        if a.f0_weak > 0.0 && b.f0_weak > 0.0 {
            out.f0_weak = (a.f0_weak.ln() * (1.0 - x) + b.f0_weak.ln() * x).exp();
            out.conf = a.conf * (1.0 - x) + b.conf * x;
        }
        out.db = a.db * (1.0 - x) + b.db * x;
        out.hf = a.hf * (1.0 - x) + b.hf * x;
        out
    }
}
