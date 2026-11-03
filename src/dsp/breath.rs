//! Separacao da voz em parte harmonica e ruido de respiracao (aspiracao).
//!
//! O PSOLA trata cada ciclo como se fosse todo periodico. Numa voz soprosa o ruido do ar e
//! recortado em graos e recolocado no periodo novo: ganha a periodicidade da nota corrigida e
//! vira um zumbido metalico. Aqui o ruido sai antes e passa ao lado do PSOLA, sem mudar de altura
//! (ruido nao tem nota); so a parte harmonica e corrigida.
//!
//! Parte harmonica = media dos ciclos vizinhos, alinhados pelas marcas de epoca. Cada ciclo e a
//! resposta do trato vocal a um pulso glotal, entao o ciclo vizinho e lido a mesma distancia da
//! sua marca (sem esticar o tempo: os formantes ressoam no mesmo tempo em ciclos de periodos
//! diferentes). O que se repete de ciclo a ciclo fica na media; o ruido, que nao se repete, cai
//! com o numero de ciclos. Entre duas marcas, as medias centradas em cada uma sao misturadas por
//! metades de Hann complementares: a soma e exata e o resultado nao tem emendas.
//!
//! O ruido do sinal desloca as marcas em fracoes de amostra, e nas altas frequencias isso basta
//! para a media borrar os harmonicos. Por isso cada vizinho e realinhado ao ciclo atual pela
//! correlacao em banda larga, com resolucao fracionaria de amostra.
//!
//! Ruido = passa-altas(voz - harmonica). Abaixo do corte o residuo e quase so erro de modelo (o
//! fantasma da nota antiga, que soaria desafinado) e a respiracao e fraca; acima dele esta o ar.
//! O passa-altas e `1 - passa-baixas FIR de fase linear`: um IIR teria fase propria e a parte
//! complementar (`voz - ruido`) deixaria passar ruido logo acima do corte. A parte harmonica
//! entregue ao PSOLA e `voz - ruido`, entao sem correcao a soma das duas partes e a propria voz,
//! amostra por amostra.
//!
//! Trechos sem vozeamento ficam inteiros na parte harmonica (que ali passa sem mudar de altura).

use super::marks::MarkTracker;
use super::util::Ring;

/// Vizinhos por lado na media.
const MAX_NB: usize = 16;
const TAPS: usize = 2 * MAX_NB + 1;
/// Maior ciclo usado no alinhamento (amostras; 62 Hz a 96 kHz cabe).
const MAX_CYCLE: usize = 2048;
/// Bloco do FIR de corte.
const FIR_BLOCK: usize = 256;

/// Frequencia de corte do ruido (Hz).
pub const SPLIT_HZ: f64 = 1000.0;
/// Meia duracao do FIR de corte (s): atraso que a separacao precisa de sinal a frente.
pub const SPLIT_HALF: f64 = 0.003;
/// No modo ao vivo o FIR e mais curto (cabe na latencia; o corte fica mais suave).
pub const SPLIT_HALF_LIVE: f64 = 0.0015;

/// Receita da media em torno de uma marca: deslocamentos ate os ciclos vizinhos e pesos
/// normalizados (o proprio ciclo tem deslocamento 0).
#[derive(Clone, Copy)]
struct Recipe {
    k: u64,
    n: usize,
    off: [f64; TAPS],
    /// Peso ja multiplicado pelo ganho do vizinho.
    w: [f32; TAPS],
    g: [f32; TAPS],
    /// Leitura pronta para instantes inteiros: parte inteira do deslocamento e os 4 pesos de
    /// Catmull-Rom da parte fracionaria (ja vezes o peso do vizinho).
    io: [i64; TAPS],
    cw: [[f32; 4]; TAPS],
}

impl Recipe {
    const NONE: Self = Self { k: u64::MAX, n: 0, off: [0.0; TAPS], w: [0.0; TAPS], g: [1.0; TAPS], io: [0; TAPS], cw: [[0.0; 4]; TAPS] };

    fn new(k: u64) -> Self {
        let mut r = Self::NONE;
        r.k = k;
        r.n = 1;
        r.w[0] = 1.0;
        r
    }

    /// Prepara a leitura depois de fixar deslocamentos e pesos.
    fn bake(&mut self) {
        for i in 0..self.n {
            let fl = self.off[i].floor();
            let f = (self.off[i] - fl) as f32;
            let (f2, f3) = (f * f, f * f * f);
            let w = self.w[i];
            self.io[i] = fl as i64;
            self.cw[i] = [
                w * (-0.5 * f3 + f2 - 0.5 * f),
                w * (1.5 * f3 - 2.5 * f2 + 1.0),
                w * (-1.5 * f3 + 2.0 * f2 + 0.5 * f),
                w * (0.5 * f3 - 0.5 * f2),
            ];
        }
    }

    #[inline]
    fn eval(&self, x: &Ring, s: i64) -> f32 {
        if self.n <= 1 {
            return x.at(s);
        }
        let mut acc = 0.0f32;
        for i in 0..self.n {
            let j = s + self.io[i];
            let c = &self.cw[i];
            acc += c[0] * x.at(j - 1) + c[1] * x.at(j) + c[2] * x.at(j + 1) + c[3] * x.at(j + 2);
        }
        acc
    }
}

pub struct BreathSplit {
    /// Parte harmonica (o que o PSOLA corrige).
    pub harm: Ring,
    /// Ruido de respiracao (passa sem correcao).
    pub noise: Ring,
    /// Residuo voz - media dos ciclos (antes do corte).
    res: Ring,
    /// Residuo calculado em `[0, rdone)`; partes separadas em `[0, rdone - half)`.
    rdone: i64,
    /// Indice da marca com `pos <= rdone` (cursor).
    k: u64,
    ra: Recipe,
    rb: Recipe,
    /// Passa-baixas FIR simetrico (2*half+1 coeficientes).
    lp: Vec<f32>,
    half: i64,
    fir_in: Vec<f32>,
    fir_acc: Vec<f32>,
    /// Passo do alinhamento (amostras): 2 acima de 64 kHz (nada de voz acima de 24 kHz importa).
    stride: usize,
    /// Alcance da media para tras e para a frente (amostras).
    past: f64,
    fut: f64,
    cycle: [f32; MAX_CYCLE + 1],
    /// Amostras separadas antes de haver marcas suficientes a frente (so deveria acontecer no
    /// modo ao vivo, que nao espera o futuro).
    pub provisional: u64,
}

impl BreathSplit {
    /// `past`/`fut`: alcance da media em segundos (`fut = 0` no modo ao vivo).
    pub fn new(sr: f32, past: f64, fut: f64) -> Self { Self::with_fir(sr, past, fut, SPLIT_HALF) }

    /// Como `new`, com a meia duracao do FIR de corte escolhida (s).
    pub fn with_fir(sr: f32, past: f64, fut: f64, fir_half: f64) -> Self {
        let sr64 = sr as f64;
        let half = (fir_half * sr64).round() as i64;
        // seno cardinal com janela de Blackman, ganho 1 em DC
        let wc = SPLIT_HZ / sr64;
        let mut lp: Vec<f32> = (-half..=half)
            .map(|i| {
                let t = i as f64;
                let sinc = if i == 0 { 2.0 * wc } else { (std::f64::consts::TAU * wc * t).sin() / (std::f64::consts::PI * t) };
                let u = std::f64::consts::PI * (t / half as f64 + 1.0);
                let win = 0.42 - 0.5 * u.cos() + 0.08 * (2.0 * u).cos();
                (sinc * win) as f32
            })
            .collect();
        let sum: f32 = lp.iter().sum();
        for c in &mut lp {
            *c /= sum;
        }
        Self {
            harm: Ring::new(1 << 17),
            noise: Ring::new(1 << 17),
            res: Ring::new(1 << 17),
            rdone: 0,
            k: 0,
            ra: Recipe::NONE,
            rb: Recipe::NONE,
            fir_in: vec![0.0; FIR_BLOCK + 2 * half as usize],
            fir_acc: vec![0.0; FIR_BLOCK],
            lp,
            half,
            stride: if sr > 64_000.0 { 2 } else { 1 },
            past: past * sr64,
            fut: fut * sr64,
            cycle: [0.0; MAX_CYCLE + 1],
            provisional: 0,
        }
    }

    /// Ate onde ja ha separacao (exclusivo).
    pub fn done(&self) -> i64 { self.rdone - self.half }

    /// Quanto sinal a frente a separacao precisa alem das marcas (amostras).
    pub fn lookahead(&self) -> i64 { self.half }

    /// Separa tudo que as marcas ja permitem de forma definitiva (com todos os vizinhos futuros).
    pub fn advance(&mut self, x: &Ring, marks: &MarkTracker) { self.run(x, x, marks, i64::MAX, false); }

    /// Garante a separacao ate `upto` (exclusivo), mesmo sem todos os vizinhos futuros.
    /// `avail` = amostras de entrada ja escritas.
    pub fn ensure(&mut self, x: &Ring, marks: &MarkTracker, upto: i64, avail: i64) { self.ensure_guided(x, x, marks, upto, avail); }

    /// Separacao desligada: a parte harmonica e a propria voz e nao ha ruido (barato; mantem a
    /// linha do tempo em dia para religar sem custo).
    pub fn bypass(&mut self, x: &Ring, avail: i64) {
        let half = self.half;
        for s in self.rdone..avail {
            self.res.set(s, 0.0);
        }
        for s in (self.rdone - half).max(0)..avail - half {
            self.harm.set(s, x.at(s));
            self.noise.set(s, 0.0);
        }
        self.rdone = self.rdone.max(avail);
    }

    /// Como `advance`, mas com o alinhamento e os ganhos dos ciclos tirados de `guide` (para
    /// medir: separar cada componente de uma mistura com a receita da propria mistura).
    pub fn advance_guided(&mut self, x: &Ring, guide: &Ring, marks: &MarkTracker) { self.run(x, guide, marks, i64::MAX, false); }

    pub fn ensure_guided(&mut self, x: &Ring, guide: &Ring, marks: &MarkTracker, upto: i64, avail: i64) {
        if upto > self.done() {
            self.run(x, guide, marks, (upto + self.half).min(avail), true);
        }
    }

    fn run(&mut self, x: &Ring, guide: &Ring, marks: &MarkTracker, upto: i64, force: bool) {
        let count = marks.count();
        if count == 0 {
            return;
        }
        let latest = marks.latest_pos();
        let r0 = self.rdone;
        while self.rdone < upto {
            let s = self.rdone;
            let sf = s as f64;
            self.k = self.k.max(marks.oldest());
            while self.k + 1 < count && marks.get(self.k + 1).pos <= sf {
                self.k += 1;
            }
            let a = *marks.get(self.k);
            let next = (self.k + 1 < count).then(|| *marks.get(self.k + 1));
            // definitivo so com a marca seguinte e o futuro das duas medias ja conhecidos
            let ready = match next {
                Some(b) => !b.voiced || b.pos + self.fut < latest,
                None => false,
            };
            if !ready {
                if !force {
                    break;
                }
                // sem a marca seguinte o sinal ainda nao chegou ate aqui
                if next.is_none() && sf > latest + 4.0 * a.period as f64 {
                    break;
                }
                self.provisional += 1;
            }
            if self.ra.k != self.k {
                self.ra = if self.rb.k == self.k { self.rb } else { self.recipe(guide, marks, self.k) };
            }
            let h = match next {
                Some(b) => {
                    if self.rb.k != self.k + 1 {
                        self.rb = self.recipe(guide, marks, self.k + 1);
                    }
                    let phi = ((sf - a.pos) / (b.pos - a.pos).max(1.0)).clamp(0.0, 1.0) as f32;
                    let wa = 0.5 + 0.5 * (std::f32::consts::PI * phi).cos();
                    let (ha, hb) = (self.ra.eval(x, s), self.rb.eval(x, s));
                    wa * ha + (1.0 - wa) * hb
                }
                None => self.ra.eval(x, s),
            };
            self.res.set(s, x.at(s) - h);
            self.rdone += 1;
        }
        // corte: ruido = residuo - passa-baixas(residuo), centrado. Em blocos contiguos, com o
        // laco interno sobre as amostras (vetoriza)
        let half = self.half;
        let hu = half as usize;
        let mut s0 = (r0 - half).max(0);
        let end = self.rdone - half;
        while s0 < end {
            let n = ((end - s0) as usize).min(FIR_BLOCK);
            let buf = &mut self.fir_in[..n + 2 * hu];
            for (j, v) in buf.iter_mut().enumerate() {
                *v = self.res.at(s0 - half + j as i64);
            }
            let acc = &mut self.fir_acc[..n];
            let c0 = self.lp[hu];
            for (a, &v) in acc.iter_mut().zip(&buf[hu..hu + n]) {
                *a = c0 * v;
            }
            for i in 1..=hu {
                let c = self.lp[hu + i];
                let (lo, hi) = (&buf[hu - i..hu - i + n], &buf[hu + i..hu + i + n]);
                for ((a, &l), &h) in acc.iter_mut().zip(lo).zip(hi) {
                    *a += c * (l + h);
                }
            }
            for j in 0..n {
                let s = s0 + j as i64;
                let nz = buf[hu + j] - acc[j];
                self.noise.set(s, nz);
                self.harm.set(s, x.at(s) - nz);
            }
            s0 += n as i64;
        }
    }

    /// Receita da media em torno da marca `k`: ciclos vozeados contiguos, de periodo parecido,
    /// dentro do alcance, com peso de Hann pela distancia, cada um realinhado ao ciclo atual.
    fn recipe(&mut self, x: &Ring, marks: &MarkTracker, k: u64) -> Recipe {
        let m = *marks.get(k);
        let mut r = Recipe::new(k);
        if !m.voiced {
            return r;
        }
        let similar = |p: f32| (p / m.period - 1.0).abs() < 0.2;
        let add = |r: &mut Recipe, d: f64, span: f64| {
            let u = (d.abs() / (span + m.period as f64)).min(1.0);
            r.off[r.n] = d;
            r.w[r.n] = (0.5 + 0.5 * (std::f64::consts::PI * u).cos()) as f32;
            r.n += 1;
        };
        let oldest = marks.oldest();
        let mut j = k;
        for _ in 0..MAX_NB {
            if j <= oldest {
                break;
            }
            j -= 1;
            let o = marks.get(j);
            let d = o.pos - m.pos;
            if !o.voiced || !similar(o.period) || -d > self.past {
                break;
            }
            add(&mut r, d, self.past);
        }
        let mut j = k;
        for _ in 0..MAX_NB {
            if j + 1 >= marks.count() {
                break;
            }
            j += 1;
            let o = marks.get(j);
            let d = o.pos - m.pos;
            if !o.voiced || !similar(o.period) || d > self.fut {
                break;
            }
            add(&mut r, d, self.fut);
        }
        if r.n > 1 {
            self.align(x, m.pos, m.period, &mut r);
        }
        let sum: f32 = r.w[..r.n].iter().sum();
        for i in 0..r.n {
            r.w[i] *= r.g[i] / sum;
        }
        r.bake();
        r
    }

    /// Realinha cada vizinho ao ciclo em `pos`: maximo da correlacao em banda larga em torno do
    /// deslocamento das marcas (ate ~1 amostra para cada lado).
    fn align(&mut self, x: &Ring, pos: f64, period: f32, r: &mut Recipe) {
        let half = ((0.5 * period).round() as usize).min(MAX_CYCLE / 2);
        let len = 2 * half + 1;
        let stride = self.stride;
        for t in (0..len).step_by(stride) {
            self.cycle[t] = x.interp(pos + t as f64 - half as f64);
        }
        let cyc = &self.cycle[..len];
        let e_k: f32 = cyc.iter().step_by(stride).map(|v| v * v).sum();
        let corr = |d: f64| -> (f32, f32) {
            // Catmull-Rom com a mesma fracao para o ciclo todo
            let p = pos + d - half as f64;
            let i0 = p.floor();
            let f = (p - i0) as f32;
            let (f2, f3) = (f * f, f * f * f);
            let w0 = -0.5 * f3 + f2 - 0.5 * f;
            let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
            let w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
            let w3 = 0.5 * f3 - 0.5 * f2;
            let i0 = i0 as i64;
            let (mut acc, mut e) = (0.0f32, 0.0f32);
            for (t, &c) in cyc.iter().enumerate().step_by(stride) {
                let i = i0 + t as i64;
                let v = w0 * x.at(i - 1) + w1 * x.at(i) + w2 * x.at(i + 1) + w3 * x.at(i + 2);
                acc += c * v;
                e += v * v;
            }
            (acc, e)
        };
        for i in 1..r.n {
            // dois passos de parabola (3 pontos cada): resolucao melhor que 1/16 de amostra
            let d0 = r.off[i];
            let mut best = (d0, corr(d0));
            for step in [0.5f64, 0.125] {
                let centre = best.0;
                let (a, c) = (corr(centre - step), corr(centre + step));
                let b = best.1;
                let den = a.0 - 2.0 * b.0 + c.0;
                if den < 0.0 {
                    let u = (0.5 * (a.0 - c.0) / den).clamp(-1.5, 1.5) as f64;
                    let d = centre + u * step;
                    let v = corr(d);
                    best = if v.0 >= b.0 { (d, v) } else { best };
                } else if a.0 > b.0 || c.0 > b.0 {
                    best = if a.0 > c.0 { (centre - step, a) } else { (centre + step, c) };
                }
            }
            r.off[i] = best.0;
            // ganho do vizinho: ataques e finais de nota mudam de volume a cada ciclo, e a media
            // nao pode deixar essa diferenca no residuo. Razao de energias, nao minimos
            // quadrados: o ruido dos dois ciclos entra igual nas duas energias, enquanto na
            // projecao ele so pesa no denominador e encolhe a harmonica (que vazaria no ruido).
            let (c, e) = best.1;
            r.g[i] = if e > 1e-12 && c > 0.0 { (e_k / e).sqrt().clamp(0.25, 4.0) } else { 1.0 };
        }
    }
}

impl Default for BreathSplit {
    /// Vazio (so para trocas temporarias dentro do motor).
    fn default() -> Self {
        Self {
            harm: Ring::new(1),
            noise: Ring::new(1),
            res: Ring::new(1),
            rdone: 0,
            k: 0,
            ra: Recipe::NONE,
            rb: Recipe::NONE,
            lp: Vec::new(),
            half: 0,
            fir_in: Vec::new(),
            fir_acc: Vec::new(),
            stride: 1,
            past: 0.0,
            fut: 0.0,
            cycle: [0.0; MAX_CYCLE + 1],
            provisional: 0,
        }
    }
}
