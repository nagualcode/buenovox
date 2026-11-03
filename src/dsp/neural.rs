//! Detector neural de apoio: uma rede convolucional pequena que olha o mesmo trecho de sinal do
//! YIN e diz onde esta a altura e se ha voz. Nao substitui o YIN (que e mais preciso quando
//! acerta): arbitra os casos em que ele erra a oitava ou o vozeamento (voz rouca, soprosa,
//! belting com subharmonicos, vogais com o 2o harmonico forte, reverb, microfone fino).
//!
//! Entradas, num eixo logaritmico de altura (2 posicoes por semitom, de 62 Hz a ~1080 Hz):
//! * a periodicidade do YIN, `1 - CMND(sr/f)`: vale fundo = periodo provavel;
//! * o espectro do trecho (janela de Hann, log, normalizado pelo nivel) lido em f/2, f, 2f, 3f,
//!   4f e 5f: na altura certa os canais harmonicos acendem juntos e o de f/2 fica apagado;
//! * nivel relativo, nivel absoluto e sibilancia, repetidos em todas as posicoes.
//!
//! Tudo em Hz e segundos: as entradas nao dependem da taxa de amostragem. Uma pilha de
//! convolucoes ao longo do eixo de altura (a mesma regra vale em qualquer altura) da a
//! probabilidade de cada posicao e, por media global, a de haver voz. Pesos treinados fora
//! (train/), embutidos no binario; a inferencia e escrita aqui, sem alocar no audio.

/// Posicoes de altura.
pub const BINS: usize = 100;
/// Posicoes por semitom.
pub const PER_ST: f32 = 2.0;
/// Altura da posicao 0 (Hz).
pub const F_LO: f32 = 62.0;
/// Harmonicos lidos no espectro.
pub const HARMS: [f32; 6] = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
/// Canais de entrada: periodicidade, espectro em cada harmonico e 3 escalares.
pub const CH: usize = 1 + HARMS.len() + 3;
pub const FEAT_LEN: usize = CH * BINS;

#[inline]
pub fn bin_hz(b: f32) -> f32 { F_LO * (b / (12.0 * PER_ST)).exp2() }

#[inline]
pub fn hz_bin(f: f32) -> f32 { 12.0 * PER_ST * (f / F_LO).log2() }

/// Monta as entradas da rede. `cmnd[tau]` (tau em amostras), `mag[k]` = |X| do trecho em
/// `nfft` pontos; `rel_db` = nivel do quadro menos o pico recente, `db` = nivel absoluto (dBFS),
/// `hf` = fracao de energia acima de ~4 kHz.
#[allow(clippy::too_many_arguments)]
pub fn features(out: &mut [f32; FEAT_LEN], cmnd: &[f32], mag: &[f32], sr: f32, nfft: usize, rel_db: f32, db: f32, hf: f32) {
    let tmax = cmnd.len() as f32 - 2.0;
    // referencia de nivel: RMS do espectro entre 60 Hz e 6 kHz
    let k_hz = nfft as f32 / sr;
    let (k0, k1) = ((60.0 * k_hz) as usize, ((6000.0 * k_hz) as usize).min(mag.len() - 1));
    let mut e = 0.0f32;
    for &m in &mag[k0..=k1] {
        e += m * m;
    }
    let reference = (e / (k1 - k0 + 1) as f32).sqrt().max(1e-9);
    let nyq = mag.len() as f32 - 2.0;
    let scal = [(rel_db / 30.0 + 1.0).clamp(0.0, 1.5), ((db + 60.0) / 60.0).clamp(0.0, 1.5), hf.clamp(0.0, 1.0)];
    for b in 0..BINS {
        let f = bin_hz(b as f32);
        let tau = (sr / f).clamp(1.0, tmax);
        let i = tau.floor() as usize;
        let fr = tau - i as f32;
        let c = cmnd[i] * (1.0 - fr) + cmnd[i + 1] * fr;
        out[b] = (1.0 - c).clamp(-1.0, 1.0);
        for (h, &mult) in HARMS.iter().enumerate() {
            let pos = f * mult * k_hz;
            let v = if pos < nyq {
                let i = pos.floor() as usize;
                let fr = pos - i as f32;
                let m = mag[i] * (1.0 - fr) + mag[i + 1] * fr;
                (1.0 + 20.0 * m / reference).ln()
            } else {
                0.0
            };
            out[(1 + h) * BINS + b] = v;
        }
        for (j, &s) in scal.iter().enumerate() {
            out[(1 + HARMS.len() + j) * BINS + b] = s;
        }
    }
}

/// Convolucao 1D ao longo do eixo de altura, com preenchimento de zeros ("same").
struct Conv {
    cin: usize,
    cout: usize,
    k: usize,
    dil: usize,
    /// [cout][cin][k]
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Conv {
    fn forward(&self, x: &[f32], y: &mut [f32], relu: bool) {
        let pad = (self.k / 2 * self.dil) as isize;
        for co in 0..self.cout {
            let yo = &mut y[co * BINS..(co + 1) * BINS];
            yo.fill(self.b[co]);
            for ci in 0..self.cin {
                let xi = &x[ci * BINS..(ci + 1) * BINS];
                for kk in 0..self.k {
                    let w = self.w[(co * self.cin + ci) * self.k + kk];
                    let sh = kk as isize * self.dil as isize - pad;
                    // y[b] += w * x[b + sh], so onde b + sh cai dentro
                    let b0 = (-sh).max(0) as usize;
                    let b1 = (BINS as isize - sh).min(BINS as isize).max(0) as usize;
                    if b0 >= b1 {
                        continue;
                    }
                    let xs = &xi[(b0 as isize + sh) as usize..(b1 as isize + sh) as usize];
                    for (o, &v) in yo[b0..b1].iter_mut().zip(xs) {
                        *o += w * v;
                    }
                }
            }
            if relu {
                for o in yo.iter_mut() {
                    *o = o.max(0.0);
                }
            }
        }
    }
}

/// Saida da rede num quadro.
#[derive(Clone, Copy, Debug, Default)]
pub struct NetOut {
    /// Altura mais provavel (Hz).
    pub f0: f32,
    /// Probabilidade concentrada em torno dela (+-1 semitom), 0..1.
    pub conf: f32,
    /// Probabilidade de haver voz, 0..1.
    pub voiced: f32,
}

pub struct PitchNet {
    layers: Vec<Conv>,
    head: Conv,
    /// Vozeamento: media e maximo de cada canal da ultima camada -> 1.
    vw: Vec<f32>,
    vb: f32,
    hid: usize,
    a: Vec<f32>,
    bbuf: Vec<f32>,
    logits: Vec<f32>,
}

/// Pesos treinados (train/export.py).
static WEIGHTS: &[u8] = include_bytes!("../../models/pitchnet.bin");

impl PitchNet {
    /// A rede com os pesos embutidos no binario.
    pub fn embedded() -> Option<Self> { Self::from_bytes(WEIGHTS) }

    /// Formato: "BVXN", versao u32, numero de camadas ocultas u32; para cada camada (e depois a
    /// cabeca) cin, cout, k, dil (u32), pesos e vieses (f32); por fim os pesos do vozeamento
    /// (2*canais) e o vies. Tudo little-endian.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        let mut p = 0usize;
        let u32_at = |p: &mut usize| -> Option<u32> {
            let v = u32::from_le_bytes(data.get(*p..*p + 4)?.try_into().ok()?);
            *p += 4;
            Some(v)
        };
        if data.get(0..4)? != b"BVXN" {
            return None;
        }
        p += 4;
        let version = u32_at(&mut p)?;
        if version != 1 {
            return None;
        }
        let n = u32_at(&mut p)? as usize;
        let read_f32s = |p: &mut usize, n: usize| -> Option<Vec<f32>> {
            let bytes = data.get(*p..*p + 4 * n)?;
            *p += 4 * n;
            Some(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        };
        let read_conv = |p: &mut usize| -> Option<Conv> {
            let (cin, cout, k, dil) = (u32_at(p)? as usize, u32_at(p)? as usize, u32_at(p)? as usize, u32_at(p)? as usize);
            let w = read_f32s(p, cout * cin * k)?;
            let b = read_f32s(p, cout)?;
            Some(Conv { cin, cout, k, dil, w, b })
        };
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            layers.push(read_conv(&mut p)?);
        }
        let head = read_conv(&mut p)?;
        let hid = layers.last()?.cout;
        let vw = read_f32s(&mut p, 2 * hid)?;
        let vb = read_f32s(&mut p, 1)?[0];
        if layers[0].cin != CH || head.cout != 1 || head.cin != hid {
            return None;
        }
        let width = layers.iter().map(|l| l.cout).max()?.max(CH);
        Some(Self { layers, head, vw, vb, hid, a: vec![0.0; width * BINS], bbuf: vec![0.0; width * BINS], logits: vec![0.0; BINS] })
    }

    /// Probabilidades de altura (softmax) e logit de vozeamento para as entradas `feat`.
    pub fn run(&mut self, feat: &[f32; FEAT_LEN], probs: &mut [f32; BINS]) -> f32 {
        self.a[..FEAT_LEN].copy_from_slice(feat);
        for l in &self.layers {
            l.forward(&self.a, &mut self.bbuf, true);
            std::mem::swap(&mut self.a, &mut self.bbuf);
        }
        self.head.forward(&self.a, &mut self.logits, false);
        let mx = self.logits.iter().fold(f32::MIN, |m, &v| m.max(v));
        let mut sum = 0.0;
        for (p, &l) in probs.iter_mut().zip(&self.logits) {
            *p = (l - mx).exp();
            sum += *p;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let mut v = self.vb;
        for c in 0..self.hid {
            let ch = &self.a[c * BINS..(c + 1) * BINS];
            let mean = ch.iter().sum::<f32>() / BINS as f32;
            let max = ch.iter().fold(f32::MIN, |m, &x| m.max(x));
            v += self.vw[c] * mean + self.vw[self.hid + c] * max;
        }
        v
    }

    /// Roda e resume: altura (media ponderada em torno do maximo), confianca e vozeamento.
    pub fn infer(&mut self, feat: &[f32; FEAT_LEN], probs: &mut [f32; BINS]) -> NetOut {
        let vlogit = self.run(feat, probs);
        let (bmax, _) = probs.iter().enumerate().fold((0, f32::MIN), |a, (i, &p)| if p > a.1 { (i, p) } else { a });
        let lo = bmax.saturating_sub(2);
        let hi = (bmax + 2).min(BINS - 1);
        let (mut sw, mut sb) = (0.0f32, 0.0f32);
        for (b, &p) in probs.iter().enumerate().take(hi + 1).skip(lo) {
            sw += p;
            sb += p * b as f32;
        }
        NetOut { f0: bin_hz(sb / sw.max(1e-9)), conf: sw, voiced: 1.0 / (1.0 + (-vlogit).exp()) }
    }
}
