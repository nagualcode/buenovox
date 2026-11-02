//! Blocos basicos: ring buffer com indice absoluto, filtros biquad, ruido suave e conversoes.

/// Ring buffer de f32 enderecado por indice absoluto de amostra (i64). O tamanho e potencia de 2,
/// entao `i & mask` funciona tambem para indices negativos (complemento de 2).
pub struct Ring {
    buf: Vec<f32>,
    mask: usize,
}

impl Ring {
    pub fn new(min_len: usize) -> Self {
        let len = min_len.next_power_of_two();
        Self { buf: vec![0.0; len], mask: len - 1 }
    }

    pub fn len(&self) -> usize { self.buf.len() }

    pub fn clear(&mut self) { self.buf.fill(0.0); }

    #[inline(always)]
    pub fn at(&self, i: i64) -> f32 { self.buf[(i as usize) & self.mask] }

    #[inline(always)]
    pub fn set(&mut self, i: i64, v: f32) { self.buf[(i as usize) & self.mask] = v; }

    #[inline(always)]
    pub fn add(&mut self, i: i64, v: f32) { self.buf[(i as usize) & self.mask] += v; }

    /// Le e zera (acumuladores de overlap-add).
    #[inline(always)]
    pub fn take(&mut self, i: i64) -> f32 {
        let k = (i as usize) & self.mask;
        let v = self.buf[k];
        self.buf[k] = 0.0;
        v
    }

    /// Interpolacao cubica de Hermite (Catmull-Rom) em posicao fracionaria.
    #[inline(always)]
    pub fn interp(&self, t: f64) -> f32 {
        let i = t.floor();
        let f = (t - i) as f32;
        let i = i as i64;
        let (y0, y1, y2, y3) = (self.at(i - 1), self.at(i), self.at(i + 1), self.at(i + 2));
        let c1 = 0.5 * (y2 - y0);
        let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
        let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
        ((c3 * f + c2) * f + c1) * f + y1
    }
}

/// Biquad (RBJ cookbook), forma direta transposta II.
#[derive(Clone, Copy, Default)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn from_coefs(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: (b0 / a0) as f32,
            b1: (b1 / a0) as f32,
            b2: (b2 / a0) as f32,
            a1: (a1 / a0) as f32,
            a2: (a2 / a0) as f32,
            z1: 0.0,
            z2: 0.0,
        }
    }

    pub fn lowpass(sr: f64, f: f64, q: f64) -> Self {
        let w = std::f64::consts::TAU * (f / sr).min(0.49);
        let (s, c) = w.sin_cos();
        let a = s / (2.0 * q);
        Self::from_coefs((1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0, 1.0 + a, -2.0 * c, 1.0 - a)
    }

    pub fn highpass(sr: f64, f: f64, q: f64) -> Self {
        let w = std::f64::consts::TAU * (f / sr).min(0.49);
        let (s, c) = w.sin_cos();
        let a = s / (2.0 * q);
        Self::from_coefs((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0, 1.0 + a, -2.0 * c, 1.0 - a)
    }

    pub fn high_shelf(sr: f64, f: f64, gain_db: f64) -> Self {
        let a = 10f64.powf(gain_db / 40.0);
        let w = std::f64::consts::TAU * (f / sr).min(0.49);
        let (s, c) = w.sin_cos();
        let alpha = s / 2.0 * std::f64::consts::SQRT_2;
        let sa = 2.0 * a.sqrt() * alpha;
        Self::from_coefs(
            a * ((a + 1.0) + (a - 1.0) * c + sa),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
            a * ((a + 1.0) + (a - 1.0) * c - sa),
            (a + 1.0) - (a - 1.0) * c + sa,
            2.0 * ((a - 1.0) - (a + 1.0) * c),
            (a + 1.0) - (a - 1.0) * c - sa,
        )
    }

    pub fn low_shelf(sr: f64, f: f64, gain_db: f64) -> Self {
        let a = 10f64.powf(gain_db / 40.0);
        let w = std::f64::consts::TAU * (f / sr).min(0.49);
        let (s, c) = w.sin_cos();
        let alpha = s / 2.0 * std::f64::consts::SQRT_2;
        let sa = 2.0 * a.sqrt() * alpha;
        Self::from_coefs(
            a * ((a + 1.0) - (a - 1.0) * c + sa),
            2.0 * a * ((a - 1.0) - (a + 1.0) * c),
            a * ((a + 1.0) - (a - 1.0) * c - sa),
            (a + 1.0) + (a - 1.0) * c + sa,
            -2.0 * ((a - 1.0) + (a + 1.0) * c),
            (a + 1.0) + (a - 1.0) * c - sa,
        )
    }

    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// Gerador pseudo-aleatorio xorshift64* (deterministico por semente: renders offline repetiveis).
#[derive(Clone, Copy)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self { Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1) }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniforme em [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f32 { (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 }

    /// Normal padrao (Box-Muller).
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Processo de Ornstein-Uhlenbeck suavizado: desvio aleatorio com desvio padrao ~`sigma` e tempo
/// de correlacao `tau` (s), passado por um passa-baixa de 3 Hz (o OU puro e aspero em escala
/// curta; a deriva de afinacao/tempo de um cantor real e lisa).
#[derive(Clone, Copy)]
pub struct Drift {
    x: f32,
    y: f32,
    pub sigma: f32,
    pub tau: f32,
}

impl Drift {
    pub fn new(sigma: f32, tau: f32) -> Self { Self { x: 0.0, y: 0.0, sigma, tau } }

    pub fn step(&mut self, dt: f32, rng: &mut Rng) -> f32 {
        let a = (-dt / self.tau).exp();
        self.x = self.x * a + self.sigma * (1.0 - a * a).sqrt() * rng.gauss();
        let b = 1.0 - (-dt * std::f32::consts::TAU * 3.0).exp();
        self.y += (self.x - self.y) * b;
        self.y
    }

    pub fn value(&self) -> f32 { self.y }
}

#[inline]
pub fn hz_to_midi(hz: f32) -> f32 { 69.0 + 12.0 * (hz / 440.0).log2() }

#[inline]
pub fn midi_to_hz(m: f32) -> f32 { 440.0 * ((m - 69.0) / 12.0).exp2() }

#[inline]
pub fn db_to_gain(db: f32) -> f32 { if db <= -59.9 { 0.0 } else { 10f32.powf(db / 20.0) } }

/// Rampa suave 0..1 (smoothstep).
#[inline]
pub fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Tabela de janela de Hann (0.5 + 0.5 cos(pi u), u em [-1, 1]) com interpolacao linear.
pub struct HannTable {
    t: Vec<f32>,
}

impl HannTable {
    const N: usize = 2048;

    pub fn new() -> Self {
        let t = (0..=Self::N + 1)
            .map(|i| {
                let u = (i as f64 / Self::N as f64).min(1.0);
                (0.5 + 0.5 * (std::f64::consts::PI * u).cos()) as f32
            })
            .collect();
        Self { t }
    }

    /// `u` em [-1, 1]; fora disso devolve 0.
    #[inline(always)]
    pub fn w(&self, u: f32) -> f32 {
        let a = u.abs();
        if a >= 1.0 {
            return 0.0;
        }
        let x = a * Self::N as f32;
        let i = x as usize;
        let f = x - i as f32;
        self.t[i] + (self.t[i + 1] - self.t[i]) * f
    }
}

impl Default for HannTable {
    fn default() -> Self { Self::new() }
}
