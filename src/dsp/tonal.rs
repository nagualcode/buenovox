//! Teoria musical do alvo: escalas, notas permitidas, escolha da nota alvo e afinacao justa.
//!
//! Tudo em "MIDI afinado": 69 + 12 log2(f / A4), com o A4 de referencia escolhido pelo usuario.
//! Um inteiro e uma nota temperada exata.

pub const NOTE_NAMES: [&str; 12] = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];

pub const SCALE_NAMES: [&str; 12] = [
    "Chromatic",
    "Major",
    "Minor",
    "Harm. Minor",
    "Mel. Minor",
    "Dorian",
    "Phrygian",
    "Lydian",
    "Mixolydian",
    "Major Pent.",
    "Minor Pent.",
    "Blues",
];

/// Intervalos (semitons a partir da tonica) de cada escala, como mascara de 12 bits.
const SCALES: [&[u8]; 12] = [
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    &[0, 2, 4, 5, 7, 9, 11],
    &[0, 2, 3, 5, 7, 8, 10],
    &[0, 2, 3, 5, 7, 8, 11],
    &[0, 2, 3, 5, 7, 9, 11],
    &[0, 2, 3, 5, 7, 9, 10],
    &[0, 1, 3, 5, 7, 8, 10],
    &[0, 2, 4, 6, 7, 9, 11],
    &[0, 2, 4, 5, 7, 9, 10],
    &[0, 2, 4, 7, 9],
    &[0, 3, 5, 7, 10],
    &[0, 3, 5, 6, 7, 10],
];

/// Mascara de classes de altura absolutas (bit 0 = C) da escala `scale` na tonalidade `key`.
pub fn scale_mask(key: u8, scale: u8) -> u16 {
    let ivs = SCALES[(scale as usize).min(SCALES.len() - 1)];
    ivs.iter().fold(0u16, |m, &iv| m | 1 << ((key as usize + iv as usize) % 12))
}

/// Desvio da afinacao justa (cents) para cada intervalo acima da raiz: razoes de 5-limite
/// (16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 9/5, 15/8).
pub const JUST_CENTS: [f32; 12] = [0.0, 11.73, 3.91, 15.64, -13.69, -1.96, -9.78, 1.96, 13.69, -15.64, 17.6, -11.73];

/// De onde vem o alvo.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Source {
    /// Tonalidade + escala.
    #[default]
    Scale,
    /// Notas MIDI seguradas sao os alvos (o piano roll do DAW e o "modo grafico").
    MidiTarget,
    /// As notas MIDI tocadas definem as classes de altura permitidas (com trava).
    MidiScale,
}

impl Source {
    pub fn from_index(i: usize) -> Self {
        match i {
            1 => Self::MidiTarget,
            2 => Self::MidiScale,
            _ => Self::Scale,
        }
    }
}

/// Contexto tonal num instante da linha do tempo (parametros + estado MIDI).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Ctx {
    pub source: Source,
    pub key: u8,
    pub scale: u8,
    /// Notas MIDI seguradas (128 bits).
    pub held: [u64; 2],
    /// Classes de altura travadas do ultimo acorde tocado (modo MidiScale).
    pub latched: u16,
    /// Pitch bend em semitons.
    pub bend: f32,
    /// Afinacao justa relativa a raiz (nota MIDI mais grave segurada, ou a tonica).
    pub pure: bool,
}

impl Ctx {
    pub fn is_held(&self, n: i32) -> bool {
        (0..128).contains(&n) && self.held[(n >> 6) as usize] & (1u64 << (n & 63)) != 0
    }

    pub fn any_held(&self) -> bool { self.held[0] | self.held[1] != 0 }

    pub fn lowest_held(&self) -> Option<i32> {
        if self.held[0] != 0 {
            Some(self.held[0].trailing_zeros() as i32)
        } else if self.held[1] != 0 {
            Some(64 + self.held[1].trailing_zeros() as i32)
        } else {
            None
        }
    }

    pub fn held_mask(&self) -> u16 {
        let mut m = 0u16;
        for n in 0..128 {
            if self.is_held(n) {
                m |= 1 << (n % 12);
            }
        }
        m
    }

    /// Classes de altura permitidas quando o alvo e por escala (Scale e MidiScale).
    pub fn pc_mask(&self) -> u16 {
        match self.source {
            Source::MidiScale if self.latched != 0 => self.latched,
            _ => scale_mask(self.key, self.scale),
        }
    }

    /// Raiz para a afinacao justa.
    pub fn root(&self) -> i32 {
        match self.source {
            Source::Scale => self.key as i32,
            _ => self.lowest_held().map_or(self.key as i32, |n| n.rem_euclid(12)),
        }
    }

    /// Ajuste fino do alvo (semitons): pitch bend + afinacao justa.
    pub fn fine(&self, note: i32) -> f32 {
        let just = if self.pure { JUST_CENTS[(note - self.root()).rem_euclid(12) as usize] / 100.0 } else { 0.0 };
        self.bend + just
    }
}

/// Nota permitida mais proxima de `c` (MIDI afinado continuo) dentro de `pc` (mascara 12 bits).
fn nearest_in_mask(c: f32, pc: u16) -> Option<i32> {
    if pc == 0 {
        return None;
    }
    let base = c.round() as i32;
    let mut best: Option<(i32, f32)> = None;
    for d in -7..=7 {
        let n = base + d;
        if pc & (1 << n.rem_euclid(12)) != 0 {
            let dist = (n as f32 - c).abs();
            if best.is_none_or(|(_, bd)| dist < bd) {
                best = Some((n, dist));
            }
        }
    }
    best.map(|b| b.0)
}

/// Candidato a alvo vindo de um contexto, ou None se o contexto nao propoe nada.
fn candidate(c: f32, ctx: &Ctx) -> Option<i32> {
    match ctx.source {
        Source::MidiTarget if ctx.any_held() => {
            // nota exata mais proxima; longe demais (cantor noutra oitava), procura a classe
            let mut best: Option<(i32, f32)> = None;
            for n in 0..128 {
                if ctx.is_held(n) {
                    let d = (n as f32 - c).abs();
                    if best.is_none_or(|(_, bd)| d < bd) {
                        best = Some((n, d));
                    }
                }
            }
            match best {
                Some((n, d)) if d <= 7.0 => Some(n),
                _ => nearest_in_mask(c, ctx.held_mask()),
            }
        }
        _ => nearest_in_mask(c, ctx.pc_mask()),
    }
}

/// Escolhe a nota alvo para o centro `c`. `ctxs` sao os contextos vizinhos no tempo (o cantor
/// antecipa ou atrasa trocas de acorde e de nota MIDI): vale o candidato mais proximo entre eles.
/// Com `prev` (alvo atual da mesma nota cantada) ha histerese: so troca se `c` passar do ponto
/// medio entre as notas com folga `hyst` (semitons).
pub fn choose_target(c: f32, ctxs: &[Ctx], prev: Option<i32>, hyst: f32) -> Option<i32> {
    let mut best: Option<(i32, f32)> = None;
    let mut prev_ok = false;
    for ctx in ctxs {
        if let Some(n) = candidate(c, ctx) {
            let d = (n as f32 - c).abs();
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((n, d));
            }
        }
        if let Some(p) = prev {
            prev_ok |= allows(ctx, p);
        }
    }
    let (n, d) = best?;
    match prev {
        Some(p) if prev_ok && p != n && (p as f32 - c).abs() < d + 2.0 * hyst => Some(p),
        _ => Some(n),
    }
}

fn allows(ctx: &Ctx, n: i32) -> bool {
    match ctx.source {
        Source::MidiTarget if ctx.any_held() => ctx.is_held(n) || ctx.held_mask() & (1 << n.rem_euclid(12)) != 0,
        _ => ctx.pc_mask() & (1 << n.rem_euclid(12)) != 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_major_mask() {
        assert_eq!(scale_mask(0, 1), 0b1010_1011_0101);
    }

    #[test]
    fn nearest_scale_note() {
        let ctx = Ctx { source: Source::Scale, key: 0, scale: 1, ..Default::default() };
        // entre C e D (ponto medio 61): decide pelo lado
        assert_eq!(choose_target(60.9, &[ctx], None, 0.1), Some(60));
        assert_eq!(choose_target(61.1, &[ctx], None, 0.1), Some(62));
        // histerese: perto do meio fica na nota anterior
        assert_eq!(choose_target(61.1, &[ctx], Some(60), 0.15), Some(60));
        assert_eq!(choose_target(61.3, &[ctx], Some(60), 0.15), Some(62));
        // E e F (ponto medio 64.5)
        assert_eq!(choose_target(64.6, &[ctx], None, 0.1), Some(65));
    }

    #[test]
    fn midi_target_octave_fallback() {
        let mut ctx = Ctx { source: Source::MidiTarget, ..Default::default() };
        ctx.held[1] |= 1 << (72 - 64);
        assert_eq!(choose_target(71.6, &[ctx], None, 0.1), Some(72));
        // cantando uma oitava abaixo do MIDI: alvo na oitava do cantor
        assert_eq!(choose_target(59.7, &[ctx], None, 0.1), Some(60));
    }

    #[test]
    fn just_third_is_flat() {
        let ctx = Ctx { source: Source::Scale, key: 0, scale: 1, pure: true, ..Default::default() };
        assert!((ctx.fine(64) + 0.1369).abs() < 1e-4);
    }
}
