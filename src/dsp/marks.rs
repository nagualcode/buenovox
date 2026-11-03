//! Marcas de analise sincronas com o periodo (epocas) para o PSOLA (herdado do Heaven).
//!
//! Em trechos vozeados ha uma marca por ciclo glotal. A primeira marca de cada trecho vai no pico
//! de maior amplitude do primeiro ciclo (perto do fechamento glotal). As seguintes sao achadas
//! por correlacao cruzada normalizada com o ciclo anterior, numa janela de +-15% do periodo. Assim
//! todos os graos ficam alinhados na mesma fase da onda e somam de forma coerente: sem
//! cancelamentos, "phasing" ou aspereza.
//!
//! Em trechos sem vozeamento as marcas ficam numa grade fixa de 5 ms.

use super::pitch::PitchTracker;

#[derive(Clone, Copy, Debug, Default)]
pub struct Mark {
    pub pos: f64,
    /// Periodo local em amostras (ou o passo da grade fixa, se nao vozeado).
    pub period: f32,
    pub voiced: bool,
    /// Ciclo de um trecho fracamente periodico (cauda soprada): a harmonia continua, mais baixa.
    pub weak: bool,
}

const CAP: usize = 16384;

pub struct MarkTracker {
    marks: Vec<Mark>,
    count: u64,
    pub uv_hop: f32,
    last: Mark,
    started: bool,
    /// Modo ao vivo: marcas alem do ultimo quadro do detector.
    pub live: bool,
}

impl MarkTracker {
    pub fn new(sr: f32) -> Self {
        let uv_hop = (sr * 0.005).round();
        Self {
            marks: vec![Mark::default(); CAP],
            count: 0,
            uv_hop,
            last: Mark { pos: 0.0, period: uv_hop, voiced: false, weak: false },
            started: false,
            live: false,
        }
    }

    fn push(&mut self, m: Mark) {
        self.marks[(self.count % CAP as u64) as usize] = m;
        self.count += 1;
        self.last = m;
    }

    /// Marca de indice absoluto `k` (valida para `oldest() <= k < count()`).
    pub fn get(&self, k: u64) -> &Mark { &self.marks[(k % CAP as u64) as usize] }

    /// Numero de marcas ja geradas.
    pub fn count(&self) -> u64 { self.count }

    /// Indice da marca mais antiga ainda guardada.
    pub fn oldest(&self) -> u64 { self.count.saturating_sub(CAP as u64 - 8) }

    pub fn latest_pos(&self) -> f64 { if self.count == 0 { f64::NEG_INFINITY } else { self.last.pos } }

    /// Gera marcas enquanto houver sinal e quadros de altura suficientes.
    /// `avail` = numero de amostras ja escritas em `pt.filtered`.
    ///
    /// No modo ao vivo (`live`) as marcas nao esperam o detector (que chega ~30 ms atrasado):
    /// alem do ultimo quadro de altura, o proximo ciclo e previsto pelo espacamento real da marca
    /// anterior e confirmado pela correlacao com o ciclo anterior, que so precisa de ~0,7 periodo
    /// de sinal a frente. Sem correlacao (consoante, fim da nota) o trecho vozeado termina ali.
    /// Inicios de nota so comecam quando o detector confirma o vozeamento.
    pub fn update(&mut self, pt: &PitchTracker, avail: i64) {
        let x = &pt.filtered;
        if !self.started {
            self.push(Mark { pos: 0.0, period: self.uv_hop, voiced: false, weak: false });
            self.started = true;
        }
        loop {
            let last = self.last;
            let step = if last.voiced { last.period as f64 } else { self.uv_hop as f64 };
            let guess = last.pos + step;
            // precisamos de quadros de altura que cubram o proximo ciclo e de sinal para a busca
            // (meio ciclo de folga basta: a latencia minima do motor depende disso)
            let ready = pt.latest_time() >= guess + 0.5 * step;
            if !ready && !self.live {
                break;
            }
            let fr = if ready { pt.at(guess) } else { pt.at(pt.latest_time()) };
            // continuacao fraca: so prolonga um trecho vozeado, nunca inicia um
            let weak = !fr.voiced() && last.voiced && fr.f0_weak > 0.0;
            if fr.voiced() || weak {
                let p_det = pt.sr() / if weak { fr.f0_weak } else { fr.f0 };
                // alem do detector, o espacamento real da marca anterior e mais atual que ele
                let extrapolating = !ready && last.voiced;
                let p = if extrapolating { last.period } else { p_det };
                let ahead = if !last.voiced { 1.0 } else if self.live { 0.7 } else { 1.3 };
                let need = guess + ahead * p as f64 + 2.0;
                if need as i64 >= avail {
                    break;
                }
                let similar = last.voiced && (p - last.period).abs() < 0.35 * p;
                let tracked = if similar { refine_ncc(x, last.pos, guess, p) } else { None };
                // a correlacao falhou com o periodo de sempre (corte abrupto, consoante): o trecho
                // vozeado termina ali, em vez de recomecar num pico qualquer do silencio
                if (weak || (self.live && last.voiced) || similar) && tracked.is_none() {
                    self.push(Mark { pos: guess, period: self.uv_hop, voiced: false, weak: false });
                    continue;
                }
                let pos = tracked.unwrap_or_else(|| {
                    // inicio de trecho vozeado: pico do primeiro ciclo
                    let a = guess.floor() as i64;
                    let b = a + p.ceil() as i64;
                    let mut best = a;
                    let mut bv = -1.0f32;
                    for i in a..b {
                        let v = x.at(i).abs();
                        if v > bv {
                            bv = v;
                            best = i;
                        }
                    }
                    best as f64
                });
                let pos = pos.max(last.pos + 0.5 * p as f64);
                // ao vivo o periodo da marca e o espacamento real (segue glides sem o detector)
                let period = if self.live && tracked.is_some() {
                    let sp = (pos - last.pos) as f32;
                    if (sp / p - 1.0).abs() < 0.25 { sp } else { p }
                } else {
                    p
                };
                self.push(Mark { pos, period, voiced: true, weak });
            } else {
                if guess as i64 + 2 >= avail {
                    break;
                }
                self.push(Mark { pos: guess, period: self.uv_hop, voiced: false, weak: false });
            }
        }
    }

    /// Marca mais proxima de `s` com o periodo ciclo a ciclo em volta dela: `(marca, meia
    /// janela, passo ate a proxima marca)`. A meia janela e a media dos espacamentos vizinhos; o
    /// passo e a distancia real ate a marca seguinte, que mantem as marcas de sintese travadas
    /// nas de analise quando nao ha correcao (e carrega o jitter natural da voz para a saida).
    /// Fora de trecho vozeado, ou sem vizinhas confiaveis, ambos sao o periodo medido. O ultimo
    /// campo diz se a marca seguinte ja existia (senao o passo e uma estimativa).
    pub fn nearest_cycle(&self, s: f64) -> (Mark, f32, f32, bool) {
        if self.count < 3 {
            let m = self.nearest(s);
            return (m, m.period, m.period, false);
        }
        let k = self.nearest_index(s);
        let m = *self.get(k);
        let oldest = self.count.saturating_sub(CAP as u64 - 8);
        if !m.voiced || k <= oldest || k + 1 >= self.count {
            return (m, m.period, m.period, k + 1 < self.count);
        }
        let (a, b) = (*self.get(k - 1), *self.get(k + 1));
        let ok = |d: f32| (d / m.period - 1.0).abs() < 0.25;
        let back = (m.pos - a.pos) as f32;
        let fwd = (b.pos - m.pos) as f32;
        let fwd = if b.voiced && ok(fwd) { fwd } else { m.period };
        let back = if a.voiced && ok(back) { back } else { m.period };
        (m, 0.5 * (back + fwd), fwd, true)
    }

    /// Posicao da primeira marca vozeada depois de `t`, se ja existir (olha poucas marcas).
    pub fn next_voiced_after(&self, t: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let mut k = self.nearest_index(t);
        for _ in 0..4 {
            let m = self.get(k);
            if m.pos > t {
                return m.voiced.then_some(m.pos);
            }
            k += 1;
            if k >= self.count {
                break;
            }
        }
        None
    }

    fn nearest_index(&self, s: f64) -> u64 {
        let last = self.count - 1;
        let oldest = self.count.saturating_sub(CAP as u64 - 8);
        if s >= self.get(last).pos {
            return last;
        }
        if s <= self.get(oldest).pos {
            return oldest;
        }
        let (mut lo, mut hi) = (oldest, last);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.get(mid).pos <= s { lo = mid } else { hi = mid }
        }
        if s - self.get(lo).pos <= self.get(hi).pos - s { lo } else { hi }
    }

    /// Marca mais proxima de `s`.
    pub fn nearest(&self, s: f64) -> Mark {
        if self.count == 0 {
            return Mark { pos: s, period: self.uv_hop, voiced: false, weak: false };
        }
        let last = self.count - 1;
        let oldest = self.count.saturating_sub(CAP as u64 - 8);
        if s >= self.get(last).pos {
            return *self.get(last);
        }
        if s <= self.get(oldest).pos {
            return *self.get(oldest);
        }
        let (mut lo, mut hi) = (oldest, last);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.get(mid).pos <= s { lo = mid } else { hi = mid }
        }
        let (a, b) = (*self.get(lo), *self.get(hi));
        if s - a.pos <= b.pos - s { a } else { b }
    }
}


/// Procura, em +-15% do periodo em torno de `guess`, a posicao cujo ciclo mais se parece com o
/// ciclo em torno de `prev` (correlacao normalizada), com refinamento sub-amostra parabolico.
fn refine_ncc(x: &super::util::Ring, prev: f64, guess: f64, p: f32) -> Option<f64> {
    let half = (p * 0.5).round() as i64;
    let radius = (p * 0.15).ceil().max(1.0) as i64;
    let pc = prev.round() as i64;
    let gc = guess.round() as i64;
    let mut e_t = 0.0f32;
    for j in -half..=half {
        let v = x.at(pc + j);
        e_t += v * v;
    }
    if e_t < 1e-10 {
        return None;
    }
    let mut scores = [0.0f32; 512];
    let n = (2 * radius + 1).min(512) as usize;
    let mut best_k = 0usize;
    let mut best = f32::MIN;
    for k in 0..n {
        let c = gc - radius + k as i64;
        let (mut num, mut e_c) = (0.0f32, 0.0f32);
        for j in -half..=half {
            let v = x.at(c + j);
            num += x.at(pc + j) * v;
            e_c += v * v;
        }
        let s = num / (e_t * e_c).sqrt().max(1e-12);
        scores[k] = s;
        if s > best {
            best = s;
            best_k = k;
        }
    }
    if best < 0.3 {
        return None;
    }
    let mut pos = (gc - radius + best_k as i64) as f64;
    if best_k > 0 && best_k + 1 < n {
        let (a, b, c) = (scores[best_k - 1], scores[best_k], scores[best_k + 1]);
        let den = a - 2.0 * b + c;
        if den.abs() > 1e-9 {
            pos += (0.5 * (a - c) / den).clamp(-0.5, 0.5) as f64;
        }
    }
    // preserva a parte fracionaria da marca anterior relativa a sua posicao arredondada
    Some(pos + (prev - pc as f64))
}
