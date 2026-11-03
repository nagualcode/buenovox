//! Interface do BuenoVOX (egui): plana e escura, na familia visual do Heaven.
//!
//! No centro, o grafico de afinacao rolando: a altura cantada (cinza), a corrigida (cor) e a nota
//! alvo (faixa), sobre as linhas das notas permitidas. Em volta, poucos controles: de onde vem o
//! alvo (TARGET), quanto e como corrigir (CORRECTION), o que preservar ou criar (EXPRESSION) e o
//! ar da voz, que passa ao lado da correcao (AIR).
//! Tudo e parametro do host, entao tudo pode ser automatizado.

use crate::dsp::tonal::{scale_mask, NOTE_NAMES, SCALE_NAMES};
use crate::truce_integration::{BuenoParams, P};
use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use truce::params::ParamInfo;
use truce::prelude::*;
use truce_egui::EditorUi;

pub const WIDTH: u32 = 960;
pub const HEIGHT: u32 = 500;

const BG: Color32 = Color32::from_rgb(28, 28, 30);
const PANEL: Color32 = Color32::from_rgb(40, 40, 42);
const CELL: Color32 = Color32::from_rgb(52, 52, 55);
const CELL_HI: Color32 = Color32::from_rgb(66, 66, 70);
const LINE: Color32 = Color32::from_rgb(70, 70, 74);
const TEXT: Color32 = Color32::from_rgb(226, 226, 226);
const DIM: Color32 = Color32::from_rgb(132, 132, 136);
const DARK_TEXT: Color32 = Color32::from_rgb(20, 20, 20);
/// Dourado quente: a voz corrigida.
const ACCENT: Color32 = Color32::from_rgb(255, 184, 77);
const INPUT: Color32 = Color32::from_rgb(150, 150, 156);
const TARGET: Color32 = Color32::from_rgb(92, 176, 255);
const CLIP: Color32 = Color32::from_rgb(235, 72, 72);

/// Segundos de historico no grafico.
const HISTORY_S: f64 = 6.0;

type Ctx = PluginContext<BuenoParams>;

#[derive(Clone, Copy)]
struct Sample {
    t: f64,
    input: f32,
    output: f32,
    target: f32,
}

pub struct BuenoGui {
    infos: HashMap<u32, ParamInfo>,
    start: Instant,
    history: VecDeque<Sample>,
    center: f32,
}

impl Default for BuenoGui {
    fn default() -> Self { Self { infos: HashMap::new(), start: Instant::now(), history: VecDeque::with_capacity(1024), center: 64.0 } }
}

impl BuenoGui {
    fn ensure_infos(&mut self, state: &Ctx) {
        if self.infos.is_empty() {
            for info in state.params().param_infos() {
                self.infos.insert(info.id, info);
            }
        }
    }

    fn set_plain(&self, state: &Ctx, id: u32, plain: f64) {
        if let Some(info) = self.infos.get(&id) {
            state.automate(id, info.range.normalize(plain));
        }
    }

    fn record(&mut self, state: &Ctx) {
        let t = self.start.elapsed().as_secs_f64();
        if self.history.back().is_some_and(|s| t - s.t < 0.008) {
            return;
        }
        let s = Sample {
            t,
            input: state.get_meter(P::MeterPitchIn),
            output: state.get_meter(P::MeterPitchOut),
            target: state.get_meter(P::MeterTarget),
        };
        self.history.push_back(s);
        while self.history.front().is_some_and(|f| t - f.t > HISTORY_S) {
            self.history.pop_front();
        }
        // o grafico segue a regiao em que se canta (devagar, para nao balancar)
        if s.input > 0.0 {
            self.center += (s.input - self.center) * 0.02;
        }
    }

    // ------------------------------------------------------------------------------ widgets

    /// Knob: arco de 270 graus, rotulo acima e valor abaixo. Arraste vertical (Shift = fino),
    /// duplo clique volta ao padrao.
    fn knob(&self, ui: &mut egui::Ui, state: &Ctx, id: u32, label: &str, bipolar: bool, enabled: bool) {
        let size = Vec2::new(60.0, 64.0);
        let (rect, resp) = ui.allocate_exact_size(size, Sense::click_and_drag());
        let info = self.infos.get(&id).copied();
        let mut value = state.get_param(id);
        let acc_id = ui.make_persistent_id(("bvx-knob", id));
        if resp.drag_started() {
            state.begin_edit(id);
            ui.memory_mut(|m| m.data.insert_temp(acc_id, value));
        }
        if resp.dragged() {
            let fine = ui.input(|i| i.modifiers.shift);
            let delta = -resp.drag_delta().y / if fine { 700.0 } else { 180.0 };
            let prev: f32 = ui.memory(|m| m.data.get_temp(acc_id)).unwrap_or(value);
            let un = (prev + delta).clamp(0.0, 1.0);
            ui.memory_mut(|m| m.data.insert_temp(acc_id, un));
            if (un - value).abs() > 1e-6 {
                value = un;
                state.set_param(id, f64::from(value));
            }
        }
        if resp.drag_stopped() {
            state.end_edit(id);
            ui.memory_mut(|m| m.data.remove_temp::<f32>(acc_id));
        }
        if resp.double_clicked() {
            if let Some(i) = info {
                state.automate(id, i.range.normalize(i.default_plain));
            }
        }
        let p = ui.painter_at(rect.expand(2.0));
        let accent = if enabled { ACCENT } else { DIM };
        p.text(Pos2::new(rect.center().x, rect.top()), Align2::CENTER_TOP, label, FontId::proportional(10.0), DIM);
        let c = Pos2::new(rect.center().x, rect.top() + 31.0);
        let r = 15.0;
        let start = std::f32::consts::FRAC_PI_4 * 3.0;
        let sweep = std::f32::consts::FRAC_PI_2 * 3.0;
        let hovered = resp.hovered() || resp.dragged();
        p.add(egui::Shape::line(arc(c, r, start, sweep), Stroke::new(3.0_f32, if hovered { CELL_HI } else { LINE })));
        let (a0, a1) = if bipolar {
            let mid = start + sweep * 0.5;
            let v = start + sweep * value;
            (mid.min(v), mid.max(v))
        } else {
            (start, start + sweep * value)
        };
        if a1 - a0 > 0.01 {
            p.add(egui::Shape::line(arc(c, r, a0, a1 - a0), Stroke::new(3.0_f32, accent)));
        }
        let ang = start + sweep * value;
        p.line_segment([c, c + Vec2::angled(ang) * (r - 4.0)], Stroke::new(1.8_f32, TEXT));
        p.text(
            Pos2::new(rect.center().x, rect.bottom()),
            Align2::CENTER_BOTTOM,
            state.format_param(id),
            FontId::proportional(10.5),
            if hovered { accent } else { TEXT },
        );
    }

    /// Grade de botoes para um parametro discreto (um botao por valor).
    #[allow(clippy::too_many_arguments)]
    fn grid(&self, ui: &mut egui::Ui, state: &Ctx, id: u32, labels: &[&str], cols: usize, width: f32, cell_h: f32, color: Color32) {
        let Some(info) = self.infos.get(&id).copied() else { return };
        let rows = labels.len().div_ceil(cols);
        let gap = 2.0;
        let (rect, _) = ui.allocate_exact_size(Vec2::new(width, rows as f32 * (cell_h + gap) - gap), Sense::hover());
        let w = (width - gap * (cols as f32 - 1.0)) / cols as f32;
        let cur = state.get_param_plain(id).round() as i64;
        let min = info.range.min().round() as i64;
        for (k, lab) in labels.iter().enumerate() {
            let (r, c) = (k / cols, k % cols);
            let cell = Rect::from_min_size(
                Pos2::new(rect.left() + c as f32 * (w + gap), rect.top() + r as f32 * (cell_h + gap)),
                Vec2::new(w, cell_h),
            );
            let resp = ui.interact(cell, ui.id().with(("grid", id, k)), Sense::click());
            let on = cur == min + k as i64;
            let fill = if on { color } else if resp.hovered() { CELL_HI } else { CELL };
            ui.painter().rect_filled(cell, CornerRadius::same(2), fill);
            ui.painter().text(cell.center(), Align2::CENTER_CENTER, *lab, FontId::proportional(11.0), if on { DARK_TEXT } else { TEXT });
            if resp.clicked() {
                self.set_plain(state, id, (min + k as i64) as f64);
            }
        }
    }

    /// Uma oitava: notas da escala (claras), notas MIDI seguradas (azul), a nota cantada (contorno)
    /// e a nota alvo (cor de destaque).
    fn keyboard(&self, ui: &mut egui::Ui, state: &Ctx, width: f32) {
        let params = state.params();
        let scale = scale_mask(params.key.index() as u8, params.scale.index() as u8);
        let held = state.get_meter(P::MeterMidi) as u16;
        let pc = |m: f32| (m > 0.0).then(|| (m.round() as i32).rem_euclid(12) as usize);
        let sung = pc(state.get_meter(P::MeterPitchIn));
        let target = pc(state.get_meter(P::MeterTarget)).filter(|_| sung.is_some());
        let h = 46.0;
        let (rect, _) = ui.allocate_exact_size(Vec2::new(width, h), Sense::hover());
        let p = ui.painter();
        let ww = width / 7.0;
        let white = [0usize, 2, 4, 5, 7, 9, 11];
        let black = [(1usize, 1.0f32), (3, 2.0), (6, 4.0), (8, 5.0), (10, 6.0)];
        let fill = |n: usize, base: Color32| {
            if target == Some(n) {
                ACCENT
            } else if held >> n & 1 == 1 {
                TARGET
            } else if scale >> n & 1 == 1 {
                base
            } else {
                CELL
            }
        };
        for (i, &n) in white.iter().enumerate() {
            let r = Rect::from_min_size(Pos2::new(rect.left() + i as f32 * ww, rect.top()), Vec2::new(ww - 1.0, h));
            p.rect_filled(r, CornerRadius::same(2), fill(n, Color32::from_rgb(200, 200, 205)));
            if sung == Some(n) {
                p.rect_stroke(r.shrink(1.0), CornerRadius::same(2), Stroke::new(2.0_f32, TEXT), egui::StrokeKind::Inside);
            }
            p.text(Pos2::new(r.center().x, r.bottom() - 3.0), Align2::CENTER_BOTTOM, NOTE_NAMES[n], FontId::proportional(9.0), DARK_TEXT);
        }
        for &(n, x) in &black {
            let r = Rect::from_min_size(Pos2::new(rect.left() + x * ww - ww * 0.3, rect.top()), Vec2::new(ww * 0.6, h * 0.58));
            p.rect_filled(r, CornerRadius::same(2), fill(n, Color32::from_rgb(90, 90, 96)));
            if sung == Some(n) {
                p.rect_stroke(r.shrink(1.0), CornerRadius::same(2), Stroke::new(2.0_f32, TEXT), egui::StrokeKind::Inside);
            }
        }
    }

    /// Grade de latencia: AUTO, LIVE, TRACK, STUDIO (o Auto e o ultimo valor do parametro).
    fn latency_grid(&self, ui: &mut egui::Ui, state: &Ctx, width: f32) {
        let id: u32 = P::Latency.into();
        let cur = state.params().latency.index() as usize;
        let items = [(3usize, "AUTO"), (0, "LIVE"), (1, "TRACK"), (2, "STUDIO")];
        let cw = (width - 3.0 * 2.0) / 4.0;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (v, label) in items {
                let (rect, resp) = ui.allocate_exact_size(Vec2::new(cw, 20.0), Sense::click());
                let on = cur == v;
                let fill = if on { ACCENT } else if resp.hovered() { CELL_HI } else { CELL };
                ui.painter().rect_filled(rect, CornerRadius::same(2), fill);
                ui.painter().text(rect.center(), Align2::CENTER_CENTER, label, FontId::proportional(10.5), if on { DARK_TEXT } else { TEXT });
                if resp.clicked() {
                    self.set_plain(state, id, v as f64);
                }
            }
        });
    }

    fn toggle(&self, ui: &mut egui::Ui, state: &Ctx, id: u32, label: &str, width: f32) {
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(width, 20.0), Sense::click());
        let on = state.get_param(id) > 0.5;
        let fill = if on { ACCENT } else if resp.hovered() { CELL_HI } else { CELL };
        ui.painter().rect_filled(rect, CornerRadius::same(2), fill);
        ui.painter().text(rect.center(), Align2::CENTER_CENTER, label, FontId::proportional(10.5), if on { DARK_TEXT } else { TEXT });
        if resp.clicked() {
            state.automate(id, if on { 0.0 } else { 1.0 });
        }
    }

    fn meter(ui: &mut egui::Ui, state: &Ctx, ids: &[u32], height: f32) {
        let w = 5.0 * ids.len() as f32 + 2.0 * (ids.len() as f32 - 1.0);
        let (rect, _) = ui.allocate_exact_size(Vec2::new(w, height), Sense::hover());
        for (k, &id) in ids.iter().enumerate() {
            let x = rect.left() + k as f32 * 7.0;
            let bar = Rect::from_min_size(Pos2::new(x, rect.top()), Vec2::new(5.0, height));
            ui.painter().rect_filled(bar, CornerRadius::same(1), CELL);
            let pk = state.get_meter(id);
            let db = 20.0 * pk.max(1e-6).log10();
            let v = ((db + 48.0) / 48.0).clamp(0.0, 1.0);
            if v > 0.001 {
                let fill = Rect::from_min_max(Pos2::new(x, bar.bottom() - height * v), bar.max);
                ui.painter().rect_filled(fill, CornerRadius::same(1), if pk > 0.98 { CLIP } else { Color32::from_rgb(110, 220, 120) });
            }
        }
    }

    // ------------------------------------------------------------------------------ secoes

    fn header(&self, ui: &mut egui::Ui, state: &Ctx) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("BUENOVOX").size(16.0).strong().color(TEXT));
            ui.label(egui::RichText::new("natural vocal tuning").size(10.0).color(DIM));
            ui.add_space(12.0);
            self.grid(ui, state, P::Style.into(), &["NATURAL", "CLASSIC", "HARD"], 3, 228.0, 20.0, ACCENT);
            ui.add_space(8.0);
            self.grid(ui, state, P::Source.into(), &["SCALE", "MIDI TARGET", "MIDI SCALE"], 3, 270.0, 20.0, TARGET);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                Self::meter(ui, state, &[P::MeterOut.into()], 18.0);
                ui.label(egui::RichText::new("OUT").size(9.0).color(DIM));
                Self::meter(ui, state, &[P::MeterIn.into()], 18.0);
                ui.label(egui::RichText::new("IN").size(9.0).color(DIM));
                ui.add_space(8.0);
                self.toggle(ui, state, P::Bypass.into(), "BYPASS", 60.0);
            });
        });
    }

    fn target_panel(&self, ui: &mut egui::Ui, state: &Ctx) {
        let w = ui.available_width();
        let midi = state.params().source.index() != 0;
        caption(ui, if midi { "KEY (fallback without MIDI)" } else { "KEY" });
        self.grid(ui, state, P::Key.into(), &NOTE_NAMES, 6, w, 20.0, TARGET);
        ui.add_space(4.0);
        caption(ui, "SCALE");
        self.grid(ui, state, P::Scale.into(), &SCALE_NAMES, 3, w, 20.0, TARGET);
        ui.add_space(6.0);
        self.toggle(ui, state, P::Pure.into(), "PURE TUNING", w);
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            self.knob(ui, state, P::A4.into(), "A4", false, true);
            self.knob(ui, state, P::BendRange.into(), "BEND", false, midi);
        });
        ui.add_space(6.0);
        caption(ui, "NOTES");
        self.keyboard(ui, state, w);
        ui.add_space(6.0);
        caption(ui, "DETECTION");
        self.toggle(ui, state, P::Neural.into(), "NEURAL ASSIST", w);
        hint(ui, "A small network checks octave and voicing.");
    }

    fn graph(&self, ui: &mut egui::Ui, state: &Ctx, rect: Rect) {
        let p = ui.painter_at(rect);
        p.rect_filled(rect, CornerRadius::same(3), Color32::from_rgb(22, 22, 24));
        let params = state.params();
        let mask = match params.source.index() {
            0 => scale_mask(params.key.index() as u8, params.scale.index() as u8),
            _ => {
                let m = state.get_meter(P::MeterMidi) as u16;
                if m != 0 { m } else { scale_mask(params.key.index() as u8, params.scale.index() as u8) }
            }
        };
        let span = 14.0f32;
        let lo = (self.center - span * 0.5).round() - 0.5;
        let y_of = |m: f32| rect.bottom() - (m - lo) / span * rect.height();
        // linhas das notas
        let first = lo.ceil() as i32;
        for n in first..(lo + span).ceil() as i32 {
            let y = y_of(n as f32);
            let allowed = mask & (1 << n.rem_euclid(12)) != 0;
            let lane = Rect::from_min_max(Pos2::new(rect.left(), y_of(n as f32 + 0.5)), Pos2::new(rect.right(), y_of(n as f32 - 0.5)));
            if allowed {
                p.rect_filled(lane, CornerRadius::ZERO, Color32::from_rgb(33, 36, 42));
            }
            p.line_segment([Pos2::new(rect.left() + 30.0, y), Pos2::new(rect.right(), y)], Stroke::new(0.5_f32, if allowed { LINE } else { CELL }));
            p.text(Pos2::new(rect.left() + 4.0, y), Align2::LEFT_CENTER, note_name(n as f32), FontId::proportional(9.0), if allowed { TEXT } else { DIM });
        }
        // curvas
        let Some(last) = self.history.back() else { return };
        let x_of = |t: f64| rect.right() - ((last.t - t) / HISTORY_S) as f32 * (rect.width() - 32.0);
        let draw = |sel: fn(&Sample) -> f32, stroke: Stroke| {
            let mut pts: Vec<Pos2> = Vec::new();
            for s in &self.history {
                let v = sel(s);
                if v > 0.0 && (v - lo) > -1.0 && (v - lo) < span + 1.0 {
                    pts.push(Pos2::new(x_of(s.t), y_of(v)));
                } else if pts.len() > 1 {
                    p.add(egui::Shape::line(std::mem::take(&mut pts), stroke));
                } else {
                    pts.clear();
                }
            }
            if pts.len() > 1 {
                p.add(egui::Shape::line(pts, stroke));
            }
        };
        draw(|s| s.target, Stroke::new(6.0_f32, TARGET.gamma_multiply(0.35)));
        draw(|s| s.input, Stroke::new(1.5_f32, INPUT));
        draw(|s| s.output, Stroke::new(2.0_f32, ACCENT));
        // leitura ao vivo
        let readout = |m: f32| -> String {
            if m <= 0.0 {
                return "—".to_string();
            }
            let n = m.round();
            format!("{} {:+.0}c", note_name(n), 100.0 * (m - n))
        };
        let top = rect.top() + 6.0;
        p.text(Pos2::new(rect.left() + 36.0, top), Align2::LEFT_TOP, format!("IN  {}", readout(last.input)), FontId::monospace(11.0), INPUT);
        p.text(Pos2::new(rect.left() + 150.0, top), Align2::LEFT_TOP, format!("OUT  {}", readout(last.output)), FontId::monospace(11.0), ACCENT);
        let vib = state.get_meter(P::MeterVib);
        p.text(Pos2::new(rect.right() - 8.0, top), Align2::RIGHT_TOP, format!("vibrato {vib:.0}c"), FontId::monospace(10.0), DIM);
    }

    fn correction_panel(&self, ui: &mut egui::Ui, state: &Ctx) {
        let natural = state.params().style.index() == 0;
        let hard = state.params().style.index() == 2;
        ui.horizontal(|ui| {
            self.knob(ui, state, P::Amount.into(), "AMOUNT", false, true);
            self.knob(ui, state, P::Speed.into(), "SPEED", false, !hard);
            self.knob(ui, state, P::Tighten.into(), "TIGHTEN", false, natural);
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            self.knob(ui, state, P::Flex.into(), "FLEX", false, !natural && !hard);
            self.knob(ui, state, P::Humanize.into(), "HUMANIZE", false, !natural && !hard);
            self.knob(ui, state, P::Scoop.into(), "SCOOP", false, natural);
        });
        ui.add_space(6.0);
        caption(ui, "EXPRESSION");
        ui.horizontal(|ui| {
            self.knob(ui, state, P::Vibrato.into(), "VIBRATO", false, !hard);
            self.knob(ui, state, P::VibCreate.into(), "CREATE VIB", false, !hard);
            self.knob(ui, state, P::Throat.into(), "THROAT", true, true);
        });
        ui.add_space(4.0);
        caption(ui, "AIR");
        let split = state.params().breath_split.value();
        ui.horizontal(|ui| {
            self.knob(ui, state, P::Breath.into(), "BREATH", false, split);
            ui.vertical(|ui| {
                ui.add_space(14.0);
                self.toggle(ui, state, P::BreathSplit.into(), "AIR SPLIT", 120.0);
                hint(ui, if split { "Breath bypasses the tuning." } else { "Breath is tuned with the voice." });
            });
        });
        ui.add_space(4.0);
        caption(ui, "LATENCY");
        let w = ui.available_width();
        // a grade mostra AUTO primeiro; o parametro guarda Live=0, Track=1, Studio=2, Auto=3
        self.latency_grid(ui, state, w);
        let chosen = state.params().latency.index() as usize;
        let meter = state.get_meter(P::MeterMode).round() as usize;
        let auto = meter >= 10;
        let active = (meter % 10).checked_sub(1);
        let text = match (chosen, active) {
            (3, Some(0)) if auto => "Auto: Live now (monitoring, 11 ms).".to_string(),
            (3, Some(_)) if auto => "Auto: Studio now (playback/bounce).".to_string(),
            (3, _) => "Auto: Live when monitoring, Studio on playback.".to_string(),
            (c, Some(a)) if c != a => "Restart playback or reload to apply.".to_string(),
            (0, _) => "Live: 11 ms, for singing on stage.".to_string(),
            (1, _) => "Track: for recording with monitoring.".to_string(),
            _ => "Studio: sees each note before tuning it.".to_string(),
        };
        hint(ui, &text);
    }
}

impl EditorUi<BuenoParams> for BuenoGui {
    fn ui(&mut self, ui: &mut egui::Ui, state: &Ctx) {
        self.ensure_infos(state);
        self.record(state);
        ui.ctx().request_repaint();
        egui::Frame::NONE.fill(BG).inner_margin(10).show(ui, |ui| {
            ui.set_min_size(ui.available_size());
            ui.spacing_mut().item_spacing = Vec2::new(6.0, 4.0);
            self.header(ui, state);
            ui.add_space(4.0);
            let (area, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
            let (wl, wr) = (232.0, 214.0);
            let left = Rect::from_min_size(area.min, Vec2::new(wl, area.height()));
            let right = Rect::from_min_size(Pos2::new(area.right() - wr, area.top()), Vec2::new(wr, area.height()));
            let mid = Rect::from_min_max(Pos2::new(left.right() + 8.0, area.top()), Pos2::new(right.left() - 8.0, area.bottom()));
            for (rect, title, k) in [(left, "TARGET", 0), (right, "CORRECTION", 1)] {
                ui.painter().rect_filled(rect, CornerRadius::same(3), PANEL);
                let inner = rect.shrink2(Vec2::new(10.0, 8.0));
                let mut child = ui.new_child(egui::UiBuilder::new().max_rect(inner).layout(egui::Layout::top_down(egui::Align::Min)));
                child.spacing_mut().item_spacing = Vec2::new(6.0, 3.0);
                child.label(egui::RichText::new(title).size(11.0).strong().color(TEXT));
                child.add_space(2.0);
                if k == 0 { self.target_panel(&mut child, state) } else { self.correction_panel(&mut child, state) }
            }
            self.graph(ui, state, mid);
        });
    }
}

fn caption(ui: &mut egui::Ui, text: &str) { ui.label(egui::RichText::new(text).size(9.0).color(DIM)); }

fn hint(ui: &mut egui::Ui, text: &str) { ui.label(egui::RichText::new(text).size(10.0).color(DIM)); }

fn arc(c: Pos2, r: f32, start: f32, sweep: f32) -> Vec<Pos2> {
    let n = ((sweep.abs() / std::f32::consts::TAU) * 64.0).ceil().max(2.0) as usize;
    (0..=n).map(|i| c + Vec2::angled(start + sweep * i as f32 / n as f32) * r).collect()
}

fn note_name(m: f32) -> String {
    let n = m.round() as i32;
    format!("{}{}", NOTE_NAMES[n.rem_euclid(12) as usize], n.div_euclid(12) - 1)
}

pub fn visuals() -> egui::Visuals {
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = PANEL;
    v.extreme_bg_color = BG;
    v.faint_bg_color = PANEL;
    v.override_text_color = Some(TEXT);
    v.window_stroke = Stroke::new(1.0_f32, LINE);
    for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
        w.corner_radius = CornerRadius::same(2);
        w.bg_stroke = Stroke::NONE;
    }
    v.selection.bg_fill = ACCENT.gamma_multiply(0.5);
    v
}
