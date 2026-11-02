//! BuenoVOX: correcao de afinacao vocal natural.
//!
//! O DSP (`dsp`) nao depende do framework e e o mesmo usado pelo CLI e pelos testes; aqui fica
//! a ponte com o truce (parametros automatizaveis, MIDI, medidores e a interface).

pub mod dsp;
#[cfg(feature = "truce-plugin")]
pub mod gui;

#[cfg(feature = "truce-plugin")]
pub mod truce_integration {
    use crate::dsp::{BrainParams, Engine, EngineParams, LatencyMode, Source, Style};
    use std::sync::Arc;
    use truce::prelude::*;

    #[derive(ParamEnum, Debug)]
    pub enum StyleChoice {
        #[name = "Natural"] V0,
        #[name = "Classic"] V1,
        #[name = "Hard"] V2,
    }

    #[derive(ParamEnum, Debug)]
    pub enum SourceChoice {
        #[name = "Scale"] V0,
        #[name = "MIDI Target"] V1,
        #[name = "MIDI Scale"] V2,
    }

    #[derive(ParamEnum, Debug)]
    pub enum KeyChoice {
        #[name = "C"] V0,
        #[name = "C#"] V1,
        #[name = "D"] V2,
        #[name = "Eb"] V3,
        #[name = "E"] V4,
        #[name = "F"] V5,
        #[name = "F#"] V6,
        #[name = "G"] V7,
        #[name = "Ab"] V8,
        #[name = "A"] V9,
        #[name = "Bb"] V10,
        #[name = "B"] V11,
    }

    #[derive(ParamEnum, Debug)]
    pub enum ScaleChoice {
        #[name = "Chromatic"] V0,
        #[name = "Major"] V1,
        #[name = "Minor"] V2,
        #[name = "Harm. Minor"] V3,
        #[name = "Mel. Minor"] V4,
        #[name = "Dorian"] V5,
        #[name = "Phrygian"] V6,
        #[name = "Lydian"] V7,
        #[name = "Mixolydian"] V8,
        #[name = "Major Pent."] V9,
        #[name = "Minor Pent."] V10,
        #[name = "Blues"] V11,
    }

    #[derive(ParamEnum, Debug)]
    pub enum LatencyChoice {
        #[name = "Track"] V0,
        #[name = "Studio"] V1,
    }

    #[derive(Params)]
    pub struct BuenoParams {
        #[param(name = "Style", default = 0)] pub style: EnumParam<StyleChoice>,
        #[param(name = "Source", default = 0)] pub source: EnumParam<SourceChoice>,
        #[param(name = "Key", default = 0)] pub key: EnumParam<KeyChoice>,
        #[param(name = "Scale", default = 0)] pub scale: EnumParam<ScaleChoice>,
        #[param(name = "Amount", range = "linear(0, 1)", default = 1.0, unit = "%")] pub amount: FloatParam,
        #[param(name = "Speed", range = "skewed(0, 400, 0.5)", default = 25.0, unit = "ms")] pub speed: FloatParam,
        #[param(name = "Flex", range = "linear(0, 1)", default = 0.0, unit = "%")] pub flex: FloatParam,
        #[param(name = "Humanize", range = "linear(0, 1)", default = 0.3, unit = "%")] pub humanize: FloatParam,
        #[param(name = "Tighten", range = "linear(0, 1)", default = 0.2, unit = "%")] pub tighten: FloatParam,
        #[param(name = "Vibrato", range = "linear(0, 2)", default = 1.0, unit = "%")] pub vibrato: FloatParam,
        #[param(name = "Create Vibrato", range = "linear(0, 1)", default = 0.0, unit = "%")] pub vib_create: FloatParam,
        #[param(name = "Scoop", range = "linear(0, 1)", default = 0.3, unit = "%")] pub scoop: FloatParam,
        #[param(name = "Throat", range = "linear(-1, 1)", default = 0.0, unit = "%")] pub throat: FloatParam,
        #[param(name = "Pure Tuning", default = false)] pub pure: BoolParam,
        #[param(name = "A4", range = "linear(415, 466)", default = 440.0, unit = "Hz")] pub a4: FloatParam,
        #[param(name = "Bend Range", range = "linear(0, 12)", default = 2)] pub bend_range: IntParam,
        #[param(name = "Latency", default = 1)] pub latency: EnumParam<LatencyChoice>,
        #[param(name = "Bypass", default = false)] pub bypass: BoolParam,

        #[meter] pub meter_in: MeterSlot,
        #[meter] pub meter_out: MeterSlot,
        /// Altura da entrada (MIDI afinado; 0 = sem voz).
        #[meter] pub meter_pitch_in: MeterSlot,
        /// Altura de saida.
        #[meter] pub meter_pitch_out: MeterSlot,
        /// Nota alvo (NaN vira 0).
        #[meter] pub meter_target: MeterSlot,
        /// Vibrato do cantor (cents RMS).
        #[meter] pub meter_vib: MeterSlot,
        /// Notas MIDI seguradas (mascara de classes de altura).
        #[meter] pub meter_midi: MeterSlot,
        /// Modo de latencia ativo + 1 (1 = Track, 2 = Studio; 0 = ainda nao processou).
        #[meter] pub meter_mode: MeterSlot,
    }

    impl BuenoParams {
        pub fn engine_params(&self) -> EngineParams {
            EngineParams {
                brain: BrainParams {
                    style: Style::from_index(self.style.index() as usize),
                    amount: self.amount.read(),
                    speed_ms: self.speed.read(),
                    flex: self.flex.read(),
                    humanize: self.humanize.read(),
                    tighten: self.tighten.read(),
                    vibrato: self.vibrato.read(),
                    vib_create: self.vib_create.read(),
                    scoop: self.scoop.read(),
                    a4: self.a4.read(),
                },
                source: Source::from_index(self.source.index() as usize),
                key: self.key.index() as u8,
                scale: self.scale.index() as u8,
                pure: self.pure.value(),
                bend_range: self.bend_range.value_i32() as f32,
                throat: self.throat.read(),
                bypass: self.bypass.value(),
            }
        }

        pub fn latency_mode(&self) -> LatencyMode { LatencyMode::from_index(self.latency.index() as usize) }
    }

    pub struct BuenoDsp {
        engine: Engine,
        in_l: Vec<f32>,
        in_r: Vec<f32>,
        out_l: Vec<f32>,
        out_r: Vec<f32>,
    }

    impl Default for BuenoDsp {
        fn default() -> Self {
            Self {
                engine: Engine::new(48_000.0, LatencyMode::Studio),
                in_l: vec![0.0; 4096],
                in_r: vec![0.0; 4096],
                out_l: vec![0.0; 4096],
                out_r: vec![0.0; 4096],
            }
        }
    }

    pub struct BuenoVox;

    impl PluginLogic for BuenoVox {
        type Params = BuenoParams;
        type DspState = BuenoDsp;

        fn init(params: &Self::Params, _cx: &InitContext) -> Self::DspState {
            BuenoDsp { engine: Engine::new(48_000.0, params.latency_mode()), ..Default::default() }
        }

        fn reset(state: &mut Self::DspState, params: &Self::Params, config: &AudioConfig) {
            // reconstroi o motor so quando a taxa ou o modo de latencia mudam (reset tambem
            // chega em play/stop); a latencia nova vale a partir daqui
            let mode = params.latency_mode();
            if (state.engine.sample_rate() - config.sample_rate as f32).abs() > 0.5 || state.engine.mode() != mode {
                state.engine = Engine::new(config.sample_rate as f32, mode);
            }
            let max_blk = config.max_block_size.max(4096);
            for v in [&mut state.in_l, &mut state.in_r, &mut state.out_l, &mut state.out_r] {
                v.resize(max_blk, 0.0);
            }
        }

        fn latency(state: &Self::DspState) -> u32 { state.engine.latency() as u32 }

        fn process(
            state: &mut Self::DspState,
            params: &Self::Params,
            buffer: &mut AudioBuffer,
            events: &EventList,
            ctx: &mut ProcessContext,
        ) -> ProcessStatus {
            let n = buffer.num_samples();
            state.engine.set_params(&params.engine_params());
            // MIDI: os eventos entram na linha do tempo com o deslocamento exato no bloco
            for event in events.iter() {
                let at = (event.sample_offset as usize).min(n);
                match &event.body {
                    EventBody::NoteOn { note, velocity, .. } => {
                        if *velocity > 0 {
                            state.engine.note_on(at, *note);
                        } else {
                            state.engine.note_off(at, *note);
                        }
                    }
                    EventBody::NoteOff { note, .. } => state.engine.note_off(at, *note),
                    EventBody::PitchBend { value, .. } => {
                        state.engine.pitch_bend(at, (f32::from((*value).min(16383)) - 8192.0) / 8192.0);
                    }
                    EventBody::ControlChange { cc, .. } if *cc == 123 || *cc == 120 => state.engine.all_notes_off(at),
                    _ => {}
                }
            }

            let mut peaks = [0.0f32; 2];
            let mut pos = 0;
            while pos < n {
                let m = (n - pos).min(state.in_l.len());
                let range = pos..pos + m;
                let ch_in = buffer.num_input_channels();
                if ch_in >= 2 {
                    state.in_l[..m].copy_from_slice(&buffer.input(0)[range.clone()]);
                    state.in_r[..m].copy_from_slice(&buffer.input(1)[range.clone()]);
                } else if ch_in == 1 {
                    state.in_l[..m].copy_from_slice(&buffer.input(0)[range.clone()]);
                    state.in_r[..m].copy_from_slice(&buffer.input(0)[range.clone()]);
                } else {
                    state.in_l[..m].fill(0.0);
                    state.in_r[..m].fill(0.0);
                }
                let (il, ir, ol, or) = (&state.in_l[..m], &state.in_r[..m], &mut state.out_l[..m], &mut state.out_r[..m]);
                state.engine.process(il, ir, ol, or);
                let pk = |s: &[f32]| s.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                peaks[0] = peaks[0].max(pk(il).max(pk(ir)));
                peaks[1] = peaks[1].max(pk(ol).max(pk(or)));
                let ch_out = buffer.num_output_channels();
                if ch_out >= 2 {
                    buffer.output(0)[range.clone()].copy_from_slice(&state.out_l[..m]);
                    buffer.output(1)[range].copy_from_slice(&state.out_r[..m]);
                } else if ch_out == 1 {
                    for (o, (&l, &r)) in buffer.output(0)[range].iter_mut().zip(state.out_l[..m].iter().zip(&state.out_r[..m])) {
                        *o = 0.5 * (l + r);
                    }
                }
                pos += m;
            }

            let mt = state.engine.meter;
            ctx.set_meter(P::MeterIn, peaks[0]);
            ctx.set_meter(P::MeterOut, peaks[1]);
            ctx.set_meter(P::MeterPitchIn, mt.input);
            ctx.set_meter(P::MeterPitchOut, mt.output);
            ctx.set_meter(P::MeterTarget, if mt.target.is_finite() && mt.input > 0.0 { mt.target } else { 0.0 });
            ctx.set_meter(P::MeterVib, mt.vib_cents);
            ctx.set_meter(P::MeterMidi, state.engine.held_mask() as f32);
            ctx.set_meter(P::MeterMode, state.engine.mode().index() as f32 + 1.0);
            ProcessStatus::Normal
        }

        fn editor(params: Arc<Self::Params>) -> Box<dyn Editor> {
            Box::new(
                truce_egui::EguiEditor::with_ui(params, (crate::gui::WIDTH, crate::gui::HEIGHT), crate::gui::BuenoGui::default())
                    .with_visuals(crate::gui::visuals())
                    .resizable(false),
            )
        }
    }

    pub use BuenoParamsParamId as P;

    truce::plugin! { logic: BuenoVox, params: BuenoParams }
}
