# BuenoVOX: plano de desenvolvimento

> Correção automática de afinação vocal (auto + entrada MIDI) em Rust, com o framework
> [truce](https://github.com/truce-audio/truce). Projeto de pesquisa, uso pessoal e estudo,
> sem fins comerciais.
>
> Documento vivo. Versão 0.1, 2026-11-01.

## Status (2026-11-03)

| Fase | Situação |
|---|---|
| F0 Fundação e régua | Crate, CLI e régua em Rust (cantor sintético de f0 conhecido, 17 testes). Falta a régua Python com datasets públicos e renders dos concorrentes. |
| F1 Análise | Detector e marcas do Heaven portados. Erro mediano de 0,27 cent no sintético. Árbitro neural (CNN de 4,8 mil parâmetros sobre periodicidade do YIN + espectro empilhado por harmônicos, treinada em vozes sintéticas fonte-filtro + 2,6 h do VocalSet rotulado pelo CREPE): no vocadito (anotação humana) erros de oitava 1,03% → 0,39%, grosseiros 3,23% → 2,63%, voz falsa 21,1% → 18,9%, sem perder voz. Sozinha a rede acerta 97,9% das alturas (o YIN 94,3%), mas sem vale do CMND perto dela a precisão fina se perde: próximo passo, usar a altura da rede onde o YIN não tem vale. |
| F2 Síntese | PSOLA por razão: preserva o jitter e trava as marcas. Meias janelas assimétricas (distância real até as marcas vizinhas): sem correção, 87 dB (Studio) e 98 dB (Live) contra o seco. Separação harmônico + ar (média de ciclos realinhados, corte FIR de fase linear em 1 kHz): fantasma −24 dB, ar que fica na parte corrigida −13 dB (Studio) numa voz com HNR de 6 dB. Falta a pré-distorção cepstral e refazer o pulsar do ar no ritmo dos ciclos novos. |
| F3 Cérebro v1 | Classic e Hard, MIDI Target, MIDI Scale, pitch bend, entonação pura. |
| F4 Cérebro v2 | Natural: segmentação por movimento, centro curto, altura percebida e deslocamento por nota. Erro de centro ≤ 2,1 cents e expressão ≤ 1,5 cent no sintético. Em canto real (vocadito, 1.339 notas anotadas como alvo): erro mediano 5,8 cents no Studio e 11,6 no Live (o cantor erra 25). Transições medidas em 2.454 passos do VocalSet (overshoot 15–19% em passos pequenos, ζ 0,47–0,63) e usadas no retune do Classic. Centro percebido e prior melódico medidos e deixados de lado (as anotações guardam a média e a nota cantada, não a percepção nem a intenção). Falta o MUSHRA. |
| F5+ Latência | Live em 14 ms: leitura com deslocamento contínuo (sem correção a saída é a entrada, 145 dB em voz real), passagem curta entre grãos onde o sinal existe, rampa na troca de receita da separação; vozes abaixo de 100 Hz passam limpas sem correção. Os 11 ms da 0.3 chiavam em voz grave real: foram validados só no sintético. Modo Auto (padrão): Live em tempo real, Studio só quando o host prepara um bounce. Presets de fábrica, teclado de notas na interface, transcrição para MIDI no CLI. |
| F5 Plugin | AU, VST3 e CLAP com gráfico de afinação. auval e pluginval passam. O clap-validator falha nos 3 testes de reprodutibilidade de estado, mas os exemplos oficiais do truce 6.3 (Gain, SysEx Echo) falham nos mesmos 3: é do framework. |
| F6–F8 | Não iniciadas. |

Diferenças em relação à §3.6: por enquanto um crate só (`src/dsp/` sem dependência de
framework, como no Heaven) em vez de um workspace.

Latências: **Live 14 ms**, Track ~105 ms, Studio ~270 ms. No Live:
* as marcas de época são previstas pelo espaçamento real do ciclo anterior e confirmadas por
  correlação;
* os grãos ficam limitados a 5 ms;
* as decisões de nota chegam ~30 ms depois do som.

Natural/Live: erro de centro ≤ 3,7 cents e expressão ≤ 4,9 cents no sintético; 10,9 cents de erro mediano em canto real.

---

## 0. A tese em uma frase

O ouvinte chora com a voz, não com a afinação. A afinação certa só remove a distração.
Então o BuenoVOX **corrige só o centro percebido de cada nota e devolve todo o resto intacto**:
o vibrato, o ataque, a respiração, o micro-tremor e a garganta do cantor. Quando algo precisa ser
*criado* (uma transição entre notas, um vibrato numa nota reta), ele segue modelos medidos em
cantores de verdade, nunca uma rampa exponencial.

Os três caminhos de mercado deixaram três lacunas:

| Família | Exemplo | O que faz bem | A lacuna que exploramos |
|---|---|---|---|
| Tempo real por quantização | Auto-Tune (Auto Mode), Waves Tune RT, Graillon, x42-autotune | Latência baixa, efeito "hard tune" icônico | Corrige o *f0 instantâneo*, não a nota. Achata vibrato e cria transições robóticas. Só em set/2026 a Antares lançou o DVA, uma camada que detecta e preserva vibrato. |
| Edição offline por notas | Melodyne, Revoice Pro, VariAudio, Flex Pitch | Corrige a nota como objeto e preserva o vibrato | Exige trabalho manual e ARA ou captura. Não funciona ao vivo nem "solta". |
| Neural / acadêmico | Deep Autotuner, KaraTuner, Diff-Pitcher, Neurodyne, BERT-APC | Infere a nota pretendida pelo contexto musical. O BERT-APC supera Auto-Tune e Melodyne em MOS de afinação (4,32 contra 3,22 e 3,08). | Modelos pesados (diffusion, vocoder neural, transformer) sem latência de plugin. Pouca ou nenhuma preocupação com tempo real. |

**O BuenoVOX junta os três:** a correção *por nota* do Melodyne, feita automaticamente com
lookahead, guiada por MIDI e por contexto musical, num motor de síntese de tempo real cuja
naturalidade é **medida**, não presumida.

---

## 1. O que aprendemos com o que já existe

### 1.1 Heaven (nosso harmonizador): a fundação

O Heaven já resolveu metade do problema, e com qualidade medida. **O BuenoVOX parte do código
e da metodologia dele.**

* **Detector de f0:** YIN via FFT a cada 4 ms, com 32 candidatos, escolha por continuidade,
  correção de sub-harmônico, portão de tessitura e mediana de 3 quadros. Erro mediano de
  **0,6 cent** num soprano sintético. Os erros de oitava em voz creaky, soprano agudo e
  transições já foram caçados e corrigidos.
* **Marcas de época** sincronizadas ao ciclo glotal (pico do primeiro ciclo e depois correlação
  cruzada normalizada).
* **TD-PSOLA:** grãos de 2 períodos com janela de Hann, interpolação cúbica e normalização de
  potência. Erro de envelope **< 1 dB entre −12 e +12 semitons**.
* **Lookahead com latência reportada** (80 ms) e eventos MIDI e de automação com carimbo de
  amostra.
* **Separação DSP / framework:** `dsp/` não depende do truce e é compartilhado pelo plugin, pelo
  CLI (`heaven_cli`) e pelos testes.
* **Metodologia "uma régua no lugar do ouvido":** quem escreve o DSP (Claude) não ouve, então
  tudo é medido. Os sinais sintéticos têm f0 conhecido, e os concorrentes foram renderizados via
  `pedalboard` para comparação. Isso continua igual no BuenoVOX.
* **Integração truce 6.3:** `#[derive(Params)]`, `MeterSlot`, egui, `midi_input = true`,
  `cargo truce install --au2 --vst3`, auval e pluginval.

O que o Heaven *não* precisou ter e o BuenoVOX precisa:

1. Correção de afinação *do próprio lead*, que é muito mais exposta que uma harmonia.
2. Segmentação de notas e decomposição da expressão.
3. Modo de baixa latência para gravar e tocar ao vivo.

### 1.2 OpenVoxTuner (C++/JUCE, AGPL-3.0)

* **Pipeline:** NoiseGate → YIN → ScaleQuantizer → RetargetEnvelope (IIR de 1ª ordem) → PSOLA →
  Harmony → Reverb. Latência de 8 a 40 ms, ARA2 e modo gráfico com curva desenhada.
* **A lição mais valiosa está no relatório de formantes deles.** Começaram com 4 filtros
  peaking em frequências fixas (500/1500/2500/3500 Hz, iguais para qualquer voz) e compensação
  `1/sqrt(r)`. Descobriram que isso é uma aproximação ruim e migraram para síntese cruzada por
  LPC. Também registram uma armadilha de sinal: a excitação é `e = x + Σ a_k x[n−k]`.
  **Nós não vamos repetir esse caminho:** com PSOLA a formante já é preservada pelo grão, e para
  shifts grandes usaremos envelope verdadeiro cepstral (§3.3).
* **Retune por IIR de 1ª ordem** é exatamente o que soa robótico (§3.4).
* **Licença AGPL:** podemos aprender com as ideias, mas **não copiar código** para um projeto
  MIT/Apache.

### 1.3 mutagen-v1

Protótipo web (React + Web Audio). YIN ingênuo O(N²) com limiar fixo de 0,15, sem marcas de
época nem tratamento de oitava. Serve como referência de **interface** (Retune, Humanize,
Flex-Tune, Vibrato, gráfico de pitch), não de DSP. Confirma o vocabulário de controles que o
usuário espera.

### 1.4 Outros projetos abertos que vale ler

* **Autotalent** (Tom Baran, GPL) e **x42-autotune / zita-at1** (Fons Adriaensen, GPL): os
  clássicos abertos de tempo real. Leitura de referência; GPL, então nada de copiar código.
* **Signalsmith Stretch** (MIT, C++, com crates Rust `signalsmith-stretch` e `ssstretch`):
  phase vocoder de alta qualidade com compensação de formantes. Útil como *baseline* de
  comparação e como motor para material polifônico ou ruidoso. O próprio autor diz que a
  formante dele é menos precisa que PSOLA em voz monofônica.
* **Rubber Band R3** (GPL): referência de qualidade, não incorporável.
* **Diff-Pitcher** (PyTorch, com checkpoints no Hugging Face) e **Neurodyne**: referências
  neurais para comparação offline.

### 1.5 SeqGAN: o que serve e o que não serve

SeqGAN (Yu et al., AAAI 2017) treina um gerador de **sequências discretas** (tokens) com um
discriminador cuja nota vira recompensa via *policy gradient* com rollouts de Monte Carlo. Ele
resolve um problema que **não temos**: o f0 é contínuo e diferenciável, e o REINFORCE é instável
e lento. Desde 2017 a área também evoluiu para diffusion, flow matching e transformers.

**A ideia que aproveitamos:** um *discriminador de naturalidade*. Na fase de pesquisa (F7)
treinamos um pequeno modelo que gera contornos de f0 (transições e vibrato) condicionados à nota,
com perda de reconstrução **mais perda adversarial** de um discriminador que distingue contornos
reais de gerados. Como o sinal é contínuo, o gradiente passa direto, sem policy gradient. Esse
discriminador também vira **métrica**: um "detector de auto-tune" treinado em vozes reais contra
vozes corrigidas. Se ele não conseguir distinguir a saída do BuenoVOX de uma voz real, ganhamos.

---

## 2. Estado da arte: pesquisa e patentes

### 2.1 Detecção de f0

| Método | Tipo | Notas para nós |
|---|---|---|
| YIN (de Cheveigné & Kawahara, 2002) e pYIN | DSP | Base do Heaven. Precisa, leve, determinística. |
| Hildebrand / Auto-Tune (US 5,973,252) | DSP | Autocorrelação com redução de taxa e truques de custo. Patente de 1998, **expirada** (~2018). |
| CREPE | Neural | Precisa mas pesada para plugin. |
| **PESTO** (Sony CSL, ISMIR 2023 / TISMIR 2025) | Neural, auto-supervisionado | **130 k parâmetros, latência < 10 ms**, VQT em streaming com convoluções em cache. Candidata forte a árbitro neural. |
| **SwiftF0** (2025) | Neural | **~96 k parâmetros**, ~42× mais rápida que CREPE em CPU. |
| FCPE (2025) e RMVPE | Neural | 10 M e 90 M parâmetros. A RMVPE extrai a voz de mixagem polifônica. Grandes demais para o caminho de tempo real; servem para gerar rótulos offline. |

**Decisão:** detector **híbrido**. O YIN do Heaven dá a precisão em cents. Uma rede minúscula
(PESTO ou SwiftF0, ~100 k parâmetros) roda em paralelo como **árbitro de oitava e de vozeamento**
e como segunda opinião em fry, soprosidade e belting.

### 2.2 Síntese (mudar a altura sem mudar a pessoa)

* **TD-PSOLA** (Moulines & Charpentier, 1990; Lent, 1989): transparente para os shifts pequenos
  da correção (tipicamente < 1 semitom). Preserva formantes por construção. Já provado no Heaven.
* **Envelope verdadeiro cepstral** (Röbel & Rodet, DAFx 2005, com versão de tempo real): o
  melhor estimador clássico de envelope para pré-distorcer formantes em shifts grandes e para o
  controle de "garganta".
* **Modelo harmônico + ruído** (HNM / WORLD / STRAIGHT): separa a parte periódica da aspiração.
  Permite **regenerar vibrato** e mexer na altura sem "zumbir" a respiração.
* **Neural:**
  * O plug-in de conversão de voz cantada em tempo real do DAFx 2023 combina *pitch shift
    clássico + filtro neural leve*, com < 60 ms de latência e 20× o tempo real em laptop. O
    artigo argumenta que vocoders neurais e paramétricos erram o pitch e não preservam o contorno
    relativo. **Esse é o padrão arquitetural certo para nós.**
  * Restauração por *shallow diffusion* auto-supervisionada (arXiv 2601.10345, jan/2026):
    desloca e volta com WORLD para criar pares sem rótulo. Ótima receita de dados, pesada demais
    para tempo real.
  * Diff-Pitcher (WASPAA 2023), Neurodyne (Interspeech 2025), CLPCNet: referências de qualidade
    offline.

### 2.3 Para onde vai a afinação (o "cérebro")

* **Deep Autotuner** (Wager et al., ICASSP 2020): CNN + GRU que prevê o shift por nota a partir
  da voz e do acompanhamento. Tem o dataset *Intonation*: 4.702 performances de karaokê com boa
  afinação.
* **KaraTuner** (Interspeech 2022) e **Diff-Pitcher**: preditor de contorno a partir da
  partitura ou de um template.
* **BERT-APC** (nov/2025): segmenta notas, estima o *pitch estacionário percebido* (média
  ponderada aprendida, MAE de 3,5 cents) e infere a nota pretendida com MusicBERT pelo contexto
  harmônico e melódico. Aplica o erro **uniformemente na nota**, o que preserva vibrato e
  glides. A acurácia em vozes muito desafinadas sobe de 71% (arredondar) para 89% (contexto).
  **Isso é exatamente o "Natural Mode" que queremos, e nós temos uma vantagem: a entrada MIDI
  dá o contexto de graça.**
* **Modelo de f0 de Saitou et al.** (2005–2007): o contorno cantado = notas + 4 flutuações:
  * *overshoot* (passar do alvo na troca de nota): sistema de 2ª ordem amortecido;
  * *preparation* (desvio contrário logo antes da troca): também 2ª ordem amortecido;
  * *vibrato* (4–7 Hz): oscilador de 2ª ordem;
  * *fine fluctuation* (> 10 Hz): ruído branco filtrado.

  **O overshoot é o que mais pesa na percepção.** É a física que o nosso retune vai imitar.
* **Percepção** (Larrouy-Maestri et al.): leigos toleram 25–40 cents de desafinação e
  especialistas ~10 cents. O especialista ouve o *intervalo*, não o contorno absoluto. *Glides*
  longos fazem a melodia soar desafinada. Daí:
  * a meta do corpo da nota é ±5 cents do alvo;
  * corrigir os intervalos importa mais que a afinação absoluta;
  * scoops excessivos devem ser *encurtados*, não apagados.
* **Coro e humanização** (Ternström, herdado do Heaven): dispersão preferida de 0–5 cents,
  tolerada até ~14.

### 2.4 Mercado em 2026

* **Antares AutoTune Advanced** (set/2026): Auto-Mode reconstruído e o **DVA**, motor *patent
  pending* que detecta o vibrato e o deixa passar sem correção mesmo com retune máximo, mais
  geração de vibrato em notas retas. O **Pro 11** mantém Retune Speed, Flex-Tune, Humanize,
  Natural Vibrato, harmonias por MIDI e ARA2. Isso confirma a direção: o mercado acabou de
  descobrir que preservar vibrato é o diferencial. Nós vamos além e preservamos *todas* as
  flutuações, não só o vibrato.
* **Celemony Melodyne:** notas como objetos. A patente **US 8,022,286** (DNA, prioridade em
  2008) é sobre a separação *polifônica* em objetos de nota. Para voz monofônica não precisamos
  dela.

### 2.5 Patentes: mapa e postura

| Patente | Titular | Assunto | Situação |
|---|---|---|---|
| US 5,973,252 | Antares (Hildebrand) | Detecção de pitch e correção de entonação | Expirada (~2018) |
| US 5,231,671 / US 5,567,901 | IVL / DigiTech | Harmonia em PSOLA, histerese, sibilantes | Expiradas |
| US 8,168,877 | (citada pelo Heaven) | Harmonia a partir de acompanhamento polifônico | Verificar |
| US 8,022,286 | Celemony | Objetos de nota em gravação polifônica | Provavelmente ativa até ~2029 |
| DVA | Antares | Detecção e preservação de vibrato | *Patent pending* (2026) |

**Postura:** o projeto é de pesquisa, pessoal e sem fins comerciais. No Brasil, a Lei 9.279/96
(art. 43, I e II) exclui do direito do titular os atos privados sem finalidade comercial e os
atos com finalidade experimental ligados a estudo ou pesquisa. Mesmo assim:

* projetamos a partir da **literatura acadêmica publicada** (PSOLA, YIN, Saitou, Röbel, BERT-APC);
* **não reproduzimos reivindicações** de patentes ativas;
* documentamos de onde vem cada ideia, como o Heaven fez.

A licença MIT-0 do código não concede direitos de patente nem muda esta postura: quem usar o
projeto comercialmente responde por essa análise. Se um dia o objetivo mudar, isso precisa de
revisão jurídica de verdade.

---

## 3. Arquitetura do BuenoVOX

### 3.1 Visão geral

```
                ┌──────────────────── ANÁLISE ────────────────────┐
 áudio ──► pré  │ f0 híbrido (YIN + árbitro neural) · vozeamento   │
 (mono)   (HP,  │ marcas de época (GCI) · sibilância/ruído ·       │
          gate) │ envelope (cepstral) · nível                      │
                └───────────────┬─────────────────────────────────┘
                                ▼
 MIDI ──────►   ┌──────────────────── CÉREBRO ────────────────────┐
 (notas, PB,    │ segmentação de notas (lookahead)                 │
  acordes,      │ decomposição: centro + vibrato + overshoot +     │
  CC)           │   preparation + deriva + flutuação fina          │
 tonalidade ─►  │ escolha da nota alvo (MIDI > contexto > escala)  │
                │ política de correção (Classic / Natural / Hard)  │
                │ curva alvo de f0 = centro corrigido + expressão  │
                └───────────────┬─────────────────────────────────┘
                                ▼ razão de pitch por época
                ┌──────────────────── SÍNTESE ────────────────────┐
                │ Motor A: TD-PSOLA (Heaven++)                     │
                │ Motor B: harmônico + ruído (modo Studio)         │
                │ formantes: envelope verdadeiro (shifts grandes,  │
                │   controle de garganta) · ruído/sibilantes sem   │
                │   shift · pós-filtro neural (pesquisa)           │
                └───────────────┬─────────────────────────────────┘
                                ▼
                            saída + medidores + curvas para a GUI
```

### 3.2 Latência: três modos honestos

O lookahead é o que separa "corrigir o f0" de "corrigir a nota". Cada modo é uma latência fixa
reportada ao host pela PDC (`fn latency()` do truce):

| Modo | Latência | O que sabe | Uso |
|---|---|---|---|
| **Live** | ~10–15 ms (1–2 períodos + quadro de análise) | Só o passado | Monitorar e tocar ao vivo. Correção estilo "Classic" melhorada. |
| **Track** | ~40 ms | O início da nota | Gravar com monitoração tolerável. Vê o ataque antes de decidir. |
| **Studio** | ~150–250 ms | A nota inteira ou quase | Mixagem. Correção por nota, a máxima naturalidade. |

O modo é escolhido por instância. Trocar latência no meio da sessão é mal suportado por vários
hosts, então a GUI avisa que pode ser preciso recarregar.

> **Risco:** o truce não expõe ARA. Por isso a edição "estilo Melodyne" (ver a faixa inteira
> antes de tocar) não existe no v1. A resposta é o modo Studio (lookahead longo) e a ideia da
> §3.5: **o clipe MIDI do DAW é o nosso editor gráfico**.

### 3.3 Síntese: escolha de tecnologia

1. **Motor A, TD-PSOLA**, herdado do Heaven e melhorado:
   * shift zero bit-transparente (null test no CI);
   * grãos com marcas de época refinadas;
   * fallback suave nos trechos não vozeados (passa sem shift, com crossfade);
   * **componente de ruído separado:** a aspiração de vozes soprosas é estimada (resíduo após a
     reconstrução harmônica) e não passa pelo PSOLA, o que evita o "zumbido" em vozes breathy.
2. **Formantes:** com PSOLA, shifts pequenos não precisam de correção. Para hard tune por MIDI
   (shifts grandes) e para o controle **Throat / Character**, aplicamos a pré-distorção por
   envelope verdadeiro cepstral (Röbel). Nada de filtros peaking fixos (a lição do
   OpenVoxTuner).
3. **Motor B, harmônico + ruído**, só no modo Studio: análise senoidal por harmônico sincronizada
   ao f0, com fases coerentes entre harmônicos, e ruído modelado à parte. Permite regenerar ou
   remodelar vibrato com mais limpeza que PSOLA e preservar jitter e shimmer naturais.
4. **Pós-filtro neural** (pesquisa, F7): rede causal minúscula treinada auto-supervisionada.
   Desloca e volta com o *nosso próprio* PSOLA, como na receita da shallow diffusion, e aprende a
   remover o artefato. Só entra se vencer no teste cego.

### 3.4 Cérebro: o coração emocional

**a) Segmentação de notas.** Com o f0 e o nível em lookahead, o cérebro marca para cada nota o
onset, o ataque/transição, o corpo estável e a liberação. Usa a histerese do Heaven contra o
vibrato (0,2 st; nota "assentada" após ~100 ms) e um decodificador de Viterbi com atraso fixo
(HMM de notas, como no pYIN-notes) para não oscilar entre candidatos.

**b) Decomposição da expressão.** O f0 de cada nota se divide em:

* **centro percebido:** média ponderada estilo SPP do BERT-APC. A primeira versão usa pesos
  heurísticos (estabilidade, nível, vozeamento); depois, pesos aprendidos;
* **vibrato:** taxa, extensão e fase estimadas localmente, numa banda de 4–8 Hz;
* **overshoot e preparation:** ajuste de um sistema de 2ª ordem (Saitou) na transição;
* **deriva lenta:** < 2 Hz, o "afundar" de fim de frase;
* **flutuação fina:** > 10 Hz, o que faz a voz parecer viva.

**c) Escolha da nota alvo**, em ordem de prioridade:

1. **MIDI segurada** (modo *MIDI Target*): a nota tocada é o alvo, com janela de antecipação ou
   atraso (o cantor raramente acerta o grid; o Heaven usou 80 ms).
2. **Contexto MIDI** (modo *MIDI Chord / Scale*): as notas seguradas definem o conjunto
   permitido.
3. **Tonalidade + escala**, com Auto-Key opcional numa tarefa de fundo do truce.
4. **Contexto melódico** (pesquisa): o prior "BERT-APC leve" decide quando o cantor está entre
   duas notas, pelo intervalo e pela harmonia.

**d) Política de correção.** Em vez de "retune speed" como filtro de 1ª ordem:

* **Natural** (padrão): o cérebro desloca *só o centro* da nota para o alvo, como um offset
  uniforme. Vibrato, deriva e flutuação fina passam intactos. Nas transições, a curva entre a
  nota antiga e a nova é **regenerada com o modelo de Saitou** (preparation + overshoot
  amortecido), com parâmetros vindos da própria transição do cantor e escalados. Scoops e quedas
  longas são *encurtados* conforme um controle "Scoop", nunca apagados.
* **Classic:** quantização tradicional com Retune Speed, Flex-Tune (só corrige perto da nota) e
  Humanize (notas longas menos travadas). Retune modelado como sistema de **2ª ordem** com
  amortecimento ajustável, porque o de 1ª ordem é o que soa robótico.
* **Hard:** retune instantâneo, o "efeito" icônico, feito com capricho: troca na marca de época
  e fase coerente, sem estalo.
* **Vibrato:** *Preserve* (padrão), *Reduce / Enhance* (escala a extensão) ou *Create* (gera
  vibrato nas notas retas, com atraso de entrada e crescimento natural de extensão e taxa,
  conforme as medições).
* **Entonação:**
  * *Equal* (padrão);
  * *Pure*: afinação justa relativa ao acorde MIDI, herdada do Heaven;
  * *Expressive* (experimental): sensíveis um pouco mais agudas, como cantores treinados fazem.

**e) Curva final:** `f0_alvo(t) = centro_corrigido(t) + transição_modelada(t) + k_v·vibrato(t)
+ deriva(t) + flutuação(t)`. A síntese recebe a razão `f0_alvo / f0_medido` em cada época.

### 3.5 Entrada MIDI: o MIDI como editor gráfico

Sem ARA, o MIDI do DAW faz o papel do editor gráfico do Melodyne:

* **Notas** = alvos (piano roll = "graph mode").
* **Pitch bend** = desvio fino do alvo. Desenhar curvas de bend no DAW é desenhar a curva de
  afinação, com precisão de amostra.
* **CC configuráveis:** quantidade de correção, vibrato, scoop e humanize automatizáveis por nota.
* **Velocity** opcional como "força" da correção naquela nota.
* Ferramenta futura: o CLI exporta a melodia detectada como `.mid` (transcrição). O usuário
  importa no DAW, corrige as notas erradas e usa como alvo. É o fluxo do Melodyne sem ARA.

### 3.6 Estrutura do código (workspace Rust)

```
buenovox/
├── Cargo.toml                  # workspace
├── truce.toml                  # vendor Jardim Sonoro, midi_input = true
├── crates/
│   ├── bvx-dsp/                # sem dependência de framework, sem alocação no process()
│   │   ├── analysis/           # pitch (YIN), arbiter (rede), epochs, voicing, envelope, noise
│   │   ├── brain/              # segment, decompose, target, policy, contour (Saitou), key
│   │   ├── synth/              # psola, hnm, formant, crossfade
│   │   └── util/               # ring, biquad, fft, smoothing
│   ├── bvx-ml/                 # inferência das redes minúsculas: pesos via include_bytes,
│   │                           # kernels próprios ou tract (feature opcional)
│   ├── bvx-plugin/             # ponte truce: params, MIDI, GUI egui, medidores, estado
│   └── bvx-cli/                # render offline + dump CSV/MIDI (o "heaven_cli" daqui)
├── tools/regua/                # Python: métricas, datasets, renders de referência, MUSHRA
├── tests/                      # sinais sintéticos com f0 conhecido, golden files
└── docs/                       # pesquisa, decisões (ADRs), resultados da régua
```

Regras de tempo real:

* nada de alocação, locks ou I/O em `process()`;
* buffers pré-alocados no `reset`;
* trabalho pesado (Auto-Key, futuras análises de frase) em `BackgroundTask` do truce, com troca
  lock-free;
* testes com o `rt-paranoid` que o próprio truce usa no CI.

Inferência neural: redes de ~100 k parâmetros cabem em kernels escritos à mão (conv 1-D e GRU,
no estilo do NeuralAmpModeler-rs). Isso evita trazer o ONNX Runtime para dentro do plugin. O
`tract` fica como opção para prototipar.

---

## 4. A régua: como saber que emociona sem poder ouvir

Nenhuma fase termina sem números. Herdamos a bateria do Heaven e ampliamos:

| Dimensão | Métrica | Meta |
|---|---|---|
| Precisão do detector | Erro mediano em cents e taxa de erro grosseiro (>50 cents) em sintéticos e no MDB-stem-synth | < 1 cent; < 0,5% grosseiro, inclusive em fry e soprosidade (VocalSet) |
| Afinação do corpo da nota | Desvio do centro percebido em relação ao alvo | ±5 cents (abaixo do limiar de especialistas, ~10) |
| Preservação do vibrato | Erro de taxa (Hz) e de extensão (cents) entre a entrada e a saída no modo Preserve | < 0,1 Hz; < 5% |
| Naturalidade das transições | Distância dos parâmetros de overshoot e preparation em relação à distribuição medida em cantores reais | Dentro de 1 desvio padrão |
| Microvariação | Jitter e shimmer de saída / entrada | 0,9–1,1 (nem achatar nem inventar) |
| Timbre | Erro de envelope nos harmônicos (dB), MCD | < 1 dB em ±12 st |
| Transparência | Null test com shift zero; FAD em relação ao seco | Bit-exato; FAD mínimo |
| Artefatos | Cliques (transientes acima do envelope local), modulação de amplitude periódica | Zero detectados |
| "Detector de auto-tune" | Discriminador treinado com voz real × corrigida (Auto-Tune, Melodyne, nós) | Acurácia dele na nossa saída ≈ 50% (acaso) |
| Desempenho | Tempo real em M-series no pior caso (96 kHz, Studio) | ≥ 30× o tempo real |
| Escuta humana | **MUSHRA cego** com o usuário e convidados contra Auto-Tune, Melodyne e Waves Tune, se disponíveis via `pedalboard` | Preferência estatisticamente significativa em naturalidade |

**Dados:**

* **Sintéticos com f0 conhecido:** gerados por nós, com vibrato e overshoot paramétricos.
* **Públicos:**
  * MDB-stem-synth e MIR-1K, para o detector;
  * **VocalSet**, com técnicas rotuladas: vibrato, belt, breathy, fry, straight. Ideal para
    teste de estresse;
  * NUS-48E, Opencpop, M4Singer e GTSinger, para notas e técnicas;
  * **Intonation** (Wager), para estatísticas de desafinação real.
* **Do usuário:** stems de *The Bedroom Sessions* e o vocal de bossa já usado no Heaven.

**Desafinação controlada:** a gente pega vozes afinadas e aplica erros realistas (o "detuner"
aprendido do BERT-APC). Assim a resposta certa é conhecida e o erro de correção vira um número.

---

## 5. Fases do desenvolvimento

Cada fase tem um entregável e um critério de saída **medido**. Não se avança com métrica no
vermelho.

### F0. Fundação e régua (1–2 semanas)
* Workspace Rust, `cargo truce new`, CI (testes, clippy, rt-paranoid).
* Python `tools/regua`: métricas da §4, downloads dos datasets e gerador de sintéticos.
* Renders de referência dos concorrentes instalados, via `pedalboard`.
* **Saída:** a régua roda de ponta a ponta num WAV qualquer e gera um relatório.

### F1. Análise (2–3 semanas)
* Porta o detector e as marcas de época do Heaven para `bvx-dsp::analysis`. Adiciona caminho de
  baixa latência (janela adaptativa ao período) para o modo Live.
* Árbitro neural: treina e distila PESTO ou SwiftF0 num modelo de ~100 k parâmetros, com
  inferência própria em Rust e teste de equivalência numérica contra o PyTorch.
* Classificador de vozeamento em 3 classes (vozeado, vozeado fraco, ruído) e estimador de
  componente de ruído.
* **Saída:** metas de precisão da §4 em sintéticos e no VocalSet, inclusive fry e breathy.

### F2. Síntese (2–3 semanas)
* PSOLA do Heaven com null test, separação harmônico/ruído, troca de razão na época e pré-
  distorção cepstral para shifts grandes.
* **Saída:** transparência, envelope < 1 dB e zero cliques num sweep de razão de −12 a +12 st e
  em modulação de 5 Hz.

### F3. Cérebro v1: Classic + MIDI (2 semanas)
* Tonalidade e escala, MIDI Target, MIDI Scale, Retune de 2ª ordem, Flex-Tune, Humanize, Hard.
* `bvx-cli` renderiza e despeja as curvas (entrada, alvo, saída) em CSV.
* **Saída:** paridade medida com o Auto-Tune Classic em afinação, melhor que ele em cliques e
  jitter.

### F4. Cérebro v2: Natural (3–5 semanas), o diferencial
* Segmentação com Viterbi e atraso fixo, decomposição da expressão, centro percebido, correção
  por offset de nota.
* Transições de Saitou. Antes disso, uma etapa de **medição**: ajustar o modelo em centenas de
  transições reais (VocalSet, GTSinger, stems do usuário) para obter distribuições de parâmetros
  por intervalo, direção e andamento.
* Vibrato Preserve / Reduce / Enhance / Create, encurtamento de scoop, entonação Pure.
* **Saída:**
  * preservação de vibrato e transições dentro das metas;
  * o "detector de auto-tune" não distingue a saída;
  * o primeiro MUSHRA com o usuário.

### F5. Plugin (2–3 semanas, em paralelo a partir de F3)
* Parâmetros automatizáveis, presets (`presets/*.preset`), estado e os três modos de latência.
* **GUI em egui:** gráfico de rolagem com f0 de entrada, alvo e saída; faixa de notas; visual de
  vibrato; teclado MIDI; medidores.
* `cargo truce validate` (auval, pluginval rigor 10, clap-validator) e `cargo truce screenshot`
  para a documentação.
* **Saída:** AU + VST3 + CLAP passando nos validadores e testados no Ableton Live.

### F6. Modo Studio com motor harmônico (3–4 semanas)
* Motor B (HNM), lookahead longo, comparação cega A contra B por tipo de voz.
* **Saída:** o motor B vence o A em vozes soprosas e no vibrato criado, ou é descartado com o
  relatório.

### F7. Pesquisa neural (contínua, com portões)
1. **Pesos do centro percebido aprendidos** (SPP).
2. **Prior de contexto melódico leve:** um transformer pequeno ou um GRU que, dado o histórico de
   notas, a tonalidade e o acorde MIDI, prevê a distribuição da nota pretendida. É destilado do
   BERT-APC ou treinado no Intonation com detuner sintético.
3. **Gerador de contornos expressivos** com perda adversarial (a ideia do SeqGAN, em versão
   contínua) e destilação em parâmetros de Saitou ou numa rede causal minúscula.
4. **Pós-filtro neural de restauração** auto-supervisionado.

Cada item só entra no plugin se vencer no MUSHRA e couber no orçamento de CPU e latência.

### F8. Afinação final e escuta (contínua)
* Rodadas de MUSHRA, ajustes de defaults e presets por estilo (pop, bossa, gospel, R&B, lírico,
  rap melódico) e documentação no estilo do README do Heaven ("como pesquisamos").

---

## 6. Interface: poucos controles, todos com propósito

Princípio herdado do Heaven: soar bem logo de cara.

* **Fonte:** Key + Scale · MIDI Target · MIDI Scale · Auto-Key.
* **Estilo:** Natural · Classic · Hard.
* **Correção:** Amount (quanto do centro corrigir) · Speed (2ª ordem; no Natural, a velocidade
  da transição) · Flex.
* **Expressão:** Vibrato (Preserve / Reduce / Enhance / Create + quantidade) · Scoop · Humanize.
* **Voz:** Throat (garganta) · Breath (nível do ruído separado).
* **Afinação:** Equal / Pure / Expressive · A4 de referência.
* **Latência:** Live / Track / Studio.

---

## 7. Riscos e respostas

| Risco | Resposta |
|---|---|
| Lookahead longo inviabiliza a gravação | Os três modos. O Live é sempre usável, mesmo sem a correção por nota. |
| Erro de segmentação gera nota errada (pior que nenhuma correção) | Confiança por nota: em dúvida, corrigir menos (Amount adaptativo). O MIDI tem prioridade. |
| Vozes extremas (fry, belt, rap, soprosidade) | VocalSet na régua desde a F1; árbitro neural; separação de ruído. |
| Sem ARA, sem edição global | MIDI como editor e transcrição para `.mid` pelo CLI. Acompanhar o truce; ARA pode virar contribuição upstream. |
| CPU das redes no áudio | Redes ≤ 100–200 k parâmetros, kernels próprios, medição no CI. |
| Licenças | Código do projeto em MIT-0. Dependências permissivas (MIT/Apache/BSD/Zlib e a Truce License 1.0, que equivale a MIT/Apache para plugins). GPL e AGPL (OpenVoxTuner, Rubber Band, Autotalent) apenas como leitura. |
| Patentes | §2.5: implementação a partir de literatura acadêmica, sem copiar reivindicações, uso não comercial. |

---

## 8. Próximos passos imediatos

1. Aprovar este plano (ou ajustar prioridades, especialmente as metas da §4 e os modos de
   latência).
2. F0: criar o workspace, o CI e a régua em Python.
3. Baixar o VocalSet e o MDB-stem-synth e gerar os sintéticos.
4. Portar o detector e o PSOLA do Heaven para `bvx-dsp` e rodar a régua como linha de base, antes
   de qualquer melhoria.

---

## Referências

**Repositórios analisados:**
* truce: https://github.com/truce-audio/truce
* Heaven: https://github.com/nagualcode/heaven
* OpenVoxTuner: https://github.com/EiffelBS/OpenVoxTuner
* mutagen-v1: https://github.com/JWaidtslow/mutagen-v1
* SeqGAN: https://github.com/LantaoYu/SeqGAN

**Pesquisa:**
* de Cheveigné & Kawahara, *YIN* (JASA, 2002)
* Moulines & Charpentier, *PSOLA* (Speech Communication, 1990)
* Röbel & Rodet, *Efficient Spectral Envelope Estimation…* (DAFx 2005): https://dafx.de/paper-archive/details/EDSTSaO17Gd_0LV03iwDfw
* Röbel & Rodet, *Real Time Signal Transposition with Envelope Preservation in the Phase
  Vocoder*: https://hal.archives-ouvertes.fr/hal-01161347
* Saitou et al., modelo de controle de f0 (WASPAA 2007): https://staff.aist.go.jp/m.goto/PAPER/WASPAA2007saitou.pdf
* Wager et al., *Deep Autotuner* (ICASSP 2020): https://arxiv.org/abs/2002.05511
* Hai & Elhilali, *Diff-Pitcher* (WASPAA 2023): https://engineering.jhu.edu/lcap/data/uploads/pdfs/waspaa2023_hai.pdf
* *BERT-APC* (2025): https://arxiv.org/html/2511.20006v1
* *Self-supervised restoration of singing voice degraded by pitch shifting using shallow
  diffusion* (2026): https://arxiv.org/html/2601.10345v1
* Gu et al., *Neurodyne* (Interspeech 2025): https://arxiv.org/abs/2505.15368
* Plug-in de conversão de voz cantada em tempo real (DAFx 2023): https://www.dafx.de/paper-archive/2023/DAFx23_paper_21.pdf
* RT-PAD-VC (DAFx 2025): https://www.audiolabs-erlangen.de/resources/NLUI/2025-DAFx-RT-PAD-VC
* *PESTO* (ISMIR 2023 / TISMIR 2025): https://arxiv.org/abs/2508.01488
* *SwiftF0* (2025): https://arxiv.org/abs/2508.18440
* *FCPE* (2025): https://arxiv.org/html/2509.15140v1
* Larrouy-Maestri et al., tolerância à desafinação: https://orbi.uliege.be/bitstream/2268/186233/2/2015_ESCOM_tolerance.pdf
* Yu et al., *SeqGAN* (AAAI 2017)

**Patentes:**
* US 5,973,252 (Antares / Hildebrand): https://patents.google.com/patent/US5973252
* US 8,022,286 (Celemony / Neubäcker): https://patents.google.com/patent/US8022286
* US 5,231,671 e US 5,567,901 (IVL / DigiTech) e US 8,168,877: citadas no README do Heaven

**Mercado:**
* AutoTune Advanced / DVA: https://www.musicradar.com/music-tech/antares-launches-autotune-advanced-with-new-dvatm-vibrato-detection-and-preservation-engine
* AutoTune Pro 11: https://www.sweetwater.com/store/detail/AutoTunePro11--antares-auto-tune-pro-11-pitch-correction-and-vocal-effects-plug-in
* Signalsmith Stretch: https://github.com/Signalsmith-Audio/signalsmith-stretch
* NeuralAmpModeler-rs: https://docs.rs/NeuralAmpModeler-rs
* tract: https://github.com/kali/tract
