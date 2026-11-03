"""Mede transicoes reais entre notas no VocalSet (altura do professor CREPE, 10 ms).

Procura passos limpos em escalas e arpejos: ~120 ms estavel numa altura, mudanca de 1 a 12
semitons, ~120 ms estavel na outra. Em cada um mede o tempo de subida (10-90%) e o overshoot,
e ajusta um sistema de 2a ordem (Saitou et al.): zeta pelo overshoot, frequencia natural pelo
tempo de subida. Resultado por tamanho de intervalo e direcao.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
files = [f for f in glob.glob(f"{HERE}/data/teacher/*.npz") if ("scales" in f or "arpeggios" in f)]


def midi(f):
    return 69 + 12 * np.log2(np.maximum(f, 1e-6) / 440)


rows = []
for f in files:
    z = np.load(f)
    p, per = midi(z["f0"]), z["per"]
    ok = per > 0.5
    n = len(p)
    i = 12
    while i < n - 40:
        a = p[i - 12 : i]
        if not ok[i - 12 : i].all() or np.ptp(a) > 0.35:
            i += 1
            continue
        p0 = np.median(a)
        # a altura sai da faixa da nota
        j = i
        while j < n - 30 and abs(p[j] - p0) < 0.3 and ok[j]:
            j += 1
        if j >= n - 30 or not ok[j : j + 30].all():
            i = j + 1
            continue
        # procura a nova nota estavel ate 300 ms depois
        found = None
        for k in range(j + 3, min(j + 30, n - 12)):
            b = p[k : k + 12]
            if np.ptp(b) < 0.35 and 0.8 < abs(np.median(b) - p0) < 12.5:
                found = k
                break
        if found is None:
            i = j + 1
            continue
        p1 = np.median(p[found + 6 : found + 18]) if found + 18 < n else np.median(p[found : found + 12])
        step = p1 - p0
        seg = (p[j - 3 : found + 18] - p0) / step  # 0 -> 1
        # tempo de subida 10-90% e overshoot
        try:
            t10 = np.argmax(seg >= 0.1)
            t90 = np.argmax(seg >= 0.9)
        except ValueError:
            i = found
            continue
        rise = (t90 - t10) * 0.01
        over = max(0.0, seg.max() - 1.0)
        if 0.0 < rise < 0.3:
            rows.append((step, rise, over))
        i = found + 12

r = np.array(rows)
print(f"{len(r)} transicoes de {len(files)} arquivos")


def fit(rise, over):
    # zeta pelo overshoot; wn pelo tempo de subida 10-90% de um 2a ordem subamortecido
    zeta = np.where(over > 1e-3, -np.log(over) / np.sqrt(np.pi**2 + np.log(over) ** 2), 1.0)
    zeta = np.clip(zeta, 0.2, 1.0)
    wn = (1.0 - 0.4167 * zeta + 2.917 * zeta**2) / rise  # aproximacao classica do tempo de subida
    return zeta, wn


for name, sel in [("todos", np.ones(len(r), bool)),
                  ("subida 1-2 st", (r[:, 0] > 0) & (abs(r[:, 0]) < 2.5)),
                  ("descida 1-2 st", (r[:, 0] < 0) & (abs(r[:, 0]) < 2.5)),
                  ("subida 3-5 st", (r[:, 0] > 0) & (abs(r[:, 0]) >= 2.5) & (abs(r[:, 0]) < 5.5)),
                  ("descida 3-5 st", (r[:, 0] < 0) & (abs(r[:, 0]) >= 2.5) & (abs(r[:, 0]) < 5.5)),
                  ("salto >= 6 st", abs(r[:, 0]) >= 5.5)]:
    x = r[sel]
    if len(x) < 5:
        continue
    rise, over = np.median(x[:, 1]), np.median(x[:, 2])
    zeta, wn = fit(np.array([rise]), np.array([over]))
    print(f"{name:>15}: {len(x):5d} | subida 10-90% {1000*rise:5.0f} ms | overshoot {100*over:4.1f}% (75%: {100*np.percentile(x[:,2],75):4.1f}%) | zeta {zeta[0]:.2f} | fn {wn[0]/(2*np.pi):.1f} Hz")
