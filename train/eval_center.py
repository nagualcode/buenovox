"""Centro percebido: o centro da nota calculado pelo cerebro contra a altura que os anotadores
do vocadito deram a cada nota (de ouvido). Metade dos trechos ajusta, a outra metade confere.

eval_center.py [chave=valor do bvx_cli...]
"""

import csv
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(os.path.dirname(HERE), "target/release/bvx_cli")
tmp = tempfile.mkdtemp()
errs = {"ajuste": [], "conferencia": []}
extra = {}
naive = {"ajuste": [], "conferencia": []}
for i, wav in enumerate(sorted(glob.glob(f"{HERE}/data/vocadito/Audio/*.wav"))):
    name = os.path.basename(wav)[:-4]
    part = "ajuste" if i % 2 == 0 else "conferencia"
    ann = np.atleast_2d(np.loadtxt(f"{HERE}/data/vocadito/Annotations/Notes/{name}_notesA1.csv", delimiter=","))
    dump = f"{tmp}/{name}.csv"
    subprocess.run([CLI, wav, f"{tmp}/o.wav", "amount=0", f"dump={dump}", *sys.argv[1:]], check=True, capture_output=True)
    rows = list(csv.DictReader(open(dump)))
    t = np.array([float(r["t"]) for r in rows])
    cn = np.array([float(r["center"]) for r in rows])
    pin = np.array([float(r["input"]) for r in rows])
    for a, f, d in ann:
        ref = 69 + 12 * np.log2(f / 440)
        k = (t > a + 0.05) & (t < a + d - 0.05) & (cn > 0) & (pin > 0)
        if k.sum() < 8:
            continue
        # o centro que vale para a nota: o do fim (quando o cerebro ja viu a nota toda)
        errs[part].append(100 * (np.median(cn[k]) - ref))
        naive[part].append(100 * (np.mean(pin[k]) - ref))
        for lab, x0 in [("media sem 80 ms iniciais", 0.08), ("media sem 150 ms iniciais", 0.15)]:
            kk = k & (t > a + x0)
            if kk.sum() >= 4:
                extra.setdefault(lab, {}).setdefault(part, []).append(100 * (np.mean(pin[kk]) - ref))
        kk = k & (t > a + 0.05)
        w = np.linspace(0.3, 1.0, kk.sum())
        extra.setdefault("media com peso crescente", {}).setdefault(part, []).append(100 * (np.sum(pin[kk] * w) / w.sum() - ref))
for part in errs:
    e, n = np.abs(errs[part]), np.abs(naive[part])
    print(f"{part:>12}: {len(e)} notas | centro do cerebro: mediana {np.median(e):.1f} c, 90% {np.percentile(e,90):.1f} c | media simples: mediana {np.median(n):.1f} c")
for lab, d in extra.items():
    print(f"{lab:>26}: " + " | ".join(f"{p} mediana {np.median(np.abs(v)):.1f} c" for p, v in d.items()))
