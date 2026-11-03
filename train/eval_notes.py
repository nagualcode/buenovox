"""Quantas notas o cerebro escolhe errado no vocadito (escala cromatica, sem MIDI): o alvo
decidido no meio da nota contra a nota anotada (arredondada ao semitom com o A4 do trecho)."""

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
wrong = total = near_half = wrong_near = 0
diffs = []
for wav in sorted(glob.glob(f"{HERE}/data/vocadito/Audio/*.wav")):
    name = os.path.basename(wav)[:-4]
    ann = np.atleast_2d(np.loadtxt(f"{HERE}/data/vocadito/Annotations/Notes/{name}_notesA1.csv", delimiter=","))
    m = 69 + 12 * np.log2(ann[:, 1] / 440)
    # afinacao de referencia do trecho (o cantor pode estar todo um pouco alto ou baixo)
    off = np.median(m - np.round(m))
    dump = f"{tmp}/{name}.csv"
    keyargs = []
    if os.environ.get("KEY"):
        # tonalidade pelos perfis de Krumhansl-Kessler sobre as notas anotadas (peso = duracao)
        maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        mnr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        h = np.zeros(12)
        for mm, d in zip(np.round(m - off).astype(int), ann[:, 2]):
            h[mm % 12] += d
        best = max(((np.corrcoef(h, np.roll(pr, k))[0, 1], k, sc) for k in range(12) for pr, sc in [(maj, "major"), (mnr, "minor")]))
        names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        keyargs = [f"key={names[best[1]]}", f"scale={best[2]}"]
    subprocess.run([CLI, wav, f"{tmp}/o.wav", f"dump={dump}", f"a4={440 * 2 ** (off / 12):.2f}", *keyargs, *sys.argv[1:]], check=True, capture_output=True)
    rows = list(csv.DictReader(open(dump)))
    t = np.array([float(r["t"]) for r in rows])
    tg = np.array([float(r["target"]) for r in rows])
    for (a, f, d), mm in zip(ann, m):
        k = (t > a + 0.3 * d) & (t < a + 0.7 * d) & np.isfinite(tg)
        if k.sum() < 3:
            continue
        want = round(mm - off)
        got = round(float(np.median(tg[k])))
        total += 1
        bad = got != want
        diffs.append(got - want)
        wrong += bad
        if abs((mm - off) - want) > 0.35:
            near_half += 1
            wrong_near += bad
print(f"{total} notas: alvo errado em {100*wrong/total:.1f}% | notas cantadas a mais de 35 c do semitom: {near_half} ({100*wrong_near/max(near_half,1):.0f}% delas erradas)")
import collections
print(collections.Counter(diffs).most_common(8))
