"""Correcao em canto real: o vocadito com as notas anotadas viram alvos MIDI.

Para cada trecho: bvx_cli com source=miditarget e as notas anotadas; altura da entrada e da
saida medidas pelo proprio detector (bvx_cli features). Por nota (sem os 60 ms das pontas):
* centro: mediana(saida) - nota alvo (cents) -> erro de afinacao que sobrou;
* expressao: desvio padrao de (saida - entrada) dentro da nota (cents) -> 0 = vibrato e
  flutuacoes do cantor intactos (a correcao so deslocou a nota inteira).

eval_correction.py [style=natural] [latency=studio]
"""

import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

from common import read_frames

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLI = os.path.join(ROOT, "target/release/bvx_cli")
extra = sys.argv[1:]


def midi(f):
    return 69 + 12 * np.log2(np.maximum(f, 1e-6) / 440.0)


centers, exprs, raw = [], [], []
by_part = {"inicio (60-150 ms)": [], "corpo (>150 ms)": []}
by_dur = {"< 250 ms": [], "250-500 ms": [], "> 500 ms": []}
tmp = tempfile.mkdtemp()
for wav in sorted(glob.glob(f"{HERE}/data/vocadito/Audio/*.wav")):
    name = os.path.basename(wav)[:-4]
    ann = np.atleast_2d(np.loadtxt(f"{HERE}/data/vocadito/Annotations/Notes/{name}_notesA1.csv", delimiter=","))
    notes = [(a, a + d, int(round(midi(f)))) for a, f, d in ann]
    spec = ";".join(f"{a:.3f}-{b:.3f}:{n}" for a, b, n in notes)
    out = f"{tmp}/{name}_out.wav"
    subprocess.run([CLI, wav, out, "source=miditarget", f"midi={spec}", *extra], check=True, capture_output=True)
    fi, fo = f"{tmp}/{name}_in.bin", f"{tmp}/{name}_out.bin"
    subprocess.run([CLI, "features", wav, fi], check=True, capture_output=True)
    subprocess.run([CLI, "features", out, fo], check=True, capture_output=True)
    mi, mo = read_frames(fi, feats=False), read_frames(fo, feats=False)
    n = min(len(mi), len(mo))
    t, pin, pout = mi[:n, 0], mi[:n, 2], mo[:n, 2]
    for a, b, note in notes:
        k = (t > a + 0.06) & (t < b - 0.06) & (pin > 0) & (pout > 0)
        if k.sum() < 10:
            continue
        mi_, mo_ = midi(pin[k]), midi(pout[k])
        for lab, (x0, x1) in {"inicio (60-150 ms)": (0.06, 0.15), "corpo (>150 ms)": (0.15, 99)}.items():
            kk = k & (t > a + x0) & (t < min(a + x1, b - 0.06))
            if kk.sum() >= 3:
                by_part[lab].append(abs(100 * (np.median(midi(pout[kk])) - note)))
        d = b - a
        by_dur["< 250 ms" if d < 0.25 else "250-500 ms" if d < 0.5 else "> 500 ms"].append(abs(100 * (np.median(mo_) - note)))
        centers.append(100 * (np.median(mo_) - note))
        raw.append(100 * (np.median(mi_) - note))
        exprs.append(100 * np.std(mo_ - mi_))
c, e, r = np.abs(centers), np.array(exprs), np.abs(raw)
print(f"{len(c)} notas | cantor: erro mediano {np.median(r):.1f} c (90% {np.percentile(r, 90):.1f}) | "
      f"saida: erro mediano {np.median(c):.1f} c (90% {np.percentile(c, 90):.1f}) | "
      f"expressao alterada: mediana {np.median(e):.1f} c (90% {np.percentile(e, 90):.1f})")
for lab, v in list(by_part.items()) + list(by_dur.items()):
    print(f"  {lab:>18}: {len(v):4d} notas, erro mediano {np.median(v):.1f} c, 90% {np.percentile(v, 90):.1f} c")
