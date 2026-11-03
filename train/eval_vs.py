"""VocalSet de teste (cantores fora do treino) contra o professor, por tecnica."""

import glob
import os
import sys

import numpy as np

from common import fmt, pitch_metrics, read_frames

HERE = os.path.dirname(os.path.abspath(__file__))
feat_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data/feat/vs")
names = [os.path.basename(l.strip())[:-4] for l in open(os.path.join(HERE, "data/vs_test.txt")) if l.strip()]
by_tech = {}
for name in names:
    fp, tp = f"{feat_dir}/{name}.bin", os.path.join(HERE, f"data/teacher/{name}.npz")
    if not (os.path.exists(fp) and os.path.exists(tp)):
        continue
    meta = read_frames(fp, feats=False)
    z = np.load(tp)
    i = np.clip(np.round(meta[:, 0] / 0.01).astype(int), 0, len(z["f0"]) - 1)
    f0, per = z["f0"][i], z["per"][i]
    keep = (per > 0.6) | (per < 0.3)
    ref = np.where(per > 0.6, f0, 0.0)[keep]
    tech = [t for t in ("vocal_fry", "breathy", "belt", "lip_trill", "straight", "vibrato", "spoken") if t in name]
    tech = tech[0] if tech else "outros"
    r, e = by_tech.setdefault(tech, ([], []))
    r.append(ref)
    e.append(meta[:, 2][keep])
allr, alle = [], []
for tech, (r, e) in sorted(by_tech.items()):
    r, e = np.concatenate(r), np.concatenate(e)
    allr.append(r)
    alle.append(e)
    print(f"VocalSet {tech:>10}: {fmt(pitch_metrics(r, e))}")
print(f"VocalSet {'todos':>10}: {fmt(pitch_metrics(np.concatenate(allr), np.concatenate(alle)))}")
