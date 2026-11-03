"""Gera o arquivo de conferencia PyTorch x Rust: entradas reais (vocadito) e as saidas esperadas.

check.py <pesos.pt> <saida.bin>: N quadros de [entradas (CH*BINS), logits (BINS), logit de voz].
"""

import glob
import struct
import sys

import numpy as np
import torch

from common import read_frames
from model import BINS, CH, PitchNet

state = torch.load(sys.argv[1])
hid = state["head.weight"].shape[1]
m = PitchNet(hid)
m.load_state_dict(state)
m.eval()
feats = []
for p in sorted(glob.glob("data/feat/vocadito/*.bin"))[:4]:
    _, f = read_frames(p)
    feats.append(f[:: max(1, len(f) // 16)][:16])
x = np.concatenate(feats).astype(np.float32)
with torch.no_grad():
    lg, vl = m(torch.from_numpy(x).view(-1, CH, BINS))
out = bytearray(b"BVXC") + struct.pack("<II", len(x), CH * BINS)
for i in range(len(x)):
    out += x[i].astype("<f4").tobytes() + lg[i].numpy().astype("<f4").tobytes() + struct.pack("<f", float(vl[i]))
open(sys.argv[2], "wb").write(out)
print(len(x), "quadros")
