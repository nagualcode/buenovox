"""Rede do detector neural de apoio (a mesma arquitetura que src/dsp/neural.rs executa).

Entrada [N, CH, BINS]: periodicidade do YIN e espectro empilhado por harmonicos num eixo log de
altura (2 posicoes por semitom a partir de 62 Hz). Saida: logits de altura por posicao e um
logit de vozeamento (media e maximo de cada canal da ultima camada).
"""

import struct

import numpy as np
import torch
import torch.nn as nn

BINS = 100
PER_ST = 2.0
F_LO = 62.0
CH = 10
# (cin, cout, k, dilatacao) das camadas ocultas
def layer_spec(hid=16):
    return [(CH, hid, 5, 1), (hid, hid, 5, 1), (hid, hid, 5, 2), (hid, hid, 5, 4)]


def bin_hz(b):
    return F_LO * 2.0 ** (np.asarray(b) / (12.0 * PER_ST))


def hz_bin(f):
    return 12.0 * PER_ST * np.log2(np.asarray(f) / F_LO)


class PitchNet(nn.Module):
    def __init__(self, hid=16):
        super().__init__()
        self.spec = layer_spec(hid)
        self.convs = nn.ModuleList(nn.Conv1d(ci, co, k, dilation=d, padding=k // 2 * d) for ci, co, k, d in self.spec)
        self.head = nn.Conv1d(hid, 1, 1)
        self.voice = nn.Linear(2 * hid, 1)

    def forward(self, x):
        for c in self.convs:
            x = torch.relu(c(x))
        logits = self.head(x).squeeze(1)
        v = self.voice(torch.cat([x.mean(-1), x.amax(-1)], 1)).squeeze(1)
        return logits, v


def export(model: PitchNet, path: str):
    """Formato lido por PitchNet::from_bytes (little-endian)."""
    out = bytearray(b"BVXN")
    out += struct.pack("<II", 1, len(model.convs))

    def conv(c: nn.Conv1d):
        w = c.weight.detach().cpu().numpy().astype("<f4")  # [cout, cin, k]
        b = c.bias.detach().cpu().numpy().astype("<f4")
        return struct.pack("<IIII", c.in_channels, c.out_channels, c.kernel_size[0], c.dilation[0]) + w.tobytes() + b.tobytes()

    for c in model.convs:
        out += conv(c)
    out += conv(model.head)
    out += model.voice.weight.detach().cpu().numpy().astype("<f4").reshape(-1).tobytes()
    out += model.voice.bias.detach().cpu().numpy().astype("<f4").tobytes()
    with open(path, "wb") as f:
        f.write(out)


if __name__ == "__main__":
    import sys

    torch.manual_seed(0)
    m = PitchNet()
    export(m, sys.argv[1] if len(sys.argv) > 1 else "models/pitchnet.bin")
    print(sum(p.numel() for p in m.parameters()), "parametros")
