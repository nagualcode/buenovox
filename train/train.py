"""Treina o detector neural de apoio e exporta os pesos para o Rust.

Dados: quadros do cantor sintetico (rotulo exato) e do VocalSet com rotulos do professor
(CREPE). Validacao: vocadito (anotacao humana, nunca usada no treino).

train.py <saida.bin> [--vocalset] [--epochs N]
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import fmt, interp_ref, pitch_metrics, read_frames
from model import BINS, CH, PitchNet, bin_hz, export, hz_bin

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")


def load_synth():
    feats, labels = [], []
    for p in sorted(glob.glob(f"{D}/synth/*.bin")):
        meta, feat = read_frames(p)
        feats.append(feat.astype(np.float16))
        labels.append(meta[:, 1].copy())
    return np.concatenate(feats), np.concatenate(labels)


def load_vocalset(listfile, max_frames=None):
    """Rotulos do professor: f0 onde a periodicidade e alta, 0 onde e baixa, -2 (ignorar) no meio."""
    feats, labels = [], []
    names = [os.path.basename(l.strip())[:-4] for l in open(listfile) if l.strip()]
    for name in names:
        fp, tp = f"{D}/feat/vs/{name}.bin", f"{D}/teacher/{name}.npz"
        if not (os.path.exists(fp) and os.path.exists(tp)):
            continue
        meta, feat = read_frames(fp)
        z = np.load(tp)
        t = meta[:, 0]
        i = np.clip(np.round(t / 0.01).astype(int), 0, len(z["f0"]) - 1)
        f0, per = z["f0"][i], z["per"][i]
        lab = np.where(per > 0.6, f0, np.where(per < 0.3, 0.0, -2.0))
        # um quadro a cada dois (os vizinhos sao quase iguais; cabe na memoria)
        feats.append(feat[::2].astype(np.float16))
        labels.append(lab[::2].astype(np.float32))
    if not feats:
        return None, None
    return np.concatenate(feats), np.concatenate(labels)


def load_vocadito():
    feats, refs = [], []
    for p in sorted(glob.glob(f"{D}/feat/vocadito/*.bin")):
        name = os.path.basename(p)[:-4]
        meta, feat = read_frames(p)
        ann = np.loadtxt(f"{D}/vocadito/Annotations/F0/{name}_f0.csv", delimiter=",")
        feats.append(feat.astype(np.float16))
        refs.append(interp_ref(ann[:, 0], ann[:, 1], meta[:, 0]))
    return np.concatenate(feats), np.concatenate(refs)


def decode(logits, vlogit):
    """Como PitchNet::infer: media ponderada (+-2 posicoes) em torno do maximo."""
    p = torch.softmax(logits, -1)
    b = p.argmax(-1, keepdim=True)
    idx = torch.arange(BINS, device=p.device)[None]
    win = (idx - b).abs() <= 2
    pw = p * win
    pos = (pw * idx).sum(-1) / pw.sum(-1).clamp_min(1e-9)
    return bin_hz(pos.cpu().numpy()), torch.sigmoid(vlogit).cpu().numpy(), pw.sum(-1).cpu().numpy()


@torch.no_grad()
def predict(model, feat, dev, bs=8192):
    out_f, out_v = [], []
    for i in range(0, len(feat), bs):
        x = torch.from_numpy(feat[i : i + bs].astype(np.float32)).view(-1, CH, BINS).to(dev)
        lg, vl = model(x)
        f, v, _ = decode(lg, vl)
        out_f.append(f)
        out_v.append(v)
    return np.concatenate(out_f), np.concatenate(out_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--vocalset", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hid", type=int, default=16)
    ap.add_argument("--vs-weight", type=float, default=1.0)
    a = ap.parse_args()
    torch.manual_seed(1)
    np.random.seed(1)
    dev = "cpu"

    xs, ys = load_synth()
    bad = ~np.isfinite(xs.astype(np.float32)).all(1)
    print(f"quadros com entradas invalidas: {bad.sum()}")
    xs, ys = xs[~bad], ys[~bad]
    src = [np.zeros(len(ys), np.int8)]
    print(f"sintetico: {len(ys)} quadros, {np.mean(ys > 0) * 100:.0f}% com voz")
    if a.vocalset:
        xv, yv = load_vocalset(f"{D}/vs_train.txt")
        if xv is not None:
            print(f"VocalSet: {len(yv)} quadros, {np.mean(yv > 0) * 100:.0f}% com voz, {np.mean(yv == -2) * 100:.0f}% ignorados")
            xs, ys = np.concatenate([xs, xv]), np.concatenate([ys, yv])
            src.append(np.ones(len(yv), np.int8))
    src = np.concatenate(src)
    xval, rval = load_vocadito()

    model = PitchNet(a.hid).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    n = len(ys)
    bs = 1024
    steps = a.epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.1)
    bins = torch.arange(BINS, dtype=torch.float32)
    sigma = 0.8
    step = 0
    for ep in range(a.epochs):
        perm = np.random.permutation(n)
        model.train()
        t0 = time.time()
        tot = 0.0
        for i in range(0, n - bs + 1, bs):
            idx = np.sort(perm[i : i + bs])
            x = torch.from_numpy(xs[idx].astype(np.float32)).view(-1, CH, BINS)
            y = torch.from_numpy(ys[idx])
            w_src = torch.from_numpy(np.where(src[idx] == 1, a.vs_weight, 1.0).astype(np.float32))
            # aumentacao: ganho no espectro (log) e ruido leve nas entradas
            x = x + 0.02 * torch.randn_like(x)
            logits, vl = model(x)
            voiced = (y >= 62.0) & (y <= 1080.0)
            known = y > -1.5
            pos = torch.from_numpy(np.clip(hz_bin(np.maximum(y.numpy(), 1.0)), 0, BINS - 1).astype(np.float32))
            q = torch.exp(-0.5 * ((bins[None] - pos[:, None]) / sigma) ** 2)
            q = q / q.sum(-1, keepdim=True)
            lp = F.log_softmax(logits, -1)
            l_pitch = torch.where(voiced, -(q * lp).sum(-1), torch.zeros(()))
            l_pitch = (l_pitch * w_src).sum() / (voiced * w_src).sum().clamp_min(1)
            l_voice = F.binary_cross_entropy_with_logits(vl, voiced.float(), reduction="none")
            l_voice = (l_voice * known * w_src).sum() / (known * w_src).sum().clamp_min(1)
            loss = l_pitch + l_voice
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot += loss.item()
        model.eval()
        f, v = predict(model, xval, dev)
        est = np.where(v > 0.5, f, 0.0)
        print(f"epoca {ep+1}: perda {tot / (n // bs):.3f} ({time.time()-t0:.0f}s) | vocadito, rede sozinha: {fmt(pitch_metrics(rval, est))}", flush=True)
    export(model.cpu(), a.out)
    torch.save(model.state_dict(), a.out + ".pt")
    print("exportado:", a.out)


if __name__ == "__main__":
    main()
