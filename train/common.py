"""Leitura dos quadros gravados por `bvx_cli dataset/features` e metricas de altura."""

import numpy as np

META = ["t", "label", "f0", "aper", "db", "net_f0", "net_voiced"]


def read_frames(path, feats=True):
    with open(path, "rb") as f:
        head = f.read(16)
        assert head[:4] == b"BVXF", path
        _, n_meta, n_feat = np.frombuffer(head[4:], "<u4")
        data = np.fromfile(f, "<f4")
    rec = n_meta + n_feat
    data = data[: len(data) // rec * rec].reshape(-1, rec)
    meta = data[:, :n_meta]
    return (meta, data[:, n_meta:]) if feats else meta


def cents(a, b):
    return 1200.0 * np.log2(np.maximum(a, 1e-6) / np.maximum(b, 1e-6))


def interp_ref(t_ref, f_ref, t):
    """f0 de referencia nos instantes t: log-linear entre pontos vozeados, 0 perto de silencio."""
    i = np.clip(np.searchsorted(t_ref, t), 1, len(t_ref) - 1)
    a, b = f_ref[i - 1], f_ref[i]
    w = (t - t_ref[i - 1]) / np.maximum(t_ref[i] - t_ref[i - 1], 1e-9)
    both = (a > 0) & (b > 0)
    out = np.where(both, np.exp(np.log(np.maximum(a, 1e-6)) * (1 - w) + np.log(np.maximum(b, 1e-6)) * w), 0.0)
    return out


def pitch_metrics(ref, est):
    """ref/est em Hz (0 = sem voz). Devolve dicionario de metricas (fracoes)."""
    rv, ev = ref > 0, est > 0
    both = rv & ev
    c = cents(est[both], ref[both])
    gross = np.abs(c) > 50
    octave = gross & (np.abs(np.abs(c) - 1200 * np.round(np.abs(c) / 1200)) < 50) & (np.round(np.abs(c) / 1200) >= 1)
    return {
        "rpa": float(np.sum(np.abs(c) <= 50) / max(rv.sum(), 1)),
        "gross": float(gross.mean()) if both.any() else 0.0,
        "octave": float(octave.mean()) if both.any() else 0.0,
        "v_recall": float(both.sum() / max(rv.sum(), 1)),
        "v_false": float((ev & ~rv).sum() / max((~rv).sum(), 1)),
        "median_c": float(np.median(np.abs(c))) if both.any() else 0.0,
        "n": int(rv.sum()),
    }


def fmt(m):
    return (f"RPA {100*m['rpa']:5.1f}%  grosseiro {100*m['gross']:5.2f}%  oitava {100*m['octave']:5.2f}%  "
            f"voz achada {100*m['v_recall']:5.1f}%  voz falsa {100*m['v_false']:5.1f}%  mediana {m['median_c']:.1f}c")
