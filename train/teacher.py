"""Rotulos do professor (CREPE "full", via torchcrepe) para audio real sem anotacao de f0.

Para cada wav grava <saida>/<nome>.npz com t (s), f0 (Hz) e periodicidade a cada 10 ms.
CREPE (MIT) so serve de professor: o detector do plugin e a rede pequena destilada.
"""

import glob
import os
import sys
import time

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sps
import torch
import torchcrepe


def load16k(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.dtype != np.float32 or np.abs(x).max() > 1.5:
        x /= 32768.0
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        x = sps.resample_poly(x, 16000, sr).astype(np.float32)
    return x


def main():
    out_dir = sys.argv[1]
    files = sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    total = 0.0
    for i, f in enumerate(files):
        dst = os.path.join(out_dir, os.path.basename(f).replace(".wav", ".npz"))
        if os.path.exists(dst):
            continue
        x = load16k(f)
        total += len(x) / 16000
        audio = torch.from_numpy(x)[None]
        with torch.inference_mode():
            f0, per = torchcrepe.predict(audio, 16000, 160, 50.0, 1200.0, "full", decoder=torchcrepe.decode.viterbi,
                                         return_periodicity=True, batch_size=256, device=dev, pad=True)
        f0 = f0[0].cpu().numpy()
        per = per[0].cpu().numpy()
        del audio
        if dev == "mps":
            torch.mps.empty_cache()
        t = np.arange(len(f0)) * 0.01
        np.savez(dst, t=t, f0=f0, per=per)
        if i % 20 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(files)}  {total:.0f}s de audio em {el:.0f}s ({total/max(el,1e-9):.1f}x)", flush=True)


if __name__ == "__main__":
    main()
