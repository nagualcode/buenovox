"""Avalia a decisao gravada (YIN, ou fusao se extraida com net=) contra a anotacao do vocadito."""

import glob
import os
import sys

import numpy as np

from common import fmt, interp_ref, pitch_metrics, read_frames

feat_dir = sys.argv[1] if len(sys.argv) > 1 else "train/data/feat/vocadito"
ann_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/vocadito/Annotations/F0")
refs, ests, nets = [], [], []
for p in sorted(glob.glob(f"{feat_dir}/*.bin")):
    name = os.path.basename(p)[:-4]
    meta = read_frames(p, feats=False)
    ann = np.loadtxt(f"{ann_dir}/{name}_f0.csv", delimiter=",")
    ref = interp_ref(ann[:, 0], ann[:, 1], meta[:, 0])
    refs.append(ref)
    ests.append(meta[:, 2])
    nets.append(np.where(meta[:, 6] > 0.5, meta[:, 5], 0.0))
ref, est = np.concatenate(refs), np.concatenate(ests)
print("decisao gravada:", fmt(pitch_metrics(ref, est)))
if np.any(np.concatenate(nets) > 0):
    print("rede sozinha:   ", fmt(pitch_metrics(ref, np.concatenate(nets))))
