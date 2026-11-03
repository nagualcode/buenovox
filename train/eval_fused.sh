#!/bin/sh
# eval_fused.sh <pesos.bin> [fusion=...]: decisao YIN+rede no vocadito (anotacao humana)
# e no VocalSet de teste (rotulos do professor).
set -e
cd "$(dirname "$0")/.."
W="$1"; shift
OUT=train/runs/eval_$(basename "$W" .bin)
mkdir -p "$OUT/vocadito" "$OUT/vs"
ls train/data/vocadito/Audio/*.wav | tr '\n' '\0' | xargs -0 -P 8 -n 1 train/feat.sh "$OUT/vocadito" 1 net="$W" "$@"
tr '\n' '\0' < train/data/vs_test.txt | xargs -0 -P 8 -n 1 train/feat.sh "$OUT/vs" 1 net="$W" "$@"
cd train
.venv/bin/python eval_vocadito.py "../$OUT/vocadito"
.venv/bin/python eval_vs.py "../$OUT/vs"
