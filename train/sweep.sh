#!/bin/sh
# sweep.sh <pesos.bin> <fusion>...: metricas do vocadito para cada conjunto de limiares
cd "$(dirname "$0")/.."
W="$1"; shift
for F in "$@"; do
  OUT=train/runs/sweep_tmp; rm -rf $OUT; mkdir -p $OUT
  ls train/data/vocadito/Audio/*.wav | tr '\n' '\0' | xargs -0 -P 8 -n 1 train/feat.sh $OUT 1 net="$W" fusion="$F"
  printf "%-28s " "$F"; (cd train && .venv/bin/python eval_vocadito.py ../$OUT | head -1)
done
