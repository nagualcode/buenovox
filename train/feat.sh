#!/bin/sh
# feat.sh <saida_dir> <stride> [opcoes...] <wav>: entradas da rede para um wav (bvx_cli features)
dir="$1"; stride="$2"; shift 2
eval "wav=\${$#}"
b=$(basename "$wav" .wav)
n=$(($# - 1))
opts=""
i=1
for a in "$@"; do [ $i -le $n ] && opts="$opts $a"; i=$((i+1)); done
exec target/release/bvx_cli features "$wav" "$dir/$b.bin" stride="$stride" $opts
