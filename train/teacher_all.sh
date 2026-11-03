#!/bin/sh
# teacher_all.sh: rotula as listas do VocalSet em lotes de 40 arquivos, um processo por lote
# (o torchcrepe na GPU acumula memoria entre arquivos; o processo novo zera).
cd "$(dirname "$0")/.."
cat train/data/vs_test.txt train/data/vs_train.txt | while read -r f; do
  b=$(basename "$f" .wav)
  [ -f "train/data/teacher/$b.npz" ] || echo "$f"
done > train/data/teacher_todo.txt
echo "faltam $(wc -l < train/data/teacher_todo.txt) arquivos"
tr '\n' '\0' < train/data/teacher_todo.txt | xargs -0 -n 40 train/.venv/bin/python -I train/teacher.py train/data/teacher
echo fim
