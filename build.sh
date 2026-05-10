#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "==> Compiling Cython extension (lddcalc.pyx)..."
python setup.py build_ext --inplace

echo "==> Compiling C binary (scaling_laws.c)..."
gcc -O2 -o src/analysis/scaling_laws src/analysis/scaling_laws.c -lm

echo "==> Done."
