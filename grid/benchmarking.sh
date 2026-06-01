#!/bin/bash

cd "C:\Users\jr8037\Desktop\pypsa_eur_data\scripts"

python benchmark_grid_reduction.py \
  --raw-buses "C:\Users\jr8037\Desktop\pypsa_eur_data\xiong2025 v07\buses.csv" \
  --raw-lines "C:\Users\jr8037\Desktop\pypsa_eur_data\xiong2025 v07\lines.csv" \
  --plants-with-bus "C:\Users\jr8037\Desktop\pypsa_eur_data\modified\plants_with_bus.csv" \
  --reductions-root "C:\Users\jr8037\Desktop\pypsa_eur_data\modified" \
  --out-dir "C:\Users\jr8037\Desktop\pypsa_eur_data\modified\benchmark_results" \
  --n-transfer-samples 250 \
  --transfer-injection-mw 1000 \
  --multiinj-k-active-min 3 \
  --multiinj-k-active-max 8 \
  --multiinj-injection-mw 1000


