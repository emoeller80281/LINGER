#!/bin/bash

SAMPLE_NUMS=(
  # "K562_human_filtered"
  "sample_1"
)

# Submit each SAMPLE_NUM as a separate job
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do

  mkdir -p "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/LOGS/${SAMPLE_NUM}"
  sbatch \
    --export=SAMPLE_NUM="$SAMPLE_NUM" \
    --output="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/LOGS/${SAMPLE_NUM}/${SAMPLE_NUM}.out" \
    --error="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/LOGS/${SAMPLE_NUM}/${SAMPLE_NUM}.err" \
    --job-name="LINGER_${SAMPLE_NUM}" \
    ./run_linger_k562.sh
done