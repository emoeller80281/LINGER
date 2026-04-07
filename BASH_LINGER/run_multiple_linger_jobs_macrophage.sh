#!/bin/bash

SAMPLE_NUMS=(
  # "Macrophase_buffer1_filtered"
  # "Macrophase_buffer2_filtered"

  "buffer_1"
  "buffer_2"
  "buffer_3"
  "buffer_4"
)

# Submit each SAMPLE_NUM as a separate job
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do

  mkdir -p "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/LOGS/${SAMPLE_NUM}"
  sbatch \
    --export=SAMPLE_NUM="$SAMPLE_NUM" \
    --output="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/LOGS/${SAMPLE_NUM}/${SAMPLE_NUM}.out" \
    --error="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/LOGS/${SAMPLE_NUM}/${SAMPLE_NUM}.err" \
    --job-name="LINGER_${SAMPLE_NUM}_MACROPHAGE" \
    ./run_linger_macrophage.sh
done