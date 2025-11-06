#!/bin/bash

SAMPLE_NUMS=(
  "filtered_multiomics_common"
)

# Submit each SAMPLE_NUM as a separate job
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do

  mkdir -p "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/LOGS/${SAMPLE_NUM}/IPS"
  sbatch \
    --export=SAMPLE_NUM="$SAMPLE_NUM" \
    --output="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/LOGS/${SAMPLE_NUM}/IPS/${SAMPLE_NUM}.out" \
    --error="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/LOGS/${SAMPLE_NUM}/IPS/${SAMPLE_NUM}.err" \
    --job-name="LINGER_${SAMPLE_NUM}_IPS" \
    LINGER/BASH_LINGER/run_linger_ips_cell.sh
done