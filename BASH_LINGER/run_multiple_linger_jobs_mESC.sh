#!/bin/bash

SAMPLE_NUMS=(
    # Need to change filename in 'run_linger_mesc.sh' to run the 70 percent subsamples
    # 70_percent_subsampled_1_E7.5_rep1
    # 70_percent_subsampled_2_E7.5_rep1
    # 70_percent_subsampled_3_E7.5_rep1
    # 70_percent_subsampled_4_E7.5_rep1
    # 70_percent_subsampled_5_E7.5_rep1
    # 70_percent_subsampled_6_E7.5_rep1
    # 70_percent_subsampled_7_E7.5_rep1
    # 70_percent_subsampled_8_E7.5_rep1
    # 70_percent_subsampled_9_E7.5_rep1
    # 70_percent_subsampled_10_E7.5_rep1

    # 70_percent_subsampled_1_E7.5_rep2
    # 70_percent_subsampled_2_E7.5_rep2
    # 70_percent_subsampled_3_E7.5_rep2
    # 70_percent_subsampled_4_E7.5_rep2
    # 70_percent_subsampled_5_E7.5_rep2
    # 70_percent_subsampled_6_E7.5_rep2
    # 70_percent_subsampled_7_E7.5_rep2
    # 70_percent_subsampled_8_E7.5_rep2
    # 70_percent_subsampled_9_E7.5_rep2
    # 70_percent_subsampled_10_E7.5_rep2

    # 70_percent_subsampled_1_E8.5_rep1
    # 70_percent_subsampled_2_E8.5_rep1
    # 70_percent_subsampled_3_E8.5_rep1
    # 70_percent_subsampled_4_E8.5_rep1
    # 70_percent_subsampled_5_E8.5_rep1
    # 70_percent_subsampled_6_E8.5_rep1
    # 70_percent_subsampled_7_E8.5_rep1
    # 70_percent_subsampled_8_E8.5_rep1
    # 70_percent_subsampled_9_E8.5_rep1
    # 70_percent_subsampled_10_E8.5_rep1

    # 70_percent_subsampled_1_E8.5_rep2
    # 70_percent_subsampled_2_E8.5_rep2
    # 70_percent_subsampled_3_E8.5_rep2
    # 70_percent_subsampled_4_E8.5_rep2
    # 70_percent_subsampled_5_E8.5_rep2
    # 70_percent_subsampled_6_E8.5_rep2
    # 70_percent_subsampled_7_E8.5_rep2
    # 70_percent_subsampled_8_E8.5_rep2
    # 70_percent_subsampled_9_E8.5_rep2
    # 70_percent_subsampled_10_E8.5_rep2

    # "1000_cells_E7.5_rep1"
    # "1000_cells_E7.5_rep2"
    # "1000_cells_E7.75_rep1"
    # "1000_cells_E8.0_rep1"
    # "1000_cells_E8.0_rep2"
    # "1000_cells_E8.5_CRISPR_T_KO"
    # "1000_cells_E8.5_CRISPR_T_WT"
    # "1000_cells_E8.5_rep1"
    # "1000_cells_E8.5_rep2"
    # "1000_cells_E8.75_rep1"
    # "1000_cells_E8.75_rep2"
    # "2000_cells_E7.5_rep1"
    # "2000_cells_E8.0_rep1"
    # "2000_cells_E8.0_rep2"
    # "2000_cells_E8.5_CRISPR_T_KO"
    # "2000_cells_E8.5_CRISPR_T_WT"
    # "2000_cells_E8.5_rep1"
    # "2000_cells_E8.5_rep2"
    # "2000_cells_E8.75_rep1"
    # "2000_cells_E8.75_rep2"
    # "3000_cells_E7.5_rep1"
    # "3000_cells_E8.0_rep1"
    # "3000_cells_E8.0_rep2"
    # "3000_cells_E8.5_CRISPR_T_KO"
    # "3000_cells_E8.5_CRISPR_T_WT"
    # "3000_cells_E8.5_rep1"
    # "3000_cells_E8.5_rep2"
    # "3000_cells_E8.75_rep2"
    # "4000_cells_E7.5_rep1"
    # "4000_cells_E8.0_rep1"
    # "4000_cells_E8.0_rep2"
    # "4000_cells_E8.5_CRISPR_T_KO"
    # "4000_cells_E8.5_CRISPR_T_WT"
    # "4000_cells_E8.5_rep1"
    # "4000_cells_E8.5_rep2"
    # "4000_cells_E8.75_rep2"
    # "5000_cells_E7.5_rep1"
    # "5000_cells_E8.5_CRISPR_T_KO"
    # "5000_cells_E8.5_CRISPR_T_WT"
    # "5000_cells_E8.5_rep1"
    # "5000_cells_E8.5_rep2"
    # "filtered_L2_E7.5_rep1"
    # "filtered_L2_E7.5_rep2"
    # "filtered_L2_E7.75_rep1"
    # "filtered_L2_E8.0_rep1"
    # "filtered_L2_E8.0_rep2"
    # "filtered_L2_E8.5_CRISPR_T_KO"
    # "filtered_L2_E8.5_rep1"
    # "filtered_L2_E8.5_rep2"
    # "filtered_L2_E8.75_rep1"
    # "filtered_L2_E8.75_rep2"

    # muon_E7.5_rep1
    # muon_E7.5_rep2
    muon_E8.5_rep1
    muon_E8.5_rep2
)

# Submit each SAMPLE_NUM as a separate job
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do

  mkdir -p "LOGS/${SAMPLE_NUM}"
  sbatch \
    --export=SAMPLE_NUM="$SAMPLE_NUM" \
    --output="LOGS/${SAMPLE_NUM}/${SAMPLE_NUM}.out" \
    --error="LOGS/${SAMPLE_NUM}/${SAMPLE_NUM}.err" \
    --job-name="LINGER_${SAMPLE_NUM}" \
    run_linger_mesc.sh
done