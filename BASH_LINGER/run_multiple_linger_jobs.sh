#!/bin/bash -l
#SBATCH --job-name="submit_multiple_linger_jobs"
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --mem=4G

SCRIPT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER"

MAX_JOBS_IN_QUEUE=25

submit_run_linger_job() {
    local CELL_TYPE=$1
    local SAMPLE_NAME=$2
    local SPECIES=$3
    local DATA_DIR=$4
    local BULK_MODEL_DIR=$5
    local METHOD=$6
    local GENOME=$7

    # Ensure the log directory exists
    mkdir -p "LOGS/${SAMPLE_NAME}"

    # Submit the job
    sbatch \
        --export=ALL,CELL_TYPE="$CELL_TYPE",SAMPLE_NAME="$SAMPLE_NAME",SPECIES="$SPECIES",DATA_DIR="$DATA_DIR",BULK_MODEL_DIR="$BULK_MODEL_DIR",METHOD="$METHOD",GENOME="$GENOME" \
        --output="LOGS/${SAMPLE_NAME}/${SAMPLE_NAME}.out" \
        --error="LOGS/${SAMPLE_NAME}/${SAMPLE_NAME}.err" \
        --job-name="LINGER_${CELL_TYPE}_${SAMPLE_NAME}" \
        "${SCRIPT_DIR}/run_linger.sh"
}

run_mESC(){
    local CELL_TYPE="mESC"

    local SAMPLE_NAMES=(
        # "muon_E7.5_rep1"
        # "muon_E7.5_rep2"
        "muon_E8.5_rep1"
        "muon_E8.5_rep2"
    )
    local SPECIES="mouse"
    local DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA"
    local BULK_MODEL_DIR=""
    local METHOD="scNN"
    local GENOME="mm10"

    # Submit each SAMPLE_NAME as a separate job
    for SAMPLE_NAME in "${SAMPLE_NAMES[@]}"; do
        # Check how many jobs are currently queued/running
        while [ "$(squeue -u $USER | grep LINGER | wc -l)" -ge "$MAX_JOBS_IN_QUEUE" ]; do
            echo "[INFO] Maximum jobs ($MAX_JOBS_IN_QUEUE) in queue. Waiting 60 seconds..."
            sleep 60
        done

        # Submit the job for each sample
        submit_run_linger_job \
            "$CELL_TYPE" \
            "$SAMPLE_NAME" \
            "$SPECIES" \
            "$DATA_DIR" \
            "$BULK_MODEL_DIR" \
            "$METHOD" \
            "$GENOME"

    done
}

run_macrophage(){
    local CELL_TYPE="macrophage"

    local SAMPLE_NAMES=(
        # "muon_buffer_1"
        # "muon_buffer_2"
        "muon_buffer_3"
        "muon_buffer_4"
    )
    local SPECIES="human"
    local DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA"
    local BULK_MODEL_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_BULK_MODEL"
    local METHOD="scNN"
    local GENOME="hg38"

    # Motif and TSS information for homer
    local TSS_MOTIF_INFO_PATH="${DATA_DIR}/LINGER_OTHER_SPECIES_TF_MOTIF_DATA/provide_data/"

    # Submit each SAMPLE_NAME as a separate job
    for SAMPLE_NAME in "${SAMPLE_NAMES[@]}"; do
        # Check how many jobs are currently queued/running
        while [ "$(squeue -u $USER | grep LINGER | wc -l)" -ge "$MAX_JOBS_IN_QUEUE" ]; do
            echo "[INFO] Maximum jobs ($MAX_JOBS_IN_QUEUE) in queue. Waiting 60 seconds..."
            sleep 60
        done

        # Submit the job for each sample
        submit_run_linger_job \
            "$CELL_TYPE" \
            "$SAMPLE_NAME" \
            "$SPECIES" \
            "$DATA_DIR" \
            "$BULK_MODEL_DIR" \
            "$METHOD" \
            "$GENOME"
    done
}

run_k562(){
    local CELL_TYPE="K562"

    local SAMPLE_NAMES=(
        "muon_sample_1"
    )
    local SPECIES="human"
    local DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA"
    local BULK_MODEL_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_BULK_MODEL"
    local METHOD="scNN"
    local GENOME="hg38"

    # Motif and TSS information for homer
    local TSS_MOTIF_INFO_PATH="${DATA_DIR}/LINGER_OTHER_SPECIES_TF_MOTIF_DATA/provide_data/"

    # Submit each SAMPLE_NAME as a separate job
    for SAMPLE_NAME in "${SAMPLE_NAMES[@]}"; do
        # Check how many jobs are currently queued/running
        while [ "$(squeue -u $USER | grep LINGER | wc -l)" -ge "$MAX_JOBS_IN_QUEUE" ]; do
            echo "[INFO] Maximum jobs ($MAX_JOBS_IN_QUEUE) in queue. Waiting 60 seconds..."
            sleep 60
        done

        # Submit the job for each sample
        submit_run_linger_job \
            "$CELL_TYPE" \
            "$SAMPLE_NAME" \
            "$SPECIES" \
            "$DATA_DIR" \
            "$BULK_MODEL_DIR" \
            "$METHOD" \
            "$GENOME"
    done
}

run_iPSC(){
    local CELL_TYPE="iPSC"

    local SAMPLE_NAMES=(
        "muon_WT_D13_rep1"
    )
    local SPECIES="human"
    local DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA"
    local BULK_MODEL_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_BULK_MODEL"
    local METHOD="scNN"
    local GENOME="hg38"

    # Motif and TSS information for homer
    local TSS_MOTIF_INFO_PATH="${DATA_DIR}/LINGER_OTHER_SPECIES_TF_MOTIF_DATA/provide_data/"

    # Submit each SAMPLE_NAME as a separate job
    for SAMPLE_NAME in "${SAMPLE_NAMES[@]}"; do
        # Check how many jobs are currently queued/running
        while [ "$(squeue -u $USER | grep LINGER | wc -l)" -ge "$MAX_JOBS_IN_QUEUE" ]; do
            echo "[INFO] Maximum jobs ($MAX_JOBS_IN_QUEUE) in queue. Waiting 60 seconds..."
            sleep 60
        done

        # Submit the job for each sample
        submit_run_linger_job \
            "$CELL_TYPE" \
            "$SAMPLE_NAME" \
            "$SPECIES" \
            "$DATA_DIR" \
            "$BULK_MODEL_DIR" \
            "$METHOD" \
            "$GENOME"
    done
}


run_mESC
run_macrophage
run_k562