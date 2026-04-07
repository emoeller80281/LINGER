#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=3G

set -euo pipefail

# ==========================================
#             USER VARIABLES
# ==========================================

# Conda environment name
CONDA_ENV_NAME="LINGER"

ACTIVEF='ReLU'

# Scripts and data paths
SCRIPTS_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER"
RESULTS_DIR="/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/${CELL_TYPE}_RESULTS"

TSS_MOTIF_INFO_PATH="${DATA_DIR}/LINGER_OTHER_SPECIES_TF_MOTIF_DATA/provide_data/"

# Sample-specific variables (you must export SAMPLE_NAME before running)
RNA_DATA_PATH="${DATA_DIR}/${SAMPLE_NAME}/${SAMPLE_NAME}_RNA.csv"
ATAC_DATA_PATH="${DATA_DIR}/${SAMPLE_NAME}/${SAMPLE_NAME}_ATAC.csv"

# Paths in the results directory
SAMPLE_RESULTS_DIR="${RESULTS_DIR}/${SAMPLE_NAME}"
SAMPLE_DATA_DIR="${DATA_DIR}/LINGER_TRAINED_MODELS/${SAMPLE_NAME}"

LOG_DIR="${SCRIPTS_DIR}/LOGS"

# ==========================================
#             SETUP FUNCTIONS
# ==========================================

validate_critical_variables() {
    echo "[INFO] Validating required variables..."
    local required_vars=(
        SAMPLE_NAME
        RNA_DATA_PATH
        ATAC_DATA_PATH
        SCRIPTS_DIR
        DATA_DIR
        RESULTS_DIR
    )
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            echo "[ERROR] Required variable '$var' is not set."
            exit 1
        fi
    done
}

check_for_running_jobs() {
    echo ""
    echo "[INFO] Checking for running jobs with the same name..."
    if [ -n "${SLURM_JOB_NAME:-}" ]; then
        local running_jobs
        running_jobs=$(squeue --name="$SLURM_JOB_NAME" --noheader | wc -l)
        if [ "$running_jobs" -gt 1 ]; then
            echo "[ERROR] Another job with the same name is already running. Exiting."
            exit 1
        else
            echo "    - No conflicting jobs detected."
        fi
    else
        echo "    - Not running under SLURM, skipping job check."
    fi
}

check_tools() {
    echo ""
    echo "[INFO] Checking for required tools (python3, conda)..."
    for tool in python3 conda; do
        if ! command -v "$tool" &> /dev/null; then
            echo "[ERROR] Required tool '$tool' is not installed or not in PATH."
            exit 1
        else
            echo "    - Found $tool"
        fi
    done
}

activate_conda_env() {
    echo ""
    echo "[INFO] Activating Conda environment '$CONDA_ENV_NAME'..."
    if ! conda activate "$CONDA_ENV_NAME"; then
        echo "[ERROR] Could not activate Conda environment '$CONDA_ENV_NAME'."
        exit 1
    fi
    echo "    - Successfully activated Conda environment."
}

check_input_files() {
    echo ""
    echo "[INFO] Checking if input RNA and ATAC files exist..."
    for file in "$RNA_DATA_PATH" "$ATAC_DATA_PATH"; do
        if [ ! -f "$file" ]; then
            echo "[ERROR] Missing input file: $file"
            exit 1
        else
            echo "    - Found $file"
        fi
    done
}

setup_directories() {
    echo ""
    echo "[INFO] Setting up necessary directories..."
    mkdir -p "$SAMPLE_RESULTS_DIR" "$SAMPLE_DATA_DIR"
    echo "    - Directories created."
}

set_slurm_job_name() {
    echo ""
    echo "[INFO] Setting dynamic SLURM job name..."
    scontrol update JobID="$SLURM_JOB_ID" JobName="LINGER_${SAMPLE_NAME}"
}

check_or_install_tss_locations() {
    # Checks for the TSS location information for species other than hg38
    echo "  1) Checking for other species TSS location directory..."

    if [ ! -d "${TSS_MOTIF_INFO_PATH}" ]; then
        echo "[WARN] TSS location not found at ${TSS_MOTIF_INFO_PATH}"
        echo "    - Attempting to download and extract..."

        confirm_code=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
            'https://drive.usercontent.google.com/download?id=1Dog5JTS_SNIoa5aohgZmOWXrTUuAKHXV' -O- | \
            sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')

        wget --load-cookies /tmp/cookies.txt \
            "https://drive.usercontent.google.com/download?export=download&confirm=${confirm_code}&id=1Dog5JTS_SNIoa5aohgZmOWXrTUuAKHXV" \
            -O "${DATA_DIR}/LINGER_OTHER_SPECIES_TF_MOTIF_DATA/provide_data.tar.gz"

        rm -rf /tmp/cookies.txt

        tar -xzf "${DATA_DIR}/LINGER_OTHER_SPECIES_TF_MOTIF_DATA/provide_data.tar.gz" -C "${DATA_DIR}"

        echo "    - Extraction complete."
    else
        echo "    - TSS location information exists"
    fi
}


check_homer() {
    echo "  2) Checking Homer installation..."

    # Check if 'homer' is installed
    if ! command -v homer &> /dev/null; then
        echo "[ERROR] Homer is not installed or not in PATH."
        echo "    - Installing Homer..."
        
        # You can define your install directory here
        HOMER_INSTALL_DIR="${HOME}/homer"

        mkdir -p "$HOMER_INSTALL_DIR"
        cd "$HOMER_INSTALL_DIR" || exit 1

        # Download and install Homer
        wget http://homer.ucsd.edu/homer/configureHomer.pl
        perl configureHomer.pl -install

        # Add to PATH temporarily (you should add permanently too)
        export PATH="$HOMER_INSTALL_DIR/bin:$PATH"

        echo "    - Homer installed at $HOMER_INSTALL_DIR"
    else
        echo "    - Homer already installed."
    fi

    HOMER_BIN=$(dirname "$(which homer)")
    HOMER_BASE=$(realpath "${HOMER_BIN}/..")
    HOMER_CONFIG="${HOMER_BASE}/share/homer/config.txt"

    # Check if Homer is installed
    if [ ! -f "$HOMER_CONFIG" ]; then
        echo "[ERROR] HOMER config.txt not found at $HOMER_CONFIG"
        echo "Make sure Homer is properly installed."
        exit 1
    fi

    # Check if GENOME exists in config.txt
    if grep -q "${GENOME}" "$HOMER_CONFIG"; then
        echo "[INFO] Homer genome '${GENOME}' already installed."
    else
        echo "[WARN] Homer genome '${GENOME}' not found. Installing..."
        perl "${HOMER_BASE}/share/homer/configureHomer.pl" -install "${GENOME}" \
            2> "${LOG_DIR}/${SAMPLE_NAME}/install_homer_species.log"
        
        if grep -q "${GENOME}" "$HOMER_CONFIG"; then
            echo "[INFO] Successfully installed '${GENOME}'."
        else
            echo "[ERROR] Failed to install '${GENOME}'. Check log: ${LOG_DIR}/${SAMPLE_NAME}/install_homer_species.log"
            exit 1
        fi
    fi

    # Add homer/bin to PATH if not already
    if [[ ":$PATH:" != *":${HOMER_BASE}/homer/bin:"* ]]; then
        echo "[INFO] Adding Homer bin to PATH."
        export PATH="${PATH}:${HOMER_BASE}/homer/bin"
    fi
}

determine_num_cpus() {
    echo ""
    echo "[INFO] Checking the number of CPUs available for parallel processing"
    if [ -z "${SLURM_CPUS_PER_TASK:-}" ]; then
        if command -v nproc &> /dev/null; then
            TOTAL_CPUS=$(nproc --all)
            case $TOTAL_CPUS in
                [1-15]) IGNORED_CPUS=1 ;;  # Reserve 1 CPU for <=15 cores
                [16-31]) IGNORED_CPUS=2 ;; # Reserve 2 CPUs for <=31 cores
                *) IGNORED_CPUS=4 ;;       # Reserve 4 CPUs for >=32 cores
            esac
            NUM_CPU=$((TOTAL_CPUS - IGNORED_CPUS))
            echo "    - Running locally. Detected $TOTAL_CPUS CPUs, reserving $IGNORED_CPUS for system tasks. Using $NUM_CPU CPUs."
        else
            NUM_CPU=1  # Fallback
            echo "    - Running locally. Unable to detect CPUs, defaulting to $NUM_CPU CPU."
        fi
    else
        NUM_CPU=${SLURM_CPUS_PER_TASK}
        echo "    - Running on SLURM. Number of CPUs allocated: ${NUM_CPU}"
    fi
}


# ==========================================
#             PIPELINE STEPS
# ==========================================

# Function to run each Python step with timing and memory tracking
run_step() {
  step_name=$1
  script_path=$2
  shift 2  # Remove the first two arguments
  echo ""
  echo "Running $step_name..."
  /usr/bin/time -v python3 "$script_path" "$@" 2> "${LOG_DIR}/${SAMPLE_NAME}/${step_name}_time_mem.log"
  echo "    Done!"
}

run_pipeline() {
    # #Run each step of the pipeline with resource tracking
    run_step "Step_010.Linger_Load_Data" "${SCRIPTS_DIR}/Step_010.Linger_Load_Data.py" \
        --rna_data_path "$RNA_DATA_PATH" \
        --atac_data_path "$ATAC_DATA_PATH" \
        --data_dir "$DATA_DIR" \
        --sample_data_dir "$SAMPLE_DATA_DIR" \
        --organism "$SPECIES" \
        --bulk_model_dir "$BULK_MODEL_DIR" \
        --cell_type "$CELL_TYPE" \
        --genome "$GENOME" \
        --method "$METHOD"

    run_step "Step_020.Linger_Training" "${SCRIPTS_DIR}/Step_020.Linger_Training.py" \
        --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
        --genome "$GENOME" \
        --method "$METHOD" \
        --sample_data_dir "$SAMPLE_DATA_DIR" \
        --activef "$ACTIVEF" \
        --organism "$SPECIES" \
        --bulk_model_dir "$BULK_MODEL_DIR"

    run_step "Step_030.Create_Cell_Population_GRN" "${SCRIPTS_DIR}/Step_030.Create_Cell_Population_GRN.py" \
        --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
        --genome "$GENOME" \
        --method "$METHOD" \
        --sample_data_dir "$SAMPLE_DATA_DIR" \
        --activef "$ACTIVEF" \
        --organism "$SPECIES" 

    run_step "Step_040.Homer_Motif_Finding" "${SCRIPTS_DIR}/Step_040.Homer_Motif_Finding.py" \
        --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
        --sample_data_dir "$SAMPLE_DATA_DIR" \
        --genome "$GENOME"

    run_step "Step_050.Create_Cell_Type_GRN" "${SCRIPTS_DIR}/Step_050.Create_Cell_Type_GRN.py" \
        --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
        --genome "$GENOME" \
        --method "$METHOD" \
        --sample_data_dir "$SAMPLE_DATA_DIR" \
        --CELL_TYPE "$CELL_TYPE" \
        --organism "$SPECIES"

    GRN_FILE="${SAMPLE_DATA_DIR}/cell_type_specific_trans_regulatory_${CELL_TYPE}.txt"
    if [ -f "${GRN_FILE}" ]; then
        mkdir -p "LINGER_INFERRED_GRNS/"
        cp "${GRN_FILE}" "LINGER_INFERRED_GRNS/linger_${CELLTYPE}_${SAMPLE_NUM}.tsv"
    fi

    # run_step "Step_055.Create_Cell_Level_GRN.py" "${SCRIPTS_DIR}/Step_055.Create_Cell_Level_GRN.py" \
    #     --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
    #     --genome "$GENOME" \
    #     --method "$METHOD" \
    #     --sample_data_dir "$SAMPLE_DATA_DIR" \
    #     --CELL_TYPE "$CELL_TYPE" \
    #     --organism "$SPECIES" \
    #     --num_cpus $NUM_CPU \
    #     --num_cells 1000
}



# ==========================================
#               MAIN
# ==========================================

echo "===== RUNNING VALIDATION CHECKS ====="
validate_critical_variables
check_for_running_jobs
check_tools
check_input_files
activate_conda_env
setup_directories
set_slurm_job_name
determine_num_cpus

if [ "${SPECIES:-}" != "human" ]; then
    echo ""
    echo "[INFO] Organism is not set to 'human'. Checking for Homer installation and TSS location file"
    check_or_install_tss_locations
    check_homer
fi

echo ""
echo "===== CHECKS COMPLETE: STARTING MAIN PIPELINE ====="
run_pipeline
