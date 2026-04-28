#!/bin/bash
#SBATCH --job-name=fm_ae_latent
#SBATCH --nodelist=desa
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=72:00:00
#SBATCH --output=/media/ssd2/Programming/_branch/graph_hdc_public/experiments/generation/results/slurm_fm_ae_%j.out
#SBATCH --error=/media/ssd2/Programming/_branch/graph_hdc_public/experiments/generation/results/slurm_fm_ae_%j.err

set -euo pipefail

PROJECT_DIR="/media/ssd2/Programming/_branch/graph_hdc_public"
VENV="${PROJECT_DIR}/.venv/bin/activate"

echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================"

source "${VENV}"

cd "${PROJECT_DIR}"

python experiments/generation/train_flow_matching__ae.py \
    --__DEBUG__ False

echo "=== Job completed at $(date) ==="
