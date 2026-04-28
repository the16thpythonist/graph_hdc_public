#!/bin/bash
#SBATCH --job-name=ae_brics
#SBATCH --nodelist=desa
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=72:00:00
#SBATCH --output=/media/ssd2/Programming/_branch/graph_hdc_public/experiments/generation/results/slurm_ae_brics_%j.out
#SBATCH --error=/media/ssd2/Programming/_branch/graph_hdc_public/experiments/generation/results/slurm_ae_brics_%j.err

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

python experiments/generation/train_autoencoder__brics.py \
    --__DEBUG__ False \
    --VERBOSE False \
    --BATCH_SIZE 256 \
    --LATENT_DIM 512 \
    --TRUNK_DIM 1024 \
    --N_ENCODER_BLOCKS 8 \
    --N_DECODER_BLOCKS 8 \
    --FFN_MULT "8/3" \
    --TRAINING_TARGET "'both'" \
    --VARIATIONAL True \
    --KL_WEIGHT 1e-4 \
    --KL_WARMUP_EPOCHS 50 \
    --EPOCHS 250

echo "=== Job completed at $(date) ==="
