#!/bin/bash
#SBATCH --job-name=ae_brics_strict
#SBATCH --nodelist=euler
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=72:00:00
#SBATCH --output=/media/ssd2/Programming/_branch/graph_hdc_public/experiments/generation/results/slurm_ae_brics_euler_%j.out
#SBATCH --error=/media/ssd2/Programming/_branch/graph_hdc_public/experiments/generation/results/slurm_ae_brics_euler_%j.err

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
    --BATCH_SIZE 64 \
    --LATENT_DIM 256 \
    --TRUNK_DIM 512 \
    --N_ENCODER_BLOCKS 8 \
    --N_DECODER_BLOCKS 8 \
    --FFN_MULT "8/3" \
    --TRAINING_TARGET "'both'" \
    --VARIATIONAL True \
    --USE_EMA False \
    --RECON_LOSS_TYPE "'mae'" \
    --DROPOUT 0.05 \
    --LEARNING_RATE 2e-4 \
    --WARMUP_EPOCHS 5 \
    --KL_WEIGHT 2e-4 \
    --KL_WARMUP_EPOCHS 50 \
    --FREE_BITS 0.1 \
    --WEIGHT_DECAY 5e-4 \
    --EPOCHS 250 \
    --USE_GENERIC_LINKING False \
    --ENUMERATE_ATTACHMENTS False

echo "=== Job completed at $(date) ==="
