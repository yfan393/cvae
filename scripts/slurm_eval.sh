#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/slurm_eval.sh
# SLURM batch script for test-set evaluation and ablation studies.
# Submit AFTER training is complete.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8 
#SBATCH -t 2880
#SBATCH -J BaseCode_cvae_eval
#SBATCH -e /data/users1/yfan14/BaseCode/cvae/logs/eval_%j.err
#SBATCH -o /data/users1/yfan14/BaseCode/cvae/logs/eval_%j.out
#SBATCH -A trends53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yfan14@gsu.edu
#SBATCH --oversubscribe
sleep 5s

echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi >&2

# === git log before copying the project ===
cd /data/users1/yfan14/BaseCode/cvae/
mkdir -p logs eval_results ablation_results
echo "Message: $(git log -1 --pretty=%B)" >&2

# cd $JOBDIR
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate 3dunet

CHECKPOINT=saved/C3DVAE/model_best.pth

# ── 1. Full test-set evaluation ───────────────────────────────────────────────
echo "=== Evaluation ==="
python evaluate/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --num_subjects 20 \
    --vis_subjects 3 \
    --save_dir eval_results/ \
    --no_mi

# ── 2. All ablation studies ───────────────────────────────────────────────────
echo "=== Ablations ==="
python ablation/run_ablations.py \
    --checkpoint   "$CHECKPOINT" \
    --num_subjects 20 \
    --ablation     all \
    --save_dir     ablation_results/

echo "Eval + ablations done.  Ended: $(date)"

	
sleep 5s