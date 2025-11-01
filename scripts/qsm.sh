#!/bin/bash -l
#SBATCH --job-name=qsm
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH -e qsm.err
#SBATCH -o qsm.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl  # or your env

python examples/states/train_score_matching_online.py \
  --wandb True \
  --project_name safety_cal \
  --run_name carbutton_qsm_seed0 \
  --seed 0 \
  --env_name SafetyCarButton1-v0 \
  --max_steps 160000 \
  --epoch_length 400 \
  --start_training 10000 \
  --eval_interval 400 \
  --log_interval 400

