#!/bin/bash -l
#SBATCH --job-name=ssm
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH -e ssm.err
#SBATCH -o ssm.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl  # or your env

python examples/states/train_safe_matching_online.py \
  --wandb True \
  --project_name gymnasium_long \
  --run_name carbutton_ssm_5lag_seed0 \
  --seed 0 \
  --env_name SafetyCarButton1-v0 \
  --max_steps 2000000 \
  --epoch_length 2000 \
  --start_training 10000 \
  --eval_interval 2000 \
  --log_interval 1000

