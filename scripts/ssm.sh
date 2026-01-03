#!/bin/bash -l
#SBATCH --job-name=ssm
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH -e ssm.err
#SBATCH -o ssm.out

export HOME=/scratch_root/wy524
mkdir -p $HOME/.config/wandb

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

python examples/states/train_safe_matching_online.py \
  --wandb True \
  --project_name gymnasium_long \
  --run_name Ant_ssm_test \
  --seed 0 \
  --env_name SafetyAntVelocity-v1 \
  --max_steps 2000000 \
  --epoch_length 2000 \
  --start_training 10000 \
  --eval_interval 2000 \
  --log_interval 1000

