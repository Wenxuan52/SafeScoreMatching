#!/bin/bash -l
#SBATCH --job-name=quadrotor_ssm
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e quadrotor_ssm.err
#SBATCH -o quadrotor_ssm.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

# cd examples

# python examples/quadrotor/train_td3_lag_quad2d_baseline.py \
#   --env_name QuadrotorTracking2D-v0 \
#   --seed 0 \
#   --max_steps 5010000 \
#   --start_training 10000 \
#   --eval_interval 500000 \
#   --eval_episodes 4 \
#   --batch_size 256 \
#   --utd_ratio 1 \
#   --save_interval 500000 \
#   --wandb False

python examples/quadrotor/train_ssm_quad2d.py \
  --mode training \
  --env_name QuadrotorTracking2D-v0 \
  --seed 1 \
  --max_steps 2010000 \
  --start_training 10000 \
  --eval_interval 50000 \
  --eval_episodes 4 \
  --batch_size 512 \
  --utd_ratio 1 \
  --save_interval 50000 \
  --wandb False