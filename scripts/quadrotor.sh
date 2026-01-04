#!/bin/bash -l
#SBATCH --job-name=quadrotor
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH -e quadrotor.err
#SBATCH -o quadrotor.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

# cd examples

# python examples/quadrotor/train_td3_quad2d_baseline.py \
#   --env_name QuadrotorTracking2D-v0 \
#   --seed 0 \
#   --max_steps 2010000 \
#   --start_training 10000 \
#   --eval_interval 50000 \
#   --eval_episodes 5 \
#   --batch_size 256 \
#   --utd_ratio 1 \
#   --save_interval 50000 \
#   --wandb False

python examples/quadrotor/train_ssm_quad2d.py \
  --mode training \
  --env_name QuadrotorTracking2D-v0 \
  --seed 0 \
  --max_steps 1010000 \
  --start_training 10000 \
  --eval_interval 50000 \
  --eval_episodes 5 \
  --batch_size 256 \
  --utd_ratio 1 \
  --save_interval 50000 \
  --wandb False