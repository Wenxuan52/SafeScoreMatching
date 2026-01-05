#!/bin/bash -l
#SBATCH --job-name=visual_quadrotor
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH -e visual_quadrotor.err
#SBATCH -o visual_quadrotor.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

python examples/quadrotor/visualize_policy_trajectory.py \
  --algo ssm \
  --checkpoint_dir results/QuadrotorTracking2D-v0/jaxrl5_quad2d_ssm_baseline/2026-01-05_seed0001 \
  --guidance_mode none \
  --checkpoint_step 1300000 \
  --episodes 1 \
  --deterministic \
  --out_dir results/visualizations/td3 \