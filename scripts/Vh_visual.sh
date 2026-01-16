#!/bin/bash -l
#SBATCH --job-name=Vh
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -e Vh_visual.err
#SBATCH -o Vh_visual.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

# 示例 1：RAC
# python examples/quadrotor/plot_vh_region.py \
#   --algo rac --ckpt_path results/QuadrotorTracking2D-v0/jaxrl5_quad2d_rac/2026-01-12_seed0001 --step 2000000 \
#   --grid_n 101 --K 64 --z_dot_list="-1,0,1" \
#   --x_min -1.5 --x_max 1.5 --z_min 0 --z_max 2 \
#   --out examples/quadrotor/figures/vh_rac.png --save_npz examples/quadrotor/figures/vh_rac.npz

# # 示例 2：SSM
python examples/quadrotor/plot_vh_region.py \
  --algo ssm --ckpt_path results/QuadrotorTracking2D-v0/jaxrl5_quad2d_ssm_baseline/2026-01-14_seed0001 --step 2000000 \
  --grid_n 101 --K 64 --z_dot_list="-1,0,1" \
  --out examples/quadrotor/figures/vh_ssm.png

# # 示例 3：SAC-Lag（注意阈值为 cost_limit）
# python examples/quadrotor/plot_vh_region.py \
#   --algo sac_lag --ckpt_path results/QuadrotorTracking2D-v0/jaxrl5_quad2d_sac_lag/2026-01-10_seed0000 --step 2000000 \
#   --grid_n 101 --K 64 --z_dot_list="-1,0,1" \
#   --out examples/quadrotor/figures/vh_saclag.png