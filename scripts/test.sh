#!/bin/bash -l
#SBATCH --job-name=cuda_test
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:10:00
#SBATCH -e test.err
#SBATCH -o test.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

python examples/states/train_online.py \
    --env_name SafetyPointGoal1-v0 \
    --seed 0 \
    --max_steps 1000
