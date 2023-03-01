#!/bin/bash
#
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=32GB
#SBATCH --account=core-rad
#SBATCH -o slurm_logs/slurm%A_%a.log
#SBATCH -e slurm_logs/slurm%A_%a.err
#SBATCH -J dl-rad-age
#SBATCH --partition=jobs-gpu
#SBATCH --array=1-20%1

echo "singularity exec --nv --no-home --pwd /workspace -B /data/core-rad/pwesp/projects/dl-rad-age:/workspace -B /data/core-rad/pwesp/data:/workspace/data /data/core-rad/containers/radler_pytorch_v3.1 bash slurm-cluster/run_script.sh ${1}"
singularity exec --nv --no-home --pwd /workspace -B /data/core-rad/pwesp/projects/dl-rad-age:/workspace -B /data/core-rad/pwesp/data:/workspace/data /data/core-rad/containers/radler_pytorch_v3.1 bash slurm-cluster/run_script.sh ${1}

# If one job from the array reaches this point, kill the entire array
scancel $SLURM_ARRAY_JOB_ID