#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_output.txt
#SBATCH --error=slurm_error.txt
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:09:00
#SBATCH --account=g34

export MASTER_ADDR=$(hostname)
export MASTER_PORT=1234

srun megatron_deepspeed_configs.sh
