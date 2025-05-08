#!/usr/bin/bash
#SBATCH --account=NAISS2024-22-1677 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 24:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/meta-project/projects/SAM-MED3D/logs/slurm/train_rl_threshold_%j.out

module load virtualenv

source /mimer/NOBACKUP/groups/meta-project/envs/medseg/bin/activate

python -m src.scripts.train_rl_threshold
