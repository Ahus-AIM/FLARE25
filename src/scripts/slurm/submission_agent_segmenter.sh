#!/usr/bin/bash
#SBATCH --account=NAISS2024-22-1677 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 24:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/meta-project/projects/SAM-MED3D/logs/slurm/submission_agent_segmenter_%j.out

module load virtualenv

source /mimer/NOBACKUP/groups/meta-project/envs/medseg/bin/activate

python -m src.submission.CVPR25_iter_eval_nodocker \
    --test_img_path data/CVPR-BiomedSegFM/3D_val_combined_curated \
    --save_path demo_seg/agent_segmenter \
    --input_temp inputs/agent_segmenter \
    --output_temp outputs/agent_segmenter \
    --segmenter_type agent_segmenter \
    --segmenter_checkpoint weights/attention_ppo_threshold/agent
