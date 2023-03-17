#!/bin/bash
#SBATCH --partition par-multi
#SBATCH --mem 256000
#SBATCH --ntasks 16
#SBATCH --time 48:00:00
#SBATCH --output %j.out
#SBATCH --error %j.err

conda activate /home/users/acz25/miniconda3/envs/sea-ice-classification
python train_scikit.py --sar_folder sar_no_stride --chart_folder chart_no_stride --model DecisionTree --grid_search --sample