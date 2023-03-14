#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000

# executables
conda activate sea-ice-classification
nvidia-smi
python train.py --model=resnet34 --classification_type=binary --criterion=ce --batch_size=256 --learning_rate=1e-3 --seed=0 --sar_band3=ratio --n_workers=4 --devices=4 --max_epochs=20
