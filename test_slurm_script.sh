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

# Unet, Binary, Angle
python test.py --username=sea-ice-classification --name=4umqowtc --checkpoint="epoch=11-step=540-v1.ckpt" --n_workers=4

# Resnet34, Binary, Angle
python test.py --username="jt847" --name="815p4954" --checkpoint="epoch=14-step=675-v1.ckpt" --n_workers=4

# Unet, Binary, Ratio
python test.py --username=jtd33 --name=zbpsbm6k --checkpoint="epoch=15-step=720-v1.ckpt" --n_workers=4

# Resnet34, Binary, Ratio
python test.py --username=sea-ice-classification --name=p5p9w15d --checkpoint="epoch=15-step=720-v1.ckpt" --n_workers=4

# Unet, Ternary, Angle
### TBC ###

# Resnet34, Ternary, Angle
python test.py --username="jt847" --name="d5r5lqbu" --checkpoint="epoch=11-step=540-v1.ckpt" --n_workers=4

# Unet, Ternary, Ratio
python test.py --username=sea-ice-classification --name=qjrk5qgr --checkpoint="epoch=1-step=90-v1.ckpt" --n_workers=4

# Resnet34, Ternary, Ratio
python test.py --username=jtd33 --name=522kln68 --checkpoint="epoch=6-step=315-v1.ckpt" --n_workers=4