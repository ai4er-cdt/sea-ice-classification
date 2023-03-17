# JASMIN Guide
### AI4ER Guided Team Challenge 2023: Sea Ice Classification

## About
JASMIN is a data analysis factility operated by the Science and Technology Facilities Council (STFC).

This is a step-by-step guide to running models on ORCHID, the batch GPU cluster on JASMIN.

## Prerequisites
1. You have a JASMIN account. See https://accounts.jasmin.ac.uk/application/new/
2. You have access to ORCHID, the GPU cluster. See https://accounts.jasmin.ac.uk/services/additional_services/orchid/ 
3. You have cloned this repository
4. You have run tiling.py to generate tiles from the original SAR and ice chart images
5. You have created the conda environment from the environment.yml file
``` 
conda env create -f environment.yml
conda activate sea-ice-classification
```
If the above does not work, you can create the environment manually:
```
conda create --name sea-ice-classification
<import each module individually>
conda activate sea-ice-classification
```

### Step 1: Login to JASMIN
```
# First, connect to the login node
ssh -A <userID>@login1.jasmin.ac.uk

# Then, connect to one of the sci servers (1-8)
ssh -AX <userID>@sci4.jasmin.ac.uk

# Finally, connect to the interactice GPU node
ssh -A gpuhost001.jc.rl.ac.uk
```

## Convolutional Neural Network Models

### Step 2: Create the train SLURM script
Use the train_slurm_script.sh file from this repository.

```
#!/bin/bash
#SBATCH --partition par-multi
#SBATCH --mem 256000
#SBATCH --ntasks 16
#SBATCH --time 48:00:00
#SBATCH --output %j.out
#SBATCH --error %j.err

conda activate sea-ice-classification
python train_scikit.py --sar_folder sar_no_stride --chart_folder chart_no_stride --model DecisionTree --grid_search --sample
```

Note: The SLURM batch queue is 'ORCHID' with Maximum runtime of 24 hours and the default runtime is 1 hour [2].

### Step 3: Run the test SLURM script
To run the script, type
```
sbatch train_slurm_script.sh 
```
This will return the following output
```
Submitted batch job 45403175
```

To check the status of the job, type
```
scontrol show job 
```

### Step 4: Create the test SLURM script

Use the train_slurm_script.sh file from this repository.

```
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

# EXAMPLE: Unet, Binary, Angle
python test.py --username=<wandb_username> --name=<wandb_job_name> --checkpoint="<model_checkpoint>" --n_workers=4
```

### Step 5: Run the test SLURM script
To run the script, type
```
sbatch test_slurm_script.sh
```
This will return the following output
```
Submitted batch job 45403175
```

To check the status of the job, type
```
scontrol show job 
```

## Decision Tree Model
### Step 1: Rerun tiling.py
To regenerate tiles without the sliding window, to avoid data leakage.
```
python tiling.py --stride=None
```

### Step 2: Create the train SLURM script
Use the train_slurm_script_scikit.sh file from this repository.

```
#!/bin/bash
#SBATCH --partition par-multi
#SBATCH --mem 256000
#SBATCH --ntasks 16
#SBATCH --time 48:00:00
#SBATCH --output %j.out
#SBATCH --error %j.err

conda activate sea-ice-classification
python train_scikit.py --sar_folder sar_no_stride --chart_folder chart_no_stride --model DecisionTree --grid_search --sample
```

### Step 3: Run the test SLURM script
To run the script, type
```
sbatch train_slurm_script_scikit.sh 
```
This will return the following output
```
Submitted batch job 45403175
```

To check the status of the job, type
```
scontrol show job 
```

### Step 4: Create the test SLURM script

Use the train_slurm_script.sh file from this repository.

```
#!/bin/bash
#SBATCH --partition par-multi
#SBATCH --mem 256000
#SBATCH --ntasks 16
#SBATCH --time 48:00:00
#SBATCH --output %j.out
#SBATCH --error %j.err

# EXAMPLE: Binary, angle
python test_scikit.py --model_name='' --sar_folder sar_no_stride --chart_folder chart_no_stride --sample --pct_sample 0.1
```

### Step 5: Run the test SLURM script
To run the script, type
```
sbatch test_slurm_script_scikit.sh
```
This will return the following output
```
Submitted batch job 45403175
```

To check the status of the job, type
```
scontrol show job 
```


## References

[1] Create a JASMIN account | JASMIN Accounts Portal. Available at: https://accounts.jasmin.ac.uk/application/new/ (Accessed: March 16, 2023). 

[2] Sign in | JASMIN Accounts Portal. Available at: https://accounts.jasmin.ac.uk/services/additional_services/orchid/ (Accessed: March 16, 2023). 
