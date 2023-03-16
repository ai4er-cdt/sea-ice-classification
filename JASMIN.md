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

### Step 1: Login to JASMIN
```
# First, connect to the login node
ssh -A <userID>@login1.jasmin.ac.uk

# Then, connect to one of the sci servers (1-8)
ssh -AX <userID@sci4.jasmin.ac.uk

# Finally, connect to the interactice GPU node
ssh -A gpuhost001.jc.rl.ac.uk
```

### Step 2: Create the train SLURM script
Use the train_slurm_script.sh file from this repository.

```
#!/bin/bash
#SBATCH --gres=gpu:4                <-- request 4 GPU nodes
#SBATCH --partition=orchid          <-- request to run on ORCHID
#SBATCH --account=orchid
#SBATCH -o %j.out                   <-- specify the job output file
#SBATCH -e %j.err                   <-- specify the job error file
#SBATCH --time=24:00:00             <-- the minimum is 1 hour, the maximum is 24 hours
#SBATCH --ntasks=1                  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000

# executables
conda activate sea-ice-classification
nvidia-smi

# EXAMPLE: Unet, Binary, Angle
python train.py --model=unet --classification_type=binary --criterion=ce --batch_size=256 --learning_rate=1e-3 --seed=0 --sar_band3=angle --n_workers=4 --devices=4 --max_epochs=20
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


## References

[1] Create a JASMIN account | JASMIN Accounts Portal. Available at: https://accounts.jasmin.ac.uk/application/new/ (Accessed: March 16, 2023). 

[2] Sign in | JASMIN Accounts Portal. Available at: https://accounts.jasmin.ac.uk/services/additional_services/orchid/ (Accessed: March 16, 2023). 
