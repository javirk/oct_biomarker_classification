#!/bin/bash
#( use ## for comments with SBATCH)
## DON'T USE SPACES AFTER COMMAS

# You must specify a valid email address!
#SBATCH --mail-user=javier.gamazo-tejero@unibe.ch
# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ws_00000

# Job name
#SBATCH --job-name="train_detector"
# Partition
#SBATCH --partition=gpu-invest # all, gpu, phi, long

# Runtime and memory
#SBATCH --time=24:00:00    # days-HH:MM:SS
#SBATCH --mem-per-cpu=4G # it's memory PER CPU, NOT TOTAL RAM! maximum RAM is 246G in total
# total RAM is mem-per-cpu * cpus-per-task

# maximum cores is 20 on all, 10 on long, 24 on gpu, 64 on phi!
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1

# on gpu partition
#SBATCH --gres=gpu:rtx3090:1

# Set the current working directory.
# All relative paths used in the job script are relative to this directory
##SBATCH --workdir=

# create job output file

#SBATCH --output=logs/slurm-%A_%a.out

# For array jobs
# Array job containing 100 tasks, run max 10 tasks at the same time
##SBATCH --array=1-100%10

# Main Python code below this line
python ./train.py -c configs/cfg_detector.yml
