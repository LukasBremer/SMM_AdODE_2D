#!/bin/bash
#SBATCH -t 0-00:00:00
#SBATCH -p medium
#SBATCH -o /scratch/users/lbremer/out/job-%A_%a.out
#SBATCH -e /scratch/users/lbremer/out/job-%A_%a.err

#SBATCH --mem=100G
#SBATCH -a 1-99

#get every decimal number from the task ID

i=$((SLURM_ARRAY_TASK_ID / 10))
j=$((SLURM_ARRAY_TASK_ID % 10))
k=0

module load python 
python Main.py $i $j $k