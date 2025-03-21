#!/bin/bash
#SBATCH -t 04:00:00
#SBATCH -p medium
#SBATCH -o /scratch/users/lbremer/adode_cluster/out/job-%A_%a.out
#SBATCH -e /scratch/users/lbremer/adode_cluster/err/job-%A_%a.err
#SBATCH -C cascadelake
#SBATCH --mem=25G
#SBATCH -a 0-99

i=$((SLURM_ARRAY_TASK_ID / 10))
j=$((SLURM_ARRAY_TASK_ID % 10))
start_indx=0
nr='T_rec_xl_3'

module load python 
source ../.venv/bin/activate
python MainRec.py $i $j $nr $start_indx