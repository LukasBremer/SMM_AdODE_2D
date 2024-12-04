#!/bin/bash
#SBATCH -t 04:00:00
#SBATCH -p medium
#SBATCH -o /scratch/users/lbremer/adode_cluster/out/job-%A_%a.out
#SBATCH -e /scratch/users/lbremer/adode_cluster/err/job-%A_%a.err
#SBATCH -C cascadelake
#SBATCH --mem=15G
#SBATCH -a 0-99

i=$((SLURM_ARRAY_TASK_ID / 10))
no_dt=30
no_points=10
dt=20

module load python 
source ../.venv/bin/activate
python NNRecMain.py $i $j $no_dt $dt $no_points 