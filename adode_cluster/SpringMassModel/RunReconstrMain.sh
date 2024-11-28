#!/bin/bash
#SBATCH -t 04:00:00
#SBATCH -p medium
#SBATCH -o /scratch/users/lbremer/adode_cluster/out/job-%A_%a.out
#SBATCH -e /scratch/users/lbremer/adode_cluster/err/job-%A_%a.err
#SBATCH -C cascadelake
#SBATCH --mem=10G
#SBATCH -a 0-99

i=20
j=50
arr_i=$((SLURM_ARRAY_TASK_ID / 10))
arr_j=$((SLURM_ARRAY_TASK_ID % 10))
no_dt=$((5 + arr_i * 2))
no_points=$((5 + arr_j * 2))
dt=$((500 / no_dt))

module load python 
source ../.venv/bin/activate
python NNRecMain.py $i $j $no_dt $dt $no_points $arr_i $arr_j 