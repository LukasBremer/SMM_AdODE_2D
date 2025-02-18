#!/bin/bash
#SBATCH --job-name=kernik
#SBATCH --output=/scratch/users/ikottla/Projects/ionic-atrical-model-fit/cluster/out/%j_kernik.out
#SBATCH --error=/scratch/users/ikottla/Projects/ionic-atrical-model-fit/cluster/error/%j_kernik.err
#SBATCH --time 0-01:30:00
#SBATCH --qos 2h
#SBATCH -p gpu
#SBATCH -G RTX5000
#SBATCH --mem 16G

module load python
source ../.venv/bin/activate
python MSD_fit.py
