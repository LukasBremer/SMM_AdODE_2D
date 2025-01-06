import jax.numpy as jnp
from jax import jit 
from jax.flatten_util import ravel_pytree

import numpy as np
import scipy
import matplotlib.pyplot as plt
import interpax

from scipy.signal import find_peaks
from adoptODE import train_adoptODE, simple_simulation, dataset_adoptODE
from HelperAndMechanics import *
import progressbar
import h5py
import argparse
import sys
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('i', type=int)
parser.add_argument('j', type=int)
parser.add_argument('nr', type=str)
parser.add_argument('start_indx',type=int)
args = parser.parse_args()
start_indx = args.start_indx

result = subprocess.run(['python3', 'SinglePeakModel.py', str(args.i), str(args.j),str(args.nr),str(args.start_indx)], capture_output=True, text=True)
eta_arr = np.load('../data/SpringMassModel/EtaSweep/eta_sweep'+args.nr+'.npy')
    
while np.all(np.isnan(eta_arr[args.i,args.j,:,0])):
    start_indx = result.returncode + 1
    result = subprocess.run(['python3', 'SinglePeakModel.py', str(args.i), str(args.j),str(args.nr),str(args.start_indx)], capture_output=True, text=True)
    eta_arr = np.load('../data/SpringMassModel/EtaSweep/eta_sweep'+args.nr+'.npy')