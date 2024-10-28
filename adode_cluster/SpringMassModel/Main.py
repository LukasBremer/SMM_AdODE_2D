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
import argparse
import subprocess
import sys
from subprocess import run

i = int(sys.argv[1])
j = int(sys.argv[2])
nr = int(sys.argv[3])

T_standard = [1,2,3,4,5]
print('start threshold 5e-7')

max_shift = 0
for _ in range(1):
    control = False
    for k in range(2,3):
        print(k)
        result = subprocess.run(['python3', 'SinglePeakModel.py', str(i), str(j), str(k), str(T_standard[k]),str(nr),str(max_indx)], capture_output=True, text=True)
        max_indx = result.returncode
        print('maxindx',max_indx)
        print(result.stdout)
        loss = np.load('../data/SpringMassModel/EtaSweep/eta_sweep'+str(nr)+'.npy')[i,j,k,1]
        print(result.stdout)
        eta_temp = np.load('../data/SpringMassModel/EtaSweep/eta_sweep'+str(nr)+'.npy')[i, j, :, 0]
        eta_temp[np.isnan(eta_temp)] = 0

        is_all_zero = np.all(eta_temp == np.array([0, 0, 0, 0, 0]))
        is_all_one = np.all(eta_temp == np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        
        for i in range(5):
            max_indx += 1
            result = subprocess.run(['python3', 'SinglePeakModel.py', str(i), str(j), str(k), str(T_standard[k]),str(nr),str(max_indx)], capture_output=True, text=True)
            max_indx = result.returncode
            if loss < 5e-7:
                print('success',loss,T_standard[k])
                control = True
                break
        
        if loss < 5e-7:
            print('success',loss,T_standard[k])
            control = True
        
    if control:
        break