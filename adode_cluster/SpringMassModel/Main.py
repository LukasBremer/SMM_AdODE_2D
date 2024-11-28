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

T_standard = [1,2,3,4,5,6,7,8,9,10]
print('start threshold 1e-7')


max_indx = 0
for _ in range(2):
    control = False
    for k in range(10):
        print('Peak',T_standard[k])
        result = subprocess.run(['python3', 'SinglePeakModel.py', str(i), str(j), str(k), str(T_standard[k]),str(nr),str(max_indx)], capture_output=True, text=True)
        print(result.stdout)
        loss = np.load('../data/SpringMassModel/EtaSweep/eta_sweep'+str(nr)+'.npy')[i,j,k,1]

        if loss < 1e-7:
            print('success',loss,T_standard[k])
            control = True

    max_indx = result.returncode
    eta_temp = np.load('../data/SpringMassModel/EtaSweep/eta_sweep'+str(nr)+'.npy')[i, j, :, 0]
    eta_temp[np.isnan(eta_temp)] = 1.1
    is_all_zero = np.all(eta_temp == np.full(10,1.1))    

    if is_all_zero or not control:    
        print('start correction loop')
        for shift in range(3):
            for k in range(10):
                result = subprocess.run(['python3', 'SinglePeakModel.py', str(i), str(j), str(k), str(T_standard[k]),str(nr),str(max_indx+ 2* shift + 1)], capture_output=True, text=True)
                print(f"start at {max_indx} + {shift} * 2", result.stdout,T_standard[k])
                loss = np.load('../data/SpringMassModel/EtaSweep/eta_sweep'+str(nr)+'.npy')[i,j,k,1]
                if loss < 1e-7:
                    print('success',loss,T_standard[k])
                    control = True
            if control:
                break
        if control:
            break
        
    if control:
        break