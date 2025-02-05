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

parser = argparse.ArgumentParser()
parser.add_argument('i', type=int)
parser.add_argument('j', type=int)
parser.add_argument('nr', type=str)
parser.add_argument('start_indx',type=int)
args = parser.parse_args()

print('start subscript')

i,j = 5 + args.i * 10 ,5 + args.j * 10

# Load from HDF5
with h5py.File('../data/SpringMassModel/MechanicalData/data_eta_05_xl.h5', 'r') as f:
    x_temp = f['x_temp'][:]
    x_cm_temp = f['x_cm_temp'][:]
    T = f['T'][:]
    dA = f['dA'][:]
    f.close()
print('loaded data')

N,size,ls = read_config(["l_0","c_a","k_ij","k_j","k_a","m","c_damp","n_0","delta_t_m","it_m","pad"])
N = T.shape[0]
l_0, c_a0, k_g0, k_p0, k_a0, m0, nu0, eta0, delta_t_m, it_m, pad = ls
eta_arr = 1 - np.load('../data/SpringMassModel/FiberOrientation/fiber_orientation.npy')
eta0,eta1,eta2,eta3= eta_arr[i-1,j-1],eta_arr[i-1,j],eta_arr[i,j],eta_arr[i,j-1]
real_params = {'l_g':l_0,'k_g':k_g0,'k_p':k_p0,'k_a':k_a0,'m':m0,'nu':nu0,'eta0':eta0,'eta1':eta1,'eta2':eta2,'eta3':eta3,'c_a': c_a0 }#,'dt':0}

delta_t = delta_t_m * it_m
t_evals = np.linspace(0,N*delta_t,N)
N_interp = int(it_m)*5

t_start_training,t_stop_training = 0,12000#T.shape[0]-3500
diff = 0
start_indx = args.start_indx
while diff < 900: 
    dA_test = dA[t_stop_training+500:,i,j]
    T_test = T[t_stop_training+500:,i,j]
    maxima_temp0, _ = find_peaks(dA_test,prominence=.0007)#,height=.001
    minima_temp0, _ = find_peaks(-dA_test,prominence=.0007)#,height=.001
    max_indx, min_indx = index_finder(maxima_temp0, minima_temp0, dA_test,start_indx)
    diff = minima_temp0[min_indx+1]-maxima_temp0[max_indx]
    start_indx += 1
    
    t_start = t_stop_training + 500 + maxima_temp0[max_indx] - 400
    t_stop = t_stop_training + 500+ minima_temp0[min_indx+1] + 800

T_rec = np.zeros((4,t_stop-t_start))
T_arr_indx = [[i-1,j-1],[i-1,j],[i,j],[i,j-1]]
T_arr = np.array([T[:,i-1,j-1],T[:,i-1,j],T[:,i,j],T[:,i,j-1]])

#define Hyperparameters for data reconstruction
no_dt,no_points = 30,10
delta_t_rec = 20
for indx in range(4):
    i_temp,j_temp = T_arr_indx[indx]
    dA_r_0_der_raw = np.append(np.diff(dA[t_start:t_stop,i_temp,j_temp],axis = 0),np.diff(dA[t_start:t_stop,i_temp,j_temp],axis = 0)[-1])
    
    x_train,x_fit,y_train = shape_data(x_temp,x_cm_temp,dA[t_start_training:t_stop_training,:,:],T[t_start_training:t_stop_training,:,:],dA_r_0_der_raw,i_temp,j_temp,no_dt,no_points,delta_t_rec,key = "dA_r_0_derivative")
    print('calculating element',indx)
    y_out_diff = euclidean_distance_trajectory(x_fit, x_train,y_train)

    T_rec[indx,:] = y_out_diff





with h5py.File('../data/SpringMassModel/StandardPeaks/T_rec.h5', 'w') as f:
    f.create_dataset(str(args.i)+str(args.j), data=T_rec)
    f.close()

