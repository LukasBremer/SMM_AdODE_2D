import numpy as np
import scipy
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import h5py
import progressbar

from HelperAndMechanics import *
from scipy.ndimage import gaussian_filter1d
import jax.numpy as jnp
from jax import jit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('i', type=int)
parser.add_argument('j', type=int)
parser.add_argument('no_dt', type=int)
parser.add_argument('dt', type=int)
parser.add_argument('no_points',type=int)

args = parser.parse_args()
i,j = args.i,args.j
N = 25000
# Load from HDF5
with h5py.File('../data/SpringMassModel/MechanicalData/data_eta_var_xl.h5', 'r') as f:
    x_temp = f['x_temp'][:]
    x_cm_temp = f['x_cm_temp'][:]
    T = f['T'][:]
    dA = f['dA'][:]
    f.close()
#normalize data
x_temp = x_temp-x_cm_temp[0,:,:,:]
dA_diff = np.diff(dA, axis=0)
min_derivative = np.min(dA_diff,axis=0)
max_derivative = np.max(dA_diff,axis=0)
dA_diff = (dA_diff - min_derivative)/(max_derivative-min_derivative)

times = [0,5000,10000,15000]

for t in range(4):
    t_start=times[t]
    t_stop = t_start + 5000
    

    dA_test = dA[t_stop+500:,i,j]
    T_test = T[t_stop+500:,i,j]

    maxima_temp0, _ = find_peaks(dA_test,prominence=.0007)
    minima_temp0, _ = find_peaks(-dA_test,prominence=.0007)
    max_indx, min_indx = index_finder(maxima_temp0, minima_temp0, dA_test,start_indx=0)

    t_start_fit = t_stop + 500 + maxima_temp0[max_indx] - 300
    t_stop_fit = t_stop + 500+ minima_temp0[min_indx+1] + 500

    dA_r_0_raw = dA[t_start_fit:t_stop_fit,i,j] - dA[t_start_fit,i,j]
    dA_r_0_der_raw = np.append(np.diff(dA[t_start_fit:t_stop_fit,i,j],axis = 0),np.diff(dA[t_start_fit:t_stop_fit,i,j],axis = 0)[-1])
    
    #get true data
    T_fit_raw = T[t_start_fit:t_stop_fit,i,j]
    start_loss = int((t_stop_fit - t_start_fit - 1000)/2)
    stop_loss = start_loss + 1000
    #initialize hyperparameters
    no_dt,no_points,dt = args.no_dt,args.no_points,args.dt
    #shape data and train the model
    x_train,x_fit, y_train = shape_data(x_temp,x_cm_temp,dA[t_start:t_stop,:,:],T[t_start:t_stop,:,:],dA_r_0_raw,i,j,no_dt,no_points,dt)
    y_out = euclidean_distance_trajectory(x_fit, x_train,y_train)
    loss = np.nanmean((y_out[start_loss:stop_loss] - T_fit_raw[start_loss:stop_loss])**2)

    x_train,x_fit,y_train = shape_data(x_temp,x_cm_temp,dA[t_start:t_stop,:,:],T[t_start:t_stop,:,:],dA_r_0_der_raw,i,j,no_dt,no_points,dt,key = "dA_r_0_derivative")
    y_out_diff = euclidean_distance_trajectory(x_fit, x_train,y_train)
    loss_diff = np.nanmean((y_out_diff[start_loss:stop_loss] - T_fit_raw[start_loss:stop_loss])**2)
    
    #load data and save the loss
    eval_hyps = np.load('../data/SpringMassModel/NearestNeighbourReconstruction/eval_hyps.npy')
    eval_hyps[i,j,t,0] = loss
    eval_hyps[i,j,t,1] = loss_diff
    np.save('../data/SpringMassModel/NearestNeighbourReconstruction/eval_hyps.npy',eval_hyps)