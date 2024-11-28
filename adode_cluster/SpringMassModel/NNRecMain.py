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
parser.add_argument('arr_i',type=int)
parser.add_argument('arr_j',type=int)

args = parser.parse_args()

N = 9000
# Load from HDF5
with h5py.File('../data/SpringMassModel/MechanicalData/data_eta05l.h5', 'r') as f:
    x_temp = f['x_temp'][:]
    x_cm_temp = f['x_cm_temp'][:]
    T = f['T'][:]
    dA = f['dA'][:]
    f.close()
#normalize each point
x_temp = x_temp-x_cm_temp[0,:,:,:]

#T = u
t_start,t_stop = 0,6000
i,j = args.i,args.j

dA_test = dA[t_stop+500:,i,j]
T_test = T[t_stop+500:,i,j]

maxima_temp0, _ = find_peaks(dA_test,prominence=.0007)#,height=.001
minima_temp0, _ = find_peaks(-dA_test,prominence=.0007)#,height=.001
max_indx, min_indx = index_finder(maxima_temp0, minima_temp0, dA_test,start_indx=0)

t_start_fit = t_stop + 500 + maxima_temp0[max_indx] - 300
t_stop_fit = t_stop + 500+ minima_temp0[min_indx+1] + 500

x_arr_fit_raw = np.array([x_temp[t_start_fit:t_stop_fit,:,i,j],x_temp[t_start_fit:t_stop_fit,:,i,j+1],x_temp[t_start_fit:t_stop_fit,:,i+1,j+1],x_temp[t_start_fit:t_stop_fit,:,i+1,j]])
dA_fit_raw = dA[t_start_fit:t_stop_fit,i,j] 
dA_r_05_fit_raw = np.array([dA[t_start_fit:t_stop_fit,i,j],dA[t_start_fit:t_stop_fit,i,j+1],dA[t_start_fit:t_stop_fit,i+1,j],dA[t_start_fit:t_stop_fit,i-1,j],dA[t_start_fit:t_stop_fit,i,j-1]])

number_of_comp,number_of_points,delta_t = args.no_dt,args.no_points,args.dt

x_train,x_fit, y_train = shape_data(x_temp,x_cm_temp,dA[t_start:t_stop,:,:],T[t_start:t_stop,:,:],dA_r_05_fit_raw,i,j,number_of_comp,number_of_points,delta_t)
y_out = euclidean_distance_trajectory(x_fit, x_train,y_train)

T_train_smoothed = gaussian_filter1d(y_out, sigma=10)
T_fit_raw = T[t_start_fit:t_stop_fit,i,j]

T_train_smoothed = gaussian_filter1d(y_out, sigma=10)
T_fit_raw = T[t_start_fit:t_stop_fit,i,j]

#calculate loss between the prediction and the data, y_out is the prediction, T_fit_raw is the data

loss = np.sqrt(np.mean((T_train_smoothed[250:-250] - T_fit_raw[250:-250])**2))
eval_hyps = np.load('../data/SpringMassModel/NearestNeighbourReconstruction/eval_hyps.npy')
eval_hyps[args.arr_i,args.arr_j] = loss
np.save('../data/SpringMassModel/NearestNeighbourReconstruction/eval_hyps.npy',eval_hyps)