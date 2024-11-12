import numpy as np
import csv
import configparser
import matplotlib.pyplot as plt
import progressbar
import scipy
from jax import jit
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter1d


def read_config(variables, mode='chaos',file='config.ini'):
    config = configparser.ConfigParser()
    config.read(file)

    if variables != []:
        for i in range(len(variables)):
            variables[i] = float(config[mode][variables[i]])

    global N,size    
    size = int(config[mode]['size'])
    N_max = float(config[mode]['N_max'])
    N_output = float(config[mode]['N_output'])
    sample_rate = float(config[mode]['sample_rate'])
    N = int((N_max - N_output)/sample_rate)
    return N,size,variables
    
def read_scalar(file,shape_of_data):
    #reads the data that is produced by PrintArray from file arrayhandling.c
    data = np.empty(shape_of_data)
    mech_reader = csv.reader(open(file, "r"), delimiter=",")
    print("shape of data: ",data.shape)

    bar = progressbar.ProgressBar(maxval=N, left_justify=True)
    bar.start()
    i = 0
    k = 0
    tot = 0
    for mech_line in mech_reader:
        if not mech_line:
            i += 1
            k = 0
            bar.update(i)
        else:
            arr_mech = np.array(list(mech_line))
            data[0,i,k] = arr_mech
            k += 1
            tot += 1
    bar.finish()
    return data

def read_vector(file,shape_of_data):

    mech_reader = csv.reader(open(file, "r"), delimiter=",")
    i,j,k,t_n=0,0,0,0
    data = np.empty(shape_of_data)
    print("shape of data: ",data.shape)

    bar = progressbar.ProgressBar(maxval=N, left_justify=True)
    bar.start()

    for mech_line in mech_reader:
        if len(mech_line) == 0:
            i += 1
            k = 0
        elif len(mech_line) == 1:
            k = 0
            i = 0
            t_n += 1
            bar.update(t_n)
        else:
            arr_mech = np.array(list(mech_line),float)
            data[t_n,k,i,:] = arr_mech    
            k += 1

    bar.finish()
    return data

def shape_input_for_adoptode(x, x_cm, T_a, i, j,l_a0):


    N,size,ls = read_config(["c_a"])
    c_a = ls

    T_a_arr = np.array([T_a[:,i-1,j-1],T_a[:,i-1,j],T_a[:,i,j],T_a[:,i,j-1]])
    x_cm_i = np.array([x_cm[:,:,i-1,j-1], x_cm[:,:,i-1,j], x_cm[:,:,i,j], x_cm[:,:,i,j-1] ])

    l_a_i = l_a0/(1+ c_a*T_a_arr)

    x_i = x[:,:,i,j]
    x_j = np.array([x[:,:,i-1,j], x[:,:,i,j+1], x[:,:,i+1,j], x[:,:,i,j-1] ])

    for k in range(4):
        x_j[k][:,0] = gaussian_filter1d(x_j[k][:,0], sigma=5)
        x_j[k][:,1] = gaussian_filter1d(x_j[k][:,1], sigma=5)
        x_cm_i[k][:,0] = gaussian_filter1d(x_cm_i[k][:,0], sigma=5)
        x_cm_i[k][:,1] = gaussian_filter1d(x_cm_i[k][:,1], sigma=5)
    x_i[:,0] = gaussian_filter1d(x_i[:,0], sigma=5)
    x_i[:,1] = gaussian_filter1d(x_i[:,1], sigma=5)

    return x_i,x_j, x_cm_i,l_a_i

@jit
def t_to_value_x(x,t_int,t):
    
    delta_t = (t_int[-1]-t_int[0])/(len(t_int))
        
    i = jnp.rint((t-t_int[0])/delta_t).astype(int)
    
    return x[:,i,:]

@jit
def t_to_value_l(x,t_int,t):
    delta_t = (t_int[-1]-t_int[0])/(len(t_int))
        
    i = jnp.rint(jnp.floor((t-t_int[0])/delta_t)).astype(int)
    
    return x[:,i]


@jit
def t_to_value_1p(x,t_int,t):
    delta_t = (t_int[-1]-t_int[0])/(len(t_int))
        
    i = jnp.rint(jnp.floor((t-t_int[0])/delta_t)).astype(int)
    
    return x[i]

def interpolate_x(x,t_eval,m):
    
    x_int = np.zeros((4,len(t_eval)*m,2))
    for i in range(4):
        t_int, x_int[i] = interpolate_spline(x[i],t_eval,m)
        
    return t_int, x_int

def interpolate_spline(arr, t_eval,m):
    # Assuming x and y are the input data points
    y0_eval = arr[:,0]
    interp_points = len(t_eval)*m
    cs0 = scipy.interpolate.CubicSpline(t_eval,y0_eval)
    # Generate interpolated values
    t_interp = np.linspace(t_eval[0], t_eval[-1], interp_points)  
    y0_interp = cs0(t_interp)

    # Assuming x and y are the input data points
    y1_eval = arr[:,1]
    interp_points = len(t_eval)*m   
    cs1 = scipy.interpolate.CubicSpline(t_eval,y1_eval)
    # Generate interpolated values
    t_interp = np.linspace(t_eval[0], t_eval[-1], interp_points)  
    y1_interp = cs1(t_interp)

    return t_interp, jnp.array([y0_interp,y1_interp]).T # transpose to get the shape as input

def interpolate_scalar(arr, t_eval,m):
    # Assuming x and y are the input data points
    
    x_int = np.zeros((4,len(t_eval)*m))
    interp_points = len(t_eval)*m
    # Generate interpolated values
    for i in range(4):    
        cs0 = scipy.interpolate.CubicSpline(t_eval,arr[i,:])
        t_interp = np.linspace(t_eval[0], t_eval[-1], interp_points)  
        x_int[i,:] = cs0(t_interp)


    return t_interp, x_int

#
#helper functions to create and read out dictionaries of the spline coefficients
#

# Function to generate a dictionary of all elements of fq_short.c
def c_to_dict(c):
    params = {}
    keys = []
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            params['c'+str(i)+'_'+str(j)] = c[i, j]
            keys.append('c'+str(i)+'_'+str(j))
    return params, keys

# Function to generate coefficients from dictionary
@jit
def dict_to_c(params, c_initial):
    c = jnp.zeros(c_initial.shape)
    for i in range(c_initial.shape[0]):
        for j in range(c_initial.shape[1]):
            c = c.at[i, j].set(params['c'+str(i)+'_'+str(j)])
    return c


# create a moving average of arr. The resulting array should have the same length as arr

def moving_average(arr,window):
    N = len(arr)
    new_arr = np.zeros(N)
    for i in range(N):
        new_arr[i] = np.mean(arr[max(0,i-window):min(N,i+window)])
    return new_arr


def make_longer_bound(arr,t_int,dt):
    for i in range(int(len(arr)/2)):
        arr = np.append(arr, arr[-1])
        arr = np.insert(arr,0,arr[0])
        t_int = np.append(t_int, t_int[-1]+ dt)
        t_int = np.insert(t_int,0,t_int[0]-dt)
    return arr, t_int
