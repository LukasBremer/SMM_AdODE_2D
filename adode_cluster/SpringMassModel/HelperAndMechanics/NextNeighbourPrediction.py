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


def euclidean_distance_trajectory(x_arr_fit_flattened, x_arr_train_flattened,y_train):
    '''calculate the euclidean distance between the fit and training data. choses shorfit eucledean distance
    shape input 
    x_arr_fit_flattened: (Dim_x, number_of_comp, 1,len_t_fit)
    x_arr_train_flattened: (Dim_x, number_of_comp, number_of_data_points,len_t_train)
    returns an tuple of indices that gives the index of the corresponding training data and time'''

    x_arr_train_flattened = jnp.array(x_arr_train_flattened)
    x_arr_fit_flattened = jnp.array(x_arr_fit_flattened)

    bar = progressbar.ProgressBar(maxval=x_arr_fit_flattened.shape[3], \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    #show the time that is to be expected for the loop


    
    y_out = []
    for current_index in range(x_arr_fit_flattened.shape[3]):
        bar.update(current_index)
        index = euclidean_dist_top_k(x_arr_fit_flattened, x_arr_train_flattened,current_index,y_train)
        y_out.append(index)
    
    return y_out

@jit
def euclidean_dist_top_k(x_arr_fit_flattened, x_arr_train_flattened, current_index,y_train):
    """
    Calculate the Euclidean distances between fit and training data, 
    and find the indices of the k smallest distances.

    Parameters:
    x_arr_fit_flattened: ndarray (Dim_x, number_of_comp, 1, len_t_fit)
        fit data array, flattened along the time dimension.
    x_arr_train_flattened: ndarray (Dim_x, number_of_comp, number_of_data_points, len_t_train)
        Training data array, flattened along the time dimension.
    current_index: int
        Index of the current time step in the fit data.
    k: int
        Number of smallest distances to retrieve.

    Returns:
    tuple
        Indices of the k smallest distances in the training data (as multi-dimensional indices).
    """
    x_current = x_arr_fit_flattened[:, :, 0, current_index]
    x_current_broadcasted = x_current.reshape(x_current.shape[0], x_current.shape[1], 1, 1)
    k=100
    
    squared_differences = (x_arr_train_flattened - x_current_broadcasted) ** 2
    
    sum_squared_differences = jnp.sum(squared_differences, axis=0)  
    sum_squared_differences = jnp.sum(sum_squared_differences, axis=0)  
    
    flattened_distances = sum_squared_differences.flatten()
    
    # Find the k smallest distances using jnp.argpartition
    smallest_indices_flat = jnp.argpartition(flattened_distances, k)[:k]  # Indices of k smallest distances
    
    # Sort the k smallest distances for ordered output
    sorted_indices_flat = smallest_indices_flat[jnp.argsort(flattened_distances[smallest_indices_flat])]
    # Convert flattened indices to multi-dimensional indices
    indices = jnp.unravel_index(sorted_indices_flat, sum_squared_differences.shape)
    next_neighbours = y_train[indices]
    weigths = (1.1-(sum_squared_differences[indices]**2/jnp.max(sum_squared_differences[indices])**2))**4
    avg = jnp.sum(next_neighbours*weigths)/jnp.sum(weigths)
    
    #jnp.average(y_train[indices])
    return avg

def shape_data(x_temp,x_cm_temp,dA,y_train_raw,x_arr_fit_raw,i,j,number_of_comp,number_of_points,delta_t):
    '''takes in the raw data and shapes it into the correct format for euclidean_distance_trajectory
    shape input
    x_arr_train_flattened_raw: (Dim_x, 1,1, len_t_train)
    x_arr_fit_flattened_raw: (Dim_x, 1,1, len_t_fit)'''
    key = "dA_r_1"
    if key == "x":
        x_arr_train_raw = np.array([x_temp[:,:,i,j],x_temp[:,:,i,j+1],x_temp[:,:,i+1,j+1],x_temp[:,:,i+1,j]])
        x_arr_train = np.swapaxes(x_arr_train_raw,1,2)
        x_arr_train_flattened_raw = x_arr_train.reshape(x_arr_train.shape[0]*x_arr_train.shape[1], 1 ,1, -1)       

        #x_arr_fit_raw = np.array([x_temp[t_start_fit:t_stop_fit,:,i,j],x_temp[t_start_fit:t_stop_fit,:,i,j+1],x_temp[t_start_fit:t_stop_fit,:,i+1,j+1],x_temp[t_start_fit:t_stop_fit,:,i+1,j]])
        x_arr_fit = np.swapaxes(x_arr_fit_raw,1,2)
        x_arr_fit_flattened_raw = x_arr_fit.reshape(x_arr_train.shape[0]*x_arr_train.shape[1], 1, 1, -1)          
        x_arr_fit_flattened = x_arr_fit_flattened_raw

        y_train = y_train_raw[:,i,j].reshape(y_train_raw.shape[0],1)
        a = int(100/number_of_points)
        for i_y in range(number_of_points-1):
            for i_x in range(number_of_points-1):
                i = int(a/2) + i_y * a
                j = int(a/2) + i_x * a
                x_arr_train_temp = np.swapaxes(np.array([x_temp[:,:,i,j],x_temp[:,:,i,j+1],x_temp[:,:,i+1,j+1],x_temp[:,:,i+1,j]]),1,2)
                x_arr_train_flattened_temp = x_arr_train_temp.reshape(x_arr_train.shape[0]*x_arr_train.shape[1], 1 ,1, -1)    
                x_arr_train_flattened_raw = np.concatenate((x_arr_train_flattened_raw ,x_arr_train_flattened_temp),axis = 2) 
                y_train = np.concatenate((y_train,y_train_raw[:,i,j].reshape(y_train.shape[0],1)),axis = 1)

    if key == "dA_r_0":
        x_arr_train = np.array([dA[:,i,j]])
        x_arr_train_flattened_raw = x_arr_train.reshape(x_arr_train.shape[0], 1 ,1, -1)       

        x_arr_fit_flattened_raw = x_arr_fit_raw.reshape(x_arr_train.shape[0 ], 1, 1, -1)          
        x_arr_fit_flattened = x_arr_fit_flattened_raw
        y_train = y_train_raw[:,i,j].reshape(y_train_raw.shape[0],1)

        a = int(100/number_of_points)
        for i_y in range(number_of_points-1):
            for i_x in range(number_of_points-1):
                i = int(a/2) + i_y * a
                j = int(a/2) + i_x * a
                x_arr_train_temp = np.array([dA[:,i,j]])
                x_arr_train_flattened_temp = x_arr_train_temp.reshape(x_arr_train.shape[0], 1 ,1, -1)    
                x_arr_train_flattened_raw = np.concatenate((x_arr_train_flattened_raw ,x_arr_train_flattened_temp),axis = 2) 
                y_train = np.concatenate((y_train,y_train_raw[:,i,j].reshape(y_train.shape[0],1)),axis = 1)
    
    if key == "dA_r_1":
        x_arr_train = np.array([dA[:,i,j],dA[:,i,j+1],dA[:,i+1,j],dA[:,i-1,j],dA[:,i,j-1]])
        x_arr_train_flattened_raw = x_arr_train.reshape(x_arr_train.shape[0], 1 ,1, -1)       

        x_arr_fit_flattened_raw = x_arr_fit_raw.reshape(x_arr_train.shape[0 ], 1, 1, -1)          
        x_arr_fit_flattened = x_arr_fit_flattened_raw
        y_train = y_train_raw[:,i,j].reshape(y_train_raw.shape[0],1)

        a = int(100/number_of_points)
        for i_y in range(number_of_points-1):
            for i_x in range(number_of_points-1):
                i = int(a/2) + i_y * a
                j = int(a/2) + i_x * a
                x_arr_train_temp = np.array([dA[:,i,j],dA[:,i,j+1],dA[:,i+1,j],dA[:,i-1,j],dA[:,i,j-1]])
                x_arr_train_flattened_temp = x_arr_train_temp.reshape(x_arr_train.shape[0], 1 ,1, -1)    
                x_arr_train_flattened_raw = np.concatenate((x_arr_train_flattened_raw ,x_arr_train_flattened_temp),axis = 2) 
                y_train = np.concatenate((y_train,y_train_raw[:,i,j].reshape(y_train.shape[0],1)),axis = 1)

    x_arr_train_flattened = x_arr_train_flattened_raw
    x_arr_fit_flattened = x_arr_fit_flattened_raw

    #concentate different time shifts of the data
    for shift in range(-int(number_of_comp/2),int(number_of_comp/2)+1):
        if shift == 0:
            continue
        else:
            x_arr_train_flattened_shifted = np.roll(x_arr_train_flattened_raw,shift * delta_t, axis=3)
            x_arr_fit_flattened_shifted = np.roll(x_arr_fit_flattened_raw,shift * delta_t, axis=3)
            x_arr_train_flattened = np.concatenate((x_arr_train_flattened,x_arr_train_flattened_shifted),axis=1)
            x_arr_fit_flattened = np.concatenate((x_arr_fit_flattened,x_arr_fit_flattened_shifted),axis=1)

    return x_arr_train_flattened, x_arr_fit_flattened, np.swapaxes((y_train),0,1)