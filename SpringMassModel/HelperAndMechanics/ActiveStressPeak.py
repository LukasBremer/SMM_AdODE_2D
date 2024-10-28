import jax.numpy as jnp
from jax import jit 
from jax.flatten_util import ravel_pytree

import numpy as np
import scipy
import matplotlib.pyplot as plt
import interpax

from scipy.signal import find_peaks
import progressbar

#model_la, t_model = make_longer_bound(l_a[0,:],t_evals,delta_t)
def stretch_peak(Delta_t_peak, t_evals, delta_t, T_standard, Delta_t_standard):
    '''stretches the standard peak to the desired length and evaluates it at the time points of the evaluation'''
    alpha = Delta_t_peak/Delta_t_standard

    t_standard = np.linspace(0,len(T_standard)*delta_t,len(T_standard))
    T_interp = interpax.CubicSpline(t_standard,T_standard)

    t_stretch = np.linspace(0,len(T_standard)*delta_t,len(t_evals))
    

    return T_interp(t_stretch) * alpha, t_stretch

def shift_peak(t_peak_start, t_evals_start, T_stand_strecht):
    '''shifts the stretched peak to the desired position'''
    i1 = t_peak_start- t_evals_start
    if i1 < 0:
        for i in range(int(abs(i1))):
            T_stand_strecht = np.append(T_stand_strecht,T_stand_strecht[-1])
            T_stand_strecht = np.delete(T_stand_strecht,0)
    else:
        for i in range(int(i1)):
            T_stand_strecht = np.append(T_stand_strecht[0],T_stand_strecht)
            T_stand_strecht = np.delete(T_stand_strecht,-1)

    return T_stand_strecht

def create_single_peak(T_standard, t_evals, t_evals_start, delta_t, t_peak_start,t_peak_stop):
    '''creates a peak based on the information of the peak position and the standard peak. The standard peak is stretched and shifted to the desired position'''
    #T_standard, Delta_t_standard = give_standard_peak('peak1')
    Delta_t_peak = t_peak_stop- t_peak_start
    Delta_t_standard = len(T_standard)

    T_strecht, t_strecht = stretch_peak(Delta_t_peak, t_evals, delta_t, T_standard, Delta_t_standard)
    T_shifted = shift_peak(t_peak_start, t_evals_start, T_strecht)

    return T_shifted

def give_standard_peak(peak):
    '''opens data and returnes the standard peak for the given key'''
    if peak == 'peak1':
        T_standard = np.array([0,0.5,1,0.5,0])
        Delta_t_standard = 1
        #read data from file
            
def create_T(T_standard, t_evals, t_evals_start, delta_t, t_peak_start,t_peak_stop):
    '''creates four peaks based on the information of the peak positions and the standard peak. The standard peak is stretched and shifted to the desired position'''
    T_approx = np.zeros((4,len(t_evals)))
    for i in range(4):
        T_temp  = create_single_peak(T_standard, t_evals, t_evals_start, delta_t, t_peak_start[i],t_peak_stop[i])
        T_approx[i,:] = T_temp

    return T_approx

def index_finder(maxima,minima,dA):
    '''finds the indices of the maxima and minima in the data'''
    max_indx = 0
    control = True
        
    while control:
        control = False
        for i in range(len(minima)):
            '''find first minima after first maxima'''
            if minima[i] > maxima[max_indx]:
                min_indx = i
                break

        if minima[min_indx] - maxima[max_indx] < 200:
            '''time distance between maxima and minima 
            choose next minima for process'''
            min_indx += 1
        if minima[min_indx] - maxima[max_indx] < 200:
            control = True
        
        if 800 < minima[min_indx] - maxima[max_indx]:
            '''time distance between maxima and minima 
            choose next minima for process'''
            max_indx += 1
        if 800 < minima[min_indx] - maxima[max_indx]:
            control = True

        # for i in range(len(maxima)):
        #     '''find if there is a maxima between the first maxima and minima
        #     redo process with next maxima if there is'''
        #     if maxima[max_indx] < maxima[i] < minima[min_indx]:
        #         max_indx += 1
        #         break
        #         control = True
            
        if abs(dA[maxima[max_indx]] - dA[minima[min_indx]]) < 0.005:
            '''hight difference between maxima and minima
            redo process with next maxima'''
            max_indx += 1
            control = True

        if minima[min_indx] < maxima[max_indx]:
            '''check if max is before min'''
            control = True

    return max_indx,min_indx