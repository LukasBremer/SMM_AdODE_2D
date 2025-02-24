#import jax and other libraries for computation
import jax.numpy as jnp
from jax import jit
from jax.scipy.signal import convolve2d
from jax.flatten_util import ravel_pytree
from jax.experimental.ode import odeint
from jax import tree_util
import jax.random as random
import numpy as np
# import AdoptODE
from adoptODE import train_adoptODE, simple_simulation, dataset_adoptODE
#import the MSD mechanics
from HelperAndMechanics import *
import h5py
# Load from HDF5
with h5py.File('../data/SpringMassModel/MechanicalData/data_eta_05_uvx.h5', 'r') as f:
    v = f['v'][:]
    u = f['u'][:]
    T = f['T'][:]
    x = f['x'][:]
    f.close()

N = T.shape[0]

def define_MSD(**kwargs_sys):

    N_sys = kwargs_sys['N_sys']

    def gen_params():
        return {key:value + kwargs_sys['par_tol']*value*np.random.uniform(-1.0, 1.0) for key,value in kwargs_sys['params_true'].items()}, {}, {}
    
    def gen_y0():
        return {'u':kwargs_sys['u0'],'v':kwargs_sys['v0'],'T':kwargs_sys['T0'],'x':kwargs_sys['x0'],'x_dot':kwargs_sys['x_dot0']}
    @jit
    def kernel(spacing):
        kernel = np.array([[1, 4, 1], [4, -20.0, 4], [1, 4, 1]]) / (spacing* spacing * 6)
        return kernel
    @jit
    def laplace(f,params):  #laplace of scalar
        f_ext = jnp.concatenate((f[0:1], f, f[-1:]), axis=0)
        f_ext = jnp.concatenate((f_ext[:, 0:1], f_ext, f_ext[:, -1:]), axis=1)
        return convolve2d(f_ext, kernel(params['spacing']), mode='valid')
    @jit
    def epsilon(u,v,rp):
        return rp['epsilon_0']+rp['mu_1']*v/(u+rp['mu_2'])
    @jit
    def epsilon_T(u):
        return 1 - 0.9*jnp.exp(-jnp.exp(-30*(jnp.abs(u) - 0.1)))
    @jit
    def eom(y, t, params, iparams, exparams):
            
            par=params
            u=y['u']
            v=y['v']
            T=y['T']
            x=y['x']
            x_dot=y['x_dot']

            dudt = par['D']*laplace(u,par)-(par['k'])*u*(u-par['a'])*(u-1) - u*v
            dvdt = epsilon(u,v,par)*(-v-(par['k'])*u*(u-par['a']-1))
            dTdt = epsilon_T(u)*(par['k_T']*jnp.abs(u)-T)
            dx_dotdt = 1/par['m'] *  (force_field_active(x,T,par) + force_field_passive(x,par) + force_field_struct(x,T,par) - x_dot * par['c_damp'])
            dxdt = x_dot

            return {'u':dudt, 'v':dvdt, 'T':dTdt, 'x':zero_out_edges(dxdt), 'x_dot':zero_out_edges(dx_dotdt)}
    @jit
    def loss(ys, params, iparams, exparams, targets):
        # u = ys['u']
        # u_target = targets['u']
        pad = 10
        x = ys['x'][:,:,pad:-pad,pad:-pad]
        x_target = targets['x'][:,:,pad:-pad,pad:-pad]
        x_dot = ys['x_dot'][:,:,pad:-pad,pad:-pad]
        x_dot_target = targets['x_dot'][:,:,pad:-pad,pad:-pad]
        
        return  jnp.nanmean((x - x_target)**2 + (x_dot-x_dot_target)**2)#jnp.nanmean((u - u_target)**2) +
            
    return eom, loss, gen_params, gen_y0, {}

def get_ini_sim(u,v,T,x,x_dot,sim_indx,size = 100,pad = 10,delta_t_e = 0.08,sampling_rate = 10,length = 30):
    '''gets the initial conditions for the simulation
    if sim_indx=0 then the initial conditions are taken from raw data
    if sim_indx=1 then the initial conditions are taken from the last state of the recent simulation'''
    t_evals = np.linspace(0, delta_t_e*sampling_rate*length, length)
    if sim_indx == 0:
        #define grid and initial conditions
        u_fit = u[:length*sampling_rate][::sampling_rate,:,:]
        T_fit = T[:length*sampling_rate][::sampling_rate,:,:]
        v_fit = v[:length*sampling_rate][::sampling_rate,:,:]
        x_fit = x[:length*sampling_rate][::sampling_rate,:,:,:]
        x_dot = np.gradient(x,axis = 0)/delta_t_e
        x_dot_fit = x_dot[:length*sampling_rate][::sampling_rate,:,:,:]
        #select input state
        u0 = u_fit[0]
        v0 = v_fit[0]
        T0 = T_fit[0]
        x0 = x_fit[0]
        x_dot0 = x_dot_fit[0]
    else:
        u0 = u[-1]
        v0 = v[-1]
        T0 = T[-1]
        x0 = x[-1]
        x_dot0 = x_dot[-1]
        
    return u0, v0, T0, x0, x_dot0, t_evals

def get_ini_fit(u_sol,v_sol,T_sol,x,x_dot,sim_indx,size = 100,pad = 10,delta_t_e = 0.08,sampling_rate = 10,length=30):
    '''gets the initial conditions for the simulation
    if sim_indx=0 then u,v,T are initialized as resting stante (0)
    if sim_indx=1 then the u,v,T are taken from the last state of the recent fit'''
    t_evals = np.linspace(0, delta_t_e*sampling_rate*length, length)
    if sim_indx == 0:
        #select input state
        u0 = jnp.zeros((size,size))
        v0 = jnp.zeros((size,size))
        T0 = jnp.zeros((size,size))
        x0 = x[0]
        x_dot0 = x_dot[0]
    else:
        u0 = u_sol[-1]
        v0 = v_sol[-1]
        T0 = T_sol[-1]
        x0 = x[-1]
        x_dot0 = x_dot[-1]
    return u0, v0, T0, x0, x_dot0, t_evals

"""
    Reads in necessary parameters from config.ini
"""
N,size,params = read_config(['D','a','k','epsilon_0','mu_1','mu_2','k_T','delta_t_e'
                             ,'k_T','k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
                             'n_0','l_0','spacing'],mode = 'chaos')

keys =['D','a','k','epsilon_0','mu_1','mu_2','k_T','delta_t_e'
        ,'k_T','k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
        'n_0','l_0','spacing']
pad = 10
tol = 0
params_true = dict(zip(keys,params))
params_low = {key: value - value*tol for key, value in params_true.items()}
params_high = {key: value + value*tol for key, value in params_true.items()}
x_dot = np.gradient(x, axis=1) / params_true['delta_t_e']
sol_list = []
sim_indx =0
u0,v0,T0,x0,x_dot0,t_evals = get_ini_sim(u,v,T,x,x_dot,sim_indx)

kwargs_sys = {'size': 100,
            'spacing': 1,
            'N_sys': 1,
            'par_tol': 0,
            'params_true': params_true,
            'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
kwargs_adoptODE = {'epochs': 10,'N_backups': 1,'lr': 1e-3,'lower_b': params_low,'upper_b': params_high,
                'lower_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0},
                    'upper_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0}}
# Setting up a dataset via simulation
Simulation_MSD = simple_simulation(define_MSD,
                            t_evals,
                            kwargs_sys,
                            kwargs_adoptODE)

x_sim = Simulation_MSD.ys['x'][0]
x_dot_sim = Simulation_MSD.ys['x_dot'][0]
T_sim = Simulation_MSD.ys['T'][0]
u_sim = Simulation_MSD.ys['u'][0]
v_sim = Simulation_MSD.ys['v'][0]

u0,v0,T0,x0,x_dot0,t_evals = get_ini_fit(u_sim,v_sim, T_sim, x_sim, x_dot_sim, sim_indx)

tol = 0.5
params_low = {key: value - value*tol for key, value in params_true.items()}
params_high = {key: value + value*tol for key, value in params_true.items()}
params_high['k_ij_pad'],params_low['k_ij_pad'] = params_true['k_ij_pad']  ,params_true['k_ij_pad']
params_high['k_a_pad'],params_low['k_a_pad'] = params_true['k_a_pad']  ,params_true['k_a_pad']

length = 30
targets = {'u':u_sim.reshape(1,length,100,100),'v':v_sim.reshape(1,length,100,100),'T':T_sim.reshape(1,length,100,100),'x':x_sim.reshape(1,length,2,size+2*pad+1,size+2*pad+1),'x_dot':x_dot_sim.reshape(1,length,2,size+2*pad+1,size+2*pad+1)}
kwargs_sys = {'size': 100,
            'spacing': 1,
            'N_sys': 1,
            'par_tol': 0.5,
            'params_true': params_true,
            'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
kwargs_adoptODE = {'epochs': 200,'N_backups': 1,'lr': 5e-2,'lower_b': params_low,'upper_b': params_high,
                'lower_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0},
                    'upper_b_y0':{'u':jnp.full_like(u0,1),'v':jnp.full_like(v0,2.5),'T':jnp.full_like(T0,3),'x':x0,'x_dot':x_dot0}}

dataset_MSD = dataset_adoptODE(define_MSD,
                                targets,
                                t_evals, 
                                kwargs_sys,
                                kwargs_adoptODE,
                                true_params=params_true)
_ = train_adoptODE(dataset_MSD)
sol_list.append(dataset_MSD.params_train)
print('Found params: ', dataset_MSD.params_train)

# for sim_indx in range(1,3):
#     u0,v0,T0,x0,x_dot0,t_evals = get_ini_sim(dataset_MSD.ys['u'][0],dataset_MSD.ys['v'][0],dataset_MSD.ys['T'][0],dataset_MSD.ys['x'][0],dataset_MSD.ys['x_dot'][0],sim_indx)
#     kwargs_sys = {'size': 100,
#             'spacing': 1,
#             'N_sys': 1,
#             'par_tol': 0,
#             'params_true': params_true,
#             'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
#     kwargs_adoptODE = {'epochs': 10,'N_backups': 1,'lr': 1e-3,'lower_b': params_low,'upper_b': params_high,
#                     'lower_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0},
#                         'upper_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0}}
#     # Setting up a dataset via simulation
#     Simulation_MSD = simple_simulation(define_MSD,
#                                 t_evals,
#                                 kwargs_sys,
#                                 kwargs_adoptODE)

#     x_sim = Simulation_MSD.ys['x'][0]
#     x_dot_sim = Simulation_MSD.ys['x_dot'][0]
#     T_sim = Simulation_MSD.ys['T'][0]
#     u_sim = Simulation_MSD.ys['u'][0]
#     v_sim = Simulation_MSD.ys['v'][0]

#     u0,v0,T0,x0,x_dot0,t_evals = get_ini_fit(dataset_MSD.ys_sol['u'][0],dataset_MSD.ys_sol['v'][0], dataset_MSD.ys_sol['T'][0], dataset_MSD.ys['x'], dataset_MSD.ys['x_dot'], sim_indx)

#     tol = 0.5
#     params_low = {key: value - value*tol for key, value in params_true.items()}
#     params_high = {key: value + value*tol for key, value in params_true.items()}
#     params_high['k_ij_pad'],params_low['k_ij_pad'] = params_true['k_ij_pad']  ,params_true['k_ij_pad']
#     params_high['k_a_pad'],params_low['k_a_pad'] = params_true['k_a_pad']  ,params_true['k_a_pad']

#     length = 30
#     targets = {'u':u_sim.reshape(1,length,100,100),'v':v_sim.reshape(1,length,100,100),'T':T_sim.reshape(1,length,100,100),'x':x_sim.reshape(1,length,2,size+2*pad+1,size+2*pad+1),'x_dot':x_dot_sim.reshape(1,length,2,size+2*pad+1,size+2*pad+1)}
#     kwargs_sys = {'size': 100,
#                 'spacing': 1,
#                 'N_sys': 1,
#                 'par_tol': 0.5,
#                 'params_true': params_true,
#                 'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
#     kwargs_adoptODE = {'epochs': 200,'N_backups': 1,'lr': 5e-2,'lower_b': params_low,'upper_b': params_high,
#                     'lower_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0},
#                         'upper_b_y0':{'u':jnp.full_like(u0,1),'v':jnp.full_like(v0,2.5),'T':jnp.full_like(T0,3),'x':x0,'x_dot':x_dot0}}

#     dataset_MSD = dataset_adoptODE(define_MSD,
#                                     targets,
#                                     t_evals, 
#                                     kwargs_sys,
#                                     kwargs_adoptODE,
#                                     true_params=params_true)
#     _ = train_adoptODE(dataset_MSD)
#     sol_list.append(dataset_MSD.params_train)
#     print('Found params: ', dataset_MSD.params_train)

np.save('MSD_fit.npy',sol_list)