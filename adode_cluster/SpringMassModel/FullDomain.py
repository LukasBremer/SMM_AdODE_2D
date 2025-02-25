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
    v = f['v'][:100]
    u = f['u'][:100]
    T = f['T'][:100]
    x = f['x'][:100]
    f.close()

def define_MSD_sim(**kwargs_sys):

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

def define_MSD_fit(**kwargs_sys):

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
            
    return eom, loss, gen_params, None , {}

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
        # dA = np.abs(compute_dA(x, 1)[:,pad:-pad,pad:-pad])
        u0 = jnp.full((1,length,100,100),.5)#dA[0]/np.max(dA[0])*np.max(u)
        v0 = jnp.full((1,length,100,100),.5*np.max(v))#dA[0]/np.max(dA[0])*np.max(v)
        T0 = jnp.full((1,length,100,100),.5*np.max(T))#dA[0]/np.max(dA[0])*np.max(T)
        x0 = x[0]
        x_dot0 = x_dot[0]
    else:
        u0 = u_sol[-1]
        u0 = np.broadcast_to(u0, (length, 100, 100))
        v0 = v_sol[-1]
        v0 = np.broadcast_to(v0, (length, 100, 100))
        T0 = T_sol[-1]
        T0 = np.broadcast_to(T0, (length, 100, 100))
        x0 = x[-1]
        x_dot0 = x_dot[-1]
    return u0, v0, T0, x0, x_dot0, t_evals

"""
    Reads in necessary parameters from config.ini
"""
N,size,params = read_config(['D','a','k','epsilon_0','mu_1','mu_2','k_T','delta_t_e'
                             ,'k_T','k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
                             'n_0','l_0','spacing'],mode = 'chaos')
N = T.shape[0]
keys =['D','a','k','epsilon_0','mu_1','mu_2','k_T','delta_t_e'
        ,'k_T','k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
        'n_0','l_0','spacing']
pad = 10
tol = .9
params_true = dict(zip(keys,params))
params_low = {key: value - value*tol for key, value in params_true.items()}
params_high = {key: value + value*tol for key, value in params_true.items()}
params_high['k_ij_pad'],params_low['k_ij_pad'] = params_true['k_ij_pad']  ,params_true['k_ij_pad']
params_high['k_a_pad'],params_low['k_a_pad'] = params_true['k_a_pad']  ,params_true['k_a_pad']
x_dot = np.gradient(x, axis=0) / params_true['delta_t_e']
sol_list = []
sim_indx =0
length = 10
sampling_rate = 20
print(length,sampling_rate)
u0,v0,T0,x0,x_dot0,t_evals = get_ini_sim(u,v,T,x,x_dot,sim_indx,length = length,sampling_rate=sampling_rate)
kwargs_sys = {'size': 100,
            'spacing': 1,
            'N_sys': 1,
            'par_tol': 0,
            'params_true': params_true,
            'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
kwargs_adoptODE = {'epochs': 10,'N_backups': 1,'lr': 1e-3,
                'lower_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0},
                    'upper_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0}}
# Setting up a dataset via simulation
Simulation_MSD = simple_simulation(define_MSD_sim,
                            t_evals,
                            kwargs_sys,
                            kwargs_adoptODE)

u0,v0,T0,x0,x_dot0,t_evals = get_ini_fit(Simulation_MSD.ys['u'][0],Simulation_MSD.ys['v'][0],
                                        Simulation_MSD.ys['T'][0], Simulation_MSD.ys['x'][0], Simulation_MSD.ys['x_dot'][0],
                                        sim_indx,length=length,sampling_rate=sampling_rate)
targets = {'u':u0,'v':v0,'T':T0,
           'x':Simulation_MSD.ys['x'][0].reshape(1,length,2,size+2*pad+1,size+2*pad+1),'x_dot':Simulation_MSD.ys['x_dot'][0].reshape(1,length,2,size+2*pad+1,size+2*pad+1)}
kwargs_sys = {'size': 100,
            'N_sys': 1,
            'par_tol': tol,
            'params_true': params_true,
            'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
kwargs_adoptODE = {'epochs': 1000,'N_backups': 3,'lr': 1e-3,'lower_b': params_low,'upper_b': params_high,
                'lr_y0':2e-3,
                'lower_b_y0':{'u':0.,'v':0.,'T':0.,'x':x0,'x_dot':x_dot0},
                'upper_b_y0':{'u':np.max(u),'v':np.max(v),'T':np.max(T),'x':x0,'x_dot':x_dot0}}
dataset_MSD = dataset_adoptODE(define_MSD_fit,
                                targets,
                                t_evals, 
                                kwargs_sys,
                                kwargs_adoptODE,
                                true_params=params_true)

_ = train_adoptODE(dataset_MSD)
sol_list.append(dataset_MSD.params_train)
print('Found params: ', dataset_MSD.params_train)

for sim_indx in range(1,4):
    print(sim_indx,'time_steps=',sampling_rate*length*sim_indx)
    u0,v0,T0,x0,x_dot0,t_evals = get_ini_sim(dataset_MSD.ys['u'][0],dataset_MSD.ys['v'][0],dataset_MSD.ys['T'][0],dataset_MSD.ys['x'][0],dataset_MSD.ys['x_dot'][0],sim_indx,length=length,sampling_rate=sampling_rate)
    kwargs_sys = {'size': 100,
            'spacing': 1,
            'N_sys': 1,
            'par_tol': 0,
            'params_true': params_true,
            'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
    kwargs_adoptODE = {'epochs': 10,'N_backups': 1,'lr': 1e-3,
                    'lower_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0},
                        'upper_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0}}
    # Setting up a dataset via simulation
    Simulation_MSD = simple_simulation(define_MSD_sim,
                                t_evals,
                                kwargs_sys,
                                kwargs_adoptODE)

    x_sim = Simulation_MSD.ys['x'][0]
    x_dot_sim = Simulation_MSD.ys['x_dot'][0]
    T_sim = Simulation_MSD.ys['T'][0]
    u_sim = Simulation_MSD.ys['u'][0]
    v_sim = Simulation_MSD.ys['v'][0]

    u0,v0,T0,x0,x_dot0,t_evals = get_ini_fit(dataset_MSD.ys_sol['u'][0],dataset_MSD.ys_sol['v'][0], dataset_MSD.ys_sol['T'][0], dataset_MSD.ys['x'][0],
                                            dataset_MSD.ys['x_dot'][0], sim_indx,length=length,sampling_rate=sampling_rate)
    targets = {'u':u0.reshape(1,length,100,100),'v':v0.reshape(1,length,100,100),'T':T0.reshape(1,length,100,100),'x':x_sim.reshape(1,length,2,size+2*pad+1,size+2*pad+1),'x_dot':x_dot_sim.reshape(1,length,2,size+2*pad+1,size+2*pad+1)}
    kwargs_sys = {'size': 100,
            'N_sys': 1,
            'par_tol': tol,
            'params_true': params_true,
            'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
    kwargs_adoptODE = {'epochs': 1000,'N_backups': 3,'lr': 1e-3,'lower_b': params_low,'upper_b': params_high,
                    'lr_y0':2e-3,
                    'lower_b_y0':{'u':0.,'v':0.,'T':0.,'x':x0,'x_dot':x_dot0},
                    'upper_b_y0':{'u':np.max(u),'v':np.max(v),'T':np.max(T),'x':x0,'x_dot':x_dot0}}
    dataset_MSD = dataset_adoptODE(define_MSD_fit,
                                targets,
                                t_evals, 
                                kwargs_sys,
                                kwargs_adoptODE,
                                true_params=params_true)
    _ = train_adoptODE(dataset_MSD)
    print(dataset_MSD.params_train)
    
    sol_list.append(dataset_MSD.params_train)

# Print fitted parameter values
print('Parameter:   True Value:   Recovered Value:')
for key in dataset_MSD.params.keys():
    print(key+(16-len(key))*' '+'{:.3f}         {:-3f}'.format(dataset_MSD.params[key], dataset_MSD.params_train[key]),'rel err',np.abs(dataset_MSD.params[key]-dataset_MSD.params_train[key])/dataset_MSD.params[key])
with h5py.File('../data/SpringMassModel/EtaSweep/FullDomain_len'+str(length)+'lr'+str(sampling_rate)+'.h5', 'w') as f:
    f.create_dataset('u_sol', data=dataset_MSD.ys_sol['u'])
    f.create_dataset('u',data=dataset_MSD.ys['u'])
    f.create_dataset('v_sol', data=dataset_MSD.ys_sol['v'])
    f.create_dataset('v',data=dataset_MSD.ys['v'])
    f.create_dataset('T_sol', data=dataset_MSD.ys_sol['T'])
    f.create_dataset('T',data=dataset_MSD.ys['T'])
    f.create_dataset('x_sol', data=dataset_MSD.ys_sol['x'])
    f.create_dataset('x',data=dataset_MSD.ys['x'])
    group = f.create_group("params_train")  # Create a group instead of a dataset
    for key, value in dataset_MSD.params_train.items():
        group.attrs[key] = value  # Store values as attributes

    