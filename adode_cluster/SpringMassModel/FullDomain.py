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

def define_MSD_rec(**kwargs_sys):
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
        u_target = targets['u']
        u = ys['u']
        return  jnp.nanmean((x - x_target)**2 + (x_dot-x_dot_target)**2)#+ jnp.nanmean((u - u_target)**2)
            
    return eom, loss, gen_params, None, {}

def initial_dataset(length, tol, sampling_rate,kwargs_training):
    '''makes initial simulation from uvx data and returns dataset for training'''
    # Read the config file
    N,size,params = read_config(['D','a','k','epsilon_0','mu_1','mu_2','k_T','delta_t_e'
                                ,'k_T','k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
                                'n_0','l_0','spacing'],mode = 'chaos')

    keys =['D','a','k','epsilon_0','mu_1','mu_2','k_T','delta_t_e'
            ,'k_T','k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
            'n_0','l_0','spacing']

    params_true = dict(zip(keys,params))
    params_low = {key: value - value*tol for key, value in params_true.items()}
    params_high = {key: value + value*tol for key, value in params_true.items()}
    x_dot = np.gradient(x, axis=0) / params_true['delta_t_e']
    params_low['k_ij_pad'], params_high['k_ij_pad'] = params_true['k_ij_pad'],params_true['k_ij_pad']
    params_low['k_a_pad'], params_high['k_a_pad'] = params_true['k_a_pad'],params_true['k_a_pad']

    u0,v0,T0,x0,x_dot0,t_evals = u[0],v[0],T[0],x[0],x_dot[0],np.linspace(0, params_true['delta_t_e']*sampling_rate*length, length)
    kwargs_sys = {'size': 100,
                'spacing': 1,
                'N_sys': 1,
                'par_tol': 0,
                'params_true': params_true,
                'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
    kwargs_adoptODE = {'epochs': 10,'N_backups': 1,'lr': 1e-3,'lower_b': params_true,'upper_b': params_true,
                    'lower_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0},
                        'upper_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0}}

    # Setting up a dataset via simulation
    Simulation_MSD = simple_simulation(define_MSD,
                                t_evals,
                                kwargs_sys,
                                kwargs_adoptODE)

    targets = {'u':jnp.full((Simulation_MSD.ys['u'].shape),.5),'v': jnp.zeros_like(Simulation_MSD.ys['v']),'T':jnp.zeros_like(Simulation_MSD.ys['T']),
            'x':Simulation_MSD.ys['x'],'x_dot':Simulation_MSD.ys['x_dot']}
    kwargs_sys = {'size': 100,
                'N_sys': 1,
                'par_tol': tol,
                'params_true': params_true}
    kwargs_adoptODE = {'epochs': kwargs_training['epochs'],'N_backups': kwargs_training['N_backups'],'lr':kwargs_training['lr'],'lower_b': params_low,'upper_b': params_high,
                    'lr_y0':kwargs_training['lr_y0'],
                    'lower_b_y0':{'u':kwargs_training['u_low'],'v':kwargs_training['v_low'],'T':kwargs_training['T_low'],'x':x0,'x_dot':x_dot0},
                        'upper_b_y0':{'u':kwargs_training['u_high'],'v':kwargs_training['v_high'],'T':kwargs_training['T_high'],'x':x0,'x_dot':x_dot0}}
    dataset_MSD = dataset_adoptODE(define_MSD_rec,
                                    targets,
                                    t_evals, 
                                    kwargs_sys,
                                    kwargs_adoptODE,
                                    true_params=params_true)
    return dataset_MSD

def continue_dataset(dataset_MSD,Simulation_MSD, length, tol, sampling_rate,kwargs_training, keep_data = True ,keep_params = True):
    # Read the config file
    N,size,params = read_config(['D','a','k','epsilon_0','mu_1','mu_2','k_T','delta_t_e'
                                ,'k_T','k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
                                'n_0','l_0','spacing'],mode = 'chaos')

    keys =['D','a','k','epsilon_0','mu_1','mu_2','k_T','delta_t_e'
            ,'k_T','k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
            'n_0','l_0','spacing']

    params_true = dict(zip(keys,params))
    params_low = {key: value - value*tol for key, value in params_true.items()}
    params_high = {key: value + value*tol for key, value in params_true.items()}
    x_dot = np.gradient(x, axis=0) / params_true['delta_t_e']
    params_low['k_ij_pad'], params_high['k_ij_pad'] = params_true['k_ij_pad'],params_true['k_ij_pad']
    params_low['k_a_pad'], params_high['k_a_pad'] = params_true['k_a_pad'],params_true['k_a_pad']

    u0,v0,T0,x0,x_dot0,t_evals = Simulation_MSD.ys['u'][0,-1],Simulation_MSD.ys['v'][0,-1],Simulation_MSD.ys['T'][0,-1],Simulation_MSD.ys['x'][0,-1],Simulation_MSD.ys['x_dot'][0,-1],np.linspace(0, params_true['delta_t_e']*sampling_rate*length, length)
    kwargs_sys = {'size': 100,
                'spacing': 1,
                'N_sys': 1,
                'par_tol': 0,
                'params_true': params_true,
                'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
    kwargs_adoptODE = {'epochs': 10,'N_backups': 1,'lr': 1e-3}
    Simulation_MSD_2 = simple_simulation(define_MSD,
                                t_evals,
                                kwargs_sys,
                                kwargs_adoptODE)

    x_tar,x_dot_tar =  Simulation_MSD_2.ys['x'],Simulation_MSD_2.ys['x_dot']    
    if keep_data == True:
        u_tar,v_tar,T_tar = jnp.broadcast_to(dataset_MSD.ys_sol['u'][0,-1],(1,length,100,100)),jnp.broadcast_to(dataset_MSD.ys_sol['v'][0,-1],(1,length,100,100)),jnp.broadcast_to(dataset_MSD.ys_sol['T'][0,-1],(1,length,100,100))
        targets = {'u':u_tar,'v':v_tar,'T':T_tar,'x':x_tar,'x_dot':x_dot_tar}
    else:
        targets = {'u':jnp.full((Simulation_MSD.ys['u'].shape),.5),'v': jnp.zeros_like(Simulation_MSD.ys['v']),'T':jnp.zeros_like(Simulation_MSD.ys['T']),
                'x':Simulation_MSD_2.ys['x'],'x_dot':Simulation_MSD_2.ys['x_dot']}
        
    if keep_params == True:
        par = dataset_MSD.params_train
        par_tol = 0
    else:
        par = params_true
        par_tol = 0
    kwargs_sys = {'size': 100,
                'N_sys': 1,
                'par_tol': par_tol,
                'params_true': par}
    
    kwargs_adoptODE = {'epochs': kwargs_training['epochs'],'N_backups': kwargs_training['N_backups'],'lr':kwargs_training['lr'],'lower_b': params_low,'upper_b': params_high,
                    'lr_y0':kwargs_training['lr_y0'],
                    'lower_b_y0':{'u':kwargs_training['u_low'],'v':kwargs_training['v_low'],'T':kwargs_training['T_low'],'x':x_tar[0,0],'x_dot':x_dot_tar[0,0]},
                    'upper_b_y0':{'u':kwargs_training['u_high'],'v':kwargs_training['v_high'],'T':kwargs_training['T_high'],'x':x_tar[0,0],'x_dot':x_dot_tar[0,0]}}
    dataset_MSD_2 = dataset_adoptODE(define_MSD_rec,
                                    targets,
                                    t_evals, 
                                    kwargs_sys,
                                    kwargs_adoptODE,
                                    true_params=params_true)
    return dataset_MSD_2, Simulation_MSD_2

def save_run(run, dataset, length, sampling_rate):
    '''prints and saves the results of the training'''
    print('Parameter:   True Value:   Recovered Value:')
    for key in dataset.params.keys():
        print(key+(16-len(key))*' '+'{:.3f}         {:-3f}'.format(dataset.params[key], dataset.params_train[key]),'rel err',np.abs(dataset.params[key]-dataset.params_train[key])/dataset.params[key])
    
    with h5py.File('../data/SpringMassModel/EtaSweep/FullDomain_len'+str(length)+'lr'+str(sampling_rate)+'.h5', 'w') as f:
        group = f.create_group(run)  # Create a group instead of a dataset
        group.create_dataset('u_sol', data=dataset.ys_sol['u'])
        group.create_dataset('u',data=dataset.ys['u'])
        group.create_dataset('v_sol', data=dataset.ys_sol['v'])
        group.create_dataset('v',data=dataset.ys['v'])
        group.create_dataset('T_sol', data=dataset.ys_sol['T'])
        group.create_dataset('T',data=dataset.ys['T'])
        group.create_dataset('x_sol', data=dataset.ys_sol['x'])
        group.create_dataset('x',data=dataset.ys['x'])
        params = group.create_group("params_train")  # Create a subgroup
        for key, value in dataset.params_train.items():
            params.attrs[key] = value  # Store values as attributes

    print(f"Data for run '{run}' has been successfully added/updated.")

def save_new_run(run, dataset_MSD, length, sampling_rate):
    '''Opens an existing HDF5 file, appends new data, and saves the results'''
    
    print('Parameter:   True Value:   Recovered Value:')
    for key in dataset_MSD.params.keys():
        print(key + (16 - len(key)) * ' ' + '{:.3f}         {:-3f}'.format(dataset_MSD.params[key], dataset_MSD.params_train[key]),
              'rel err', np.abs(dataset_MSD.params[key] - dataset_MSD.params_train[key]) / dataset_MSD.params[key])
    
    file_path = '../data/SpringMassModel/EtaSweep/FullDomain_len' + str(length) + 'lr' + str(sampling_rate) + '.h5'
    
    with h5py.File(file_path, 'a') as f:  # Open the file in append mode ('a')
        # Check if the group already exists; if not, create it
        if run not in f:
            group = f.create_group(run)  # Create the group for the run
        else:
            group = f[run]  # Access the existing group

        # Create or update datasets
        group.create_dataset('u_sol', data=dataset_MSD.ys_sol['u'], overwrite=True)
        group.create_dataset('u', data=dataset_MSD.ys['u'], overwrite=True)
        group.create_dataset('v_sol', data=dataset_MSD.ys_sol['v'], overwrite=True)
        group.create_dataset('v', data=dataset_MSD.ys['v'], overwrite=True)
        group.create_dataset('T_sol', data=dataset_MSD.ys_sol['T'], overwrite=True)
        group.create_dataset('T', data=dataset_MSD.ys['T'], overwrite=True)
        group.create_dataset('x_sol', data=dataset_MSD.ys_sol['x'], overwrite=True)
        group.create_dataset('x', data=dataset_MSD.ys['x'], overwrite=True)
        
        # Create or update the params subgroup and store parameters as attributes
        if "params_train" not in group:
            params = group.create_group("params_train")
        else:
            params = group["params_train"]  # Access existing subgroup
        
        for key, value in dataset_MSD.params_train.items():
            params.attrs[key] = value  # Store values as attributes

    print(f"Data for run '{run}' has been successfully added/updated.")

kwargs_training = {'epochs': 200,'N_backups': 4,
                    'lr': 3e-2,'lr_y0':3e-2, 
                    'u_low':0,'u_high':95,
                    'v_low':0,'v_high':np.max(v),
                    'T_low':0,'T_high':np.max(T)}
#start initial training
length = 5
tol = 0.5
sampling_rate = 20
dataset_MSD = initial_dataset(length , tol, sampling_rate,kwargs_training)
params_final, losses, errors, params_history = train_adoptODE(dataset_MSD, print_interval=10, save_interval=10)

run = 'run0'
save_run(run, dataset_MSD, length, sampling_rate)
#continue training

for i in range(1,3):
    dataset_MSD,Simulation_MSD = continue_dataset(dataset_MSD,Simulation_MSD, length, tol, sampling_rate,kwargs_training,keep_data= False)
    params_final, losses, errors, params_history = train_adoptODE(dataset_MSD, print_interval=10, save_interval=10)
    run = 'run'+str(i)
    save_new_run(run, dataset_MSD, length, sampling_rate)