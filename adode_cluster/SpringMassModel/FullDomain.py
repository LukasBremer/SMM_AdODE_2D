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
    
    if kwargs_sys['eta_var'] == True: 
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
                dx_dotdt = 1/par['m'] *  (force_field_active_n_var(x,T,par,kwargs_sys) + force_field_passive_n_var(x,par,kwargs_sys) + force_field_struct(x,T,par) - x_dot * par['c_damp'])
                dxdt = x_dot

                return {'u':dudt, 'v':dvdt, 'T':dTdt, 'x':zero_out_edges(dxdt), 'x_dot':zero_out_edges(dx_dotdt)}
    else:
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
        iparams = {'testpar':0}
        params = {key:value + kwargs_sys['par_tol']*value*np.random.uniform(-1.0, 1.0) for key,value in kwargs_sys['params_true'].items()}
        iparams = {key:jnp.array([value + kwargs_sys['par_tol']*value*np.random.uniform(-1.0, 1.0) for _ in range(kwargs_sys['N_sys'])]) for key,value in kwargs_sys['params_true'].items()}
        # iparams = {key:np.full(kwargs_sys['N_sys'],[value + kwargs_sys['par_tol']*value*np.random.uniform(-1.0, 1.0)]) for key,value in kwargs_sys['params_true'].items()}
        return  params,{}, {}
    
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
    
    if kwargs_sys['eta_var'] == True: 
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
                dx_dotdt = 1/par['m'] *  (force_field_active_n_var_rec(x,T,par,kwargs_sys['n_gaussians']) + force_field_passive_n_var_rec(x,par,kwargs_sys['n_gaussians']) + force_field_struct(x,T,par) - x_dot * par['c_damp'])
                dxdt = x_dot

                return {'u':dudt, 'v':dvdt, 'T':dTdt, 'x':zero_out_edges(dxdt), 'x_dot':zero_out_edges(dx_dotdt)}
    else:
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

@jit
def multi_meas_constraint(ys, params, iparams, exparams, ys_target):
    param_list = ['k_ij', 'k_a', 'c_a', 'l_0', 'D', 'a', 
                  'k', 'mu_1', 'mu_2', 'k_T']
    
    # Vectorized sum of squared sums
    mmc = 1e-3 * sum(jnp.sum(iparams[key])**2 for key in param_list)
    
    return 0.*mmc

def initial_dataset(length, tol, sampling_rate,kwargs_training):
    '''makes initial simulation from uvx data and returns dataset for training'''
    # Read the config file
    N,size,params = read_config(['D','a','k','epsilon_0','mu_1','mu_2','delta_t_e'
                                ,'k_T','k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
                                'n_0','l_0','spacing'],mode = 'chaos')

    keys =['D','a','k','epsilon_0','mu_1','mu_2','delta_t_e'
            ,'k_T','k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
            'n_0','l_0','spacing']

    
    params_true = dict(zip(keys,params))
    params_low = {key: value - value*tol for key, value in params_true.items()}
    params_high = {key: value + value*tol for key, value in params_true.items()}
    x_dot = np.gradient(x, axis=0) / params_true['delta_t_e']
    params_low['k_ij_pad'], params_high['k_ij_pad'] = params_true['k_ij_pad'],params_true['k_ij_pad']
    params_low['k_a_pad'], params_high['k_a_pad'] = params_true['k_a_pad'],params_true['k_a_pad']
    params_low['delta_t_e'], params_high['delta_t_e'] = params_true['delta_t_e'],params_true['delta_t_e']
    
    u0,v0,T0,x0,x_dot0,t_evals = u[0],v[0],T[0],x[0],x_dot[0],np.linspace(0, params_true['delta_t_e']*sampling_rate*length, length)
    kwargs_sys = {'size': 100,
                'spacing': 1,
                'N_sys': 1,
                'par_tol': 0,
                'n_dist':kwargs_training['n_dist'],
                'eta_var':kwargs_training['eta_var'],
                'params_true': params_true,
                'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
    kwargs_adoptODE = {'epochs': 10,'N_backups': 1,'lr': 1e-3,'lower_b': params_true,'upper_b': params_true,
                    'lower_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0},
                    'upper_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0}}
    Simulation_MSD = simple_simulation(define_MSD,
                                    t_evals,
                                    kwargs_sys,
                                    kwargs_adoptODE)
    
    targets = {'u':jnp.full((Simulation_MSD.ys['u'].shape),.5),'v': jnp.zeros_like(Simulation_MSD.ys['v']),'T':jnp.zeros_like(Simulation_MSD.ys['T']),
            'x':Simulation_MSD.ys['x'],'x_dot':Simulation_MSD.ys['x_dot']}
    
    for _ in range(1,kwargs_training['N_sys']):
        u0,v0,T0,x0,x_dot0,t_evals = Simulation_MSD.ys['u'][0,-1],Simulation_MSD.ys['v'][0,-1],Simulation_MSD.ys['T'][0,-1],Simulation_MSD.ys['x'][0,-1],Simulation_MSD.ys['x_dot'][0,-1],np.linspace(0, params_true['delta_t_e']*sampling_rate*length, length)
        # u[0],v[0],T[0],x[0],x_dot[0],np.linspace(0, params_true['delta_t_e']*sampling_rate*length, length)
        
        kwargs_sys = {'size': 100,
                    'spacing': 1,
                    'N_sys': 1,
                    'par_tol': 0,
                    'n_dist':n_dist,
                    'eta_var':kwargs_training['eta_var'],
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
        targets = {'u':jnp.concatenate((targets['u'],targets['u'][0:1]),axis=0),'T':jnp.concatenate((targets['T'],targets['T'][0:1]),axis=0), 'v':jnp.concatenate((targets['v'],targets['v'][0:1]),axis=0),
                    'x':jnp.concatenate((targets['x'],Simulation_MSD.ys['x']),axis=0),'x_dot':jnp.concatenate((targets['x_dot'],Simulation_MSD.ys['x_dot']),axis=0)}
        
    if kwargs_training['eta_var'] == True:
        params_gaussian = {'Amp'+str(i)+str(j):0.1 for i in range(kwargs_training['n_gaussians']) for j in range(kwargs_training['n_gaussians'])}
        params_gaussian_low = {'Amp'+str(i)+str(j):-1 for i in range(kwargs_training['n_gaussians']) for j in range(kwargs_training['n_gaussians'])}
        params_gaussian_high = {'Amp'+str(i)+str(j):1 for i in range(kwargs_training['n_gaussians']) for j in range(kwargs_training['n_gaussians'])}
        params_true = params_true|params_gaussian
        params_low = params_low|params_gaussian_low
        params_high = params_high|params_gaussian_high



    kwargs_sys = {'size': 100,
                'N_sys': kwargs_training['N_sys'],
                'par_tol': tol,
                'eta_var':kwargs_training['eta_var'],
                'params_true': params_true,
                'n_gaussians':kwargs_training['n_gaussians']}
    kwargs_adoptODE = {'epochs': kwargs_training['epochs'],'N_backups': kwargs_training['N_backups'],'lr':kwargs_training['lr'],
                    'lower_b': params_low,'upper_b': params_high,
                    'lr_y0':kwargs_training['lr_y0'],
                    # 'lr_ip': kwargs_training['lr_ip'],
                    'lower_b_y0':{'u':kwargs_training['u_low'],'v':kwargs_training['v_low'],'T':kwargs_training['T_low'],'x':x0,'x_dot':x_dot0},
                    'upper_b_y0':{'u':kwargs_training['u_high'],'v':kwargs_training['v_high'],'T':kwargs_training['T_high'],'x':x0,'x_dot':x_dot0}}
    if 'multi_measurement_constraint' in kwargs_training:
        kwargs_adoptODE = kwargs_adoptODE|{'multi_measurement_constraint':kwargs_training['multi_measurement_constraint']}
    dataset_MSD = dataset_adoptODE(define_MSD_rec,
                                    targets,
                                    t_evals, 
                                    kwargs_sys,
                                    kwargs_adoptODE,
                                    true_params=params_true)#,
                                    #true_iparams=params_true)
    return dataset_MSD,Simulation_MSD

def continue_dataset(dataset_MSD,Simulation_MSD, length, tol, sampling_rate, kwargs_training, tol_AP, keep_data = True ,keep_params = True):
    # Read the config file
    N,size,params = read_config(['D','a','k','epsilon_0','mu_1','mu_2','delta_t_e'
                                ,'k_T','k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
                                'n_0','l_0','spacing'],mode = 'chaos')

    keys =['D','a','k','epsilon_0','mu_1','mu_2','delta_t_e'
            ,'k_T','k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
            'n_0','l_0','spacing']
    keys_electric = ['D','a','k','epsilon_0','mu_1','mu_2']
    params_true = dict(zip(keys,params))
    params_low = {key: value - value*tol for key, value in params_true.items()}
    params_high = {key: value + value*tol for key, value in params_true.items()}
    x_dot = np.gradient(x, axis=0) / params_true['delta_t_e']
    params_low['k_ij_pad'], params_high['k_ij_pad'] = params_true['k_ij_pad'],params_true['k_ij_pad']
    params_low['k_a_pad'], params_high['k_a_pad'] = params_true['k_a_pad'],params_true['k_a_pad']
    params_low['delta_t_e'], params_high['delta_t_e'] = params_true['delta_t_e'],params_true['delta_t_e']
    for key in keys_electric:
        params_low[key],params_high[key] = params_true[key]- params_true[key] * tol_AP, params_true[key] + params_true[key] * tol_AP

    u0,v0,T0,x0,x_dot0,t_evals = Simulation_MSD.ys['u'][0,-1],Simulation_MSD.ys['v'][0,-1],Simulation_MSD.ys['T'][0,-1],Simulation_MSD.ys['x'][0,-1],Simulation_MSD.ys['x_dot'][0,-1],np.linspace(0, params_true['delta_t_e']*sampling_rate*length, length)
    kwargs_sys = {'size': 100,
                'spacing': 1,
                'N_sys': 1,
                'par_tol': 0,
                'n_dist':kwargs_training['n_dist'],
                'eta_var':kwargs_training['eta_var'],
                'params_true': params_true,
                'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
    kwargs_adoptODE = {'epochs': 10,'N_backups': 1,'lr': 1e-3,'lower_b': params_true,'upper_b': params_true,
                    'lower_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0},
                    'upper_b_y0':{'u':u0,'v':v0,'T':T0,'x':x0,'x_dot':x_dot0}}
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
    
    for run_ipar in range(1,kwargs_training['N_sys']):
        u0,v0,T0,x0,x_dot0,t_evals = Simulation_MSD_2.ys['u'][0,-1],Simulation_MSD_2.ys['v'][0,-1],Simulation_MSD_2.ys['T'][0,-1],Simulation_MSD_2.ys['x'][0,-1],Simulation_MSD_2.ys['x_dot'][0,-1],np.linspace(0, params_true['delta_t_e']*sampling_rate*length, length)
        kwargs_sys = {'size': 100,
                    'spacing': 1,
                    'N_sys': 1,
                    'par_tol': 0,
                    'params_true': params_true,
                    'n_dist':kwargs_training['n_dist'],
                    'eta_var':kwargs_training['eta_var'],
                    'u0': u0,'v0': v0,'T0': T0,'x0': x0,'x_dot0': x_dot0}
        kwargs_adoptODE = {'epochs': 10,'N_backups': 1,'lr': 1e-3}
        Simulation_MSD_2 = simple_simulation(define_MSD,
                                    t_evals,
                                    kwargs_sys,
                                    kwargs_adoptODE)

        x_tar,x_dot_tar =  Simulation_MSD_2.ys['x'],Simulation_MSD_2.ys['x_dot']    
        if keep_data == True:
            u_tar,v_tar,T_tar = jnp.broadcast_to(dataset_MSD.ys_sol['u'][run_ipar,-1],(1,length,100,100)),jnp.broadcast_to(dataset_MSD.ys_sol['v'][run_ipar,-1],(1,length,100,100)),jnp.broadcast_to(dataset_MSD.ys_sol['T'][run_ipar,-1],(1,length,100,100))
            targets = {'u':jnp.concatenate((targets['u'],u_tar),axis=0),'T':jnp.concatenate((targets['T'],T_tar),axis=0), 'v':jnp.concatenate((targets['v'],v_tar),axis=0),
                        'x':jnp.concatenate((targets['x'],x_tar),axis=0),'x_dot':jnp.concatenate((targets['x_dot'],x_dot_tar),axis=0)}
        else:
            targets = {'u':jnp.concatenate((targets['u'],jnp.full((Simulation_MSD.ys['u'].shape),.5)),axis=0),'v': jnp.concatenate((targets['v'],jnp.zeros_like(Simulation_MSD.ys['v'])),axis=0),'T':jnp.concatenate((targets['T'],jnp.zeros_like(Simulation_MSD.ys['T'])),axis=0),
                    'x':jnp.concatenate((targets['x'],x_tar),axis=0),'x_dot':jnp.concatenate((targets['x_dot'],x_dot_tar),axis=0)}
    
    if keep_params == True:
        params_true = dataset_MSD.params
        params_train = dataset_MSD.params_train
        par_tol = 0.2
        if kwargs_training['eta_var'] == True:
            params_gaussian_low = {'Amp'+str(i)+str(j):-1 for i in range(kwargs_training['n_gaussians']) for j in range(kwargs_training['n_gaussians'])}
            params_gaussian_high = {'Amp'+str(i)+str(j):1 for i in range(kwargs_training['n_gaussians']) for j in range(kwargs_training['n_gaussians'])}
            params_low = params_low|params_gaussian_low
            params_high = params_high|params_gaussian_high
    else:
        params_train = params_true
        par_tol = tol
        if kwargs_training['eta_var'] == True:
            params_gaussian = {'Amp'+str(i)+str(j):0.1 for i in range(kwargs_training['n_gaussians']) for j in range(kwargs_training['n_gaussians'])}
            params_gaussian_low = {'Amp'+str(i)+str(j):-1 for i in range(kwargs_training['n_gaussians']) for j in range(kwargs_training['n_gaussians'])}
            params_gaussian_high = {'Amp'+str(i)+str(j):1 for i in range(kwargs_training['n_gaussians']) for j in range(kwargs_training['n_gaussians'])}
            params_true = params_true|params_gaussian
            params_low = params_low|params_gaussian_low
            params_high = params_high|params_gaussian_high
        

    kwargs_sys = {'size': 100,
                'N_sys': kwargs_training['N_sys'],
                'par_tol': par_tol,
                'eta_var':kwargs_training['eta_var'],
                'params_true': params_train,
                'n_gaussians':kwargs_training['n_gaussians']}
    
    kwargs_adoptODE = {'epochs': kwargs_training['epochs'],'N_backups': kwargs_training['N_backups'],'lr':kwargs_training['lr'],'lower_b': params_low,'upper_b': params_high,
                    'lr_y0':kwargs_training['lr_y0'],
                    # 'lr_ip': kwargs_training['lr_ip'],
                    'lower_b_y0':{'u':kwargs_training['u_low'],'v':kwargs_training['v_low'],'T':kwargs_training['T_low'],'x':x_tar[0,0],'x_dot':x_dot_tar[0,0]},
                    'upper_b_y0':{'u':kwargs_training['u_high'],'v':kwargs_training['v_high'],'T':kwargs_training['T_high'],'x':x_tar[0,0],'x_dot':x_dot_tar[0,0]}}
    # print(kwargs_adoptODE)
    # if 'multi_measurement_constraint' in kwargs_training:
    #     kwargs_adoptODE = kwargs_adoptODE|{'multi_measurement_constraint':kwargs_training['multi_measurement_constraint']}
    dataset_MSD_2 = dataset_adoptODE(define_MSD_rec,
                                    targets,
                                    t_evals, 
                                    kwargs_sys,
                                    kwargs_adoptODE,
                                    true_params=params_true)#,
                                    # true_iparams=params_true)
    return dataset_MSD_2, Simulation_MSD_2

def save_run(run, Simulation, dataset, length, sampling_rate,tol,keep_data,keep_params,eta_var):
    '''prints and saves the results of the training'''
    print('Parameter:   True Value:   Recovered Value:')
    for key in dataset.params.keys():
        print(key+(16-len(key))*' '+'{:.3f}         {:-3f}'.format(dataset.params[key], dataset.params_train[key]),'rel err',np.abs(dataset.params[key]-dataset.params_train[key])/dataset.params[key])
    
    
    file_path = (
        f"../data/SpringMassModel/EtaSweep/"
        f"FullDomain_len{length}_lr{sampling_rate}_tol0{str(tol).split('.')[1]}_"
        f"keepdata{keep_data}_keepparams{keep_params}_etavar{eta_var}.h5"
    )

    with h5py.File(file_path, 'w') as f:
        group = f.create_group(run)  # Create a group instead of a dataset
        group.create_dataset('u_sol', data=dataset.ys_sol['u'])
        group.create_dataset('u',data=Simulation.ys['u'])
        group.create_dataset('v_sol', data=dataset.ys_sol['v'])
        group.create_dataset('v',data=Simulation.ys['v'])
        group.create_dataset('T_sol', data=dataset.ys_sol['T'])
        group.create_dataset('T',data=Simulation.ys['T'])
        group.create_dataset('x_sol', data=dataset.ys_sol['x'])
        group.create_dataset('x',data=Simulation.ys['x'])
        params = group.create_group("params_train")  # Create a subgroup
        for key, value in dataset.params_train.items():
            params.attrs[key] = value  # Store values as attributes
    f.close()
    print(f"Data for run '{run}' has been successfully added/updated.")
    
def save_new_run(run, Simulation, dataset, length, sampling_rate,tol,keep_data,keep_params,eta_var):
    '''Opens an existing HDF5 file, appends new data, and saves the results'''
    
    print('Parameter:   True Value:   Recovered Value:')
    for key in dataset.params.keys():
        print(key + (16 - len(key)) * ' ' + '{:.3f}         {:-3f}'.format(dataset.params[key], dataset.params_train[key]),
              'rel err', np.abs(dataset.params[key] - dataset.params_train[key]) / dataset.params[key])
    
    file_path = (
        f"../data/SpringMassModel/EtaSweep/"
        f"FullDomain_len{length}_lr{sampling_rate}_tol0{str(tol).split('.')[1]}_"
        f"keepdata{keep_data}_keepparams{keep_params}_etavar{eta_var}.h5"
    )
    with h5py.File(file_path, 'a') as f:  # Open the file in append mode ('a')
        # Check if the group already exists; if not, create it
        if run not in f:
            group = f.create_group(run)  # Create the group for the run
        else:
            group = f[run]  # Access the existing group

        # Create or update datasets
        group.create_dataset('u_sol', data=dataset.ys_sol['u'])
        group.create_dataset('u', data = Simulation.ys['u'])
        group.create_dataset('v_sol', data=dataset.ys_sol['v'])
        group.create_dataset('v', data = Simulation.ys['v'])
        group.create_dataset('T_sol', data=dataset.ys_sol['T'])
        group.create_dataset('T', data = Simulation.ys['T'])
        group.create_dataset('x_sol', data=dataset.ys_sol['x'])
        group.create_dataset('x', data = Simulation.ys['x'])
        
        # Create or update the params subgroup and store parameters as attributes
        if "params_train" not in group:
            params = group.create_group("params_train")
        else:
            params = group["params_train"]  # Access existing subgroup
        
        for key, value in dataset.params_train.items():
            params.attrs[key] = value  # Store values as attributes
    
    f.close()
    print(f"Data for run '{run}' has been successfully added/updated.")


n_dist = np.load('../data/SpringMassModel/FiberOrientation/fiber_orientation.npy')
kwargs_training = {'epochs': 300,'N_backups': 5,
                    'lr': 2e-3,'lr_y0':2e-2, 
                    # 'lr_ip':1e-3,
                    'u_low':0,'u_high':99,
                    'v_low':0,'v_high':np.max(v),
                    'T_low':0,'T_high':np.max(T),
                    'eta_var':True,
                    'n_dist':n_dist,
                    'n_gaussians':3,
                    'N_sys':1}
#start initial Dataset and train
length = 15
tol = 0.99
sampling_rate = 15
dataset_MSD,Simulation_MSD = initial_dataset(length , tol, sampling_rate,kwargs_training)
print('start training')
params_final, losses, errors, params_history = train_adoptODE(dataset_MSD, print_interval=10, save_interval=10)

keep_data = True
keep_params = True
run = 'run0'
save_run(run, Simulation_MSD ,dataset_MSD, length, sampling_rate,tol,keep_data,keep_params,kwargs_training['eta_var'])

#continue training
for i in range(1,5):
    # overwrite old simulation and dataset with new one
    print(length,tol,sampling_rate,i)
    tol_AP = 1 - .2*i
    dataset_MSD,Simulation_MSD = continue_dataset(dataset_MSD,Simulation_MSD, length, tol, sampling_rate,kwargs_training,tol_AP,keep_data= keep_data,keep_params=keep_params)
    params_final, losses, errors, params_history = train_adoptODE(dataset_MSD, print_interval=10, save_interval=10)
    run = 'run'+str(i)
    save_new_run(run, Simulation_MSD, dataset_MSD, length, sampling_rate,tol,keep_data,keep_params,kwargs_training['eta_var'])

