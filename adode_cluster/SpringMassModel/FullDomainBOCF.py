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

file_path = '../data/SpringMassModel/MechanicalData/MSD_BOCF.h5'

with h5py.File(file_path, 'r') as f:
    data = f['datasetBOCF_MSD_AP_fit']  # Access the group corresponding to the given 'run'
    print(data.keys())
    
    # Accessing the datasets
    params_AP = {}
    u_AP = np.array(data['u_AP'])
    u_BOCF = np.array(data['u_BOCF'])
    v_AP = np.array(data['v_AP'])
    v_BOCF = np.array(data['v_BOCF'])
    T_AP = np.array(data['T_AP'])
    T_BOCF = np.array(data['T_BOCF'])
    x_BOCF = np.array(data['x_BOCF'])
    x_dot_BOCF = np.array(data['x_dot_BOCF'])
    
    params_group = data['params_train_AP']
    for key in params_group.attrs:
        params_AP[key] = params_group.attrs[key]  # Store as dictionary

def define_MSD_rec(**kwargs_sys):
    N_sys = kwargs_sys['N_sys']

    def gen_params():
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
    
    @jit
    def eom(y, t, params, iparams, exparams):
            par=params
            u=y['u']
            v=y['v']
            T=y['T']
            x=y['x']
            x_dot=y['x_dot']
            dudt = par['D']*laplace(u,par)-(10**par['k'])*u*(u-par['a'])*(u-1) - u*v
            dvdt = epsilon(u,v,par)*(-v-(10**par['k'])*u*(u-par['a']-1))
            dTdt = epsilon_T(u)*(par['k_T']*jnp.abs(u)-T)
            dx_dotdt = 1/par['m'] *  (force_field_active(x,T,par) + force_field_passive(x,par) + force_field_struct(x,T,par) - x_dot * par['c_damp'])
            dxdt = x_dot

            return {'u':dudt, 'v':dvdt, 'T':dTdt, 'x':zero_out_edges(dxdt), 'x_dot':zero_out_edges(dx_dotdt)}
    @jit
    def normalize(tensor):
        return tensor # / (jnp.std(tensor) + 1e-6)  # Avoid division by zero
    @jit
    def loss(ys, params, iparams, exparams, targets):
        pad = 10
        domain_size = kwargs_sys['size'] + 1 
       
        x = ys['x'][:,:,pad:-pad,pad:-pad]
        x_target = targets['x'][:,:,pad:-pad,pad:-pad]
        x_dot = ys['x_dot'][:,:,pad:-pad,pad:-pad]
        x_dot_target = targets['x_dot'][:,:,pad:-pad,pad:-pad]
        
        u_target = targets['u']
        u = ys['u']
        T_target = targets['T']
        T = ys['T']

        dA = compute_dA(x,1)
        dA_target = compute_dA(x_target,1)
        
        laplacian_dA = laplacian_filter(dA)
        laplacian_dA_target = laplacian_filter(dA_target)
        
        x_vals = jnp.linspace(0, domain_size-1,domain_size)
        z_vals = jnp.linspace(0, domain_size-1,domain_size)
        x_grid, z_grid = jnp.meshgrid(x_vals, z_vals)
        xy_grid = jnp.array([x_grid, z_grid])
        xy_grid = jnp.reshape(xy_grid,(1,2,domain_size,domain_size))
        norm_x = jnp.nanmean(jnp.linalg.norm(x_target-xy_grid,axis=1))
        norm_x_dot = jnp.nanmean(jnp.linalg.norm(x_dot_target,axis=1))
        norm_dA = jnp.nanmean(dA_target)
        norm_dA_laplacian = jnp.max(laplacian_dA_target)
        
        u_diff = jnp.nanmean(jnp.max(jnp.max(u,axis =-1),axis = -1)[0] - jnp.min(jnp.min(u,axis =-1),axis = -1)[0]) + .01
        
        loss_x = jnp.nanmean(normalize((x - x_target)/norm_x)**2)
        loss_x_dot = jnp.nanmean(normalize((x_dot - x_dot_target)/norm_x_dot)**2)
        loss_dA = jnp.nanmean(normalize((dA - dA_target)/norm_dA)**2)
        loss_laplacian = jnp.nanmean(normalize((laplacian_dA - laplacian_dA_target)/norm_dA_laplacian)**2)


        loss = loss_x + loss_x_dot + loss_dA + loss_laplacian
        return loss #- 5e-3*1*(jnp.log(u_diff)) #+ jnp.nanmean((dA - dA_target)**2) 
            
    return eom, loss, gen_params, None, {}

def initial_dataset(kwargs_training):
    '''returns dataset for training'''
    t_evals = jnp.linspace(0, 50, 100)/12.9
    u_AP_init = u_AP
    targets = {}
    targets['u'] = jnp.full((u_AP_init).shape,.4)
    targets['v'] = jnp.full((u_AP_init).shape,.0)
    targets['T'] = jnp.full((u_AP_init).shape,1.)
    targets['x'] = x_BOCF[:,:100]
    targets['x_dot'] = x_dot_BOCF[:,:100]*12.9

    N,size,params = read_config(['D','a','epsilon_0','k_T','k','mu_1','mu_2','delta_t_e'
        ,'k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
        'n_0','l_0','spacing'],mode = 'chaos')
    keys =['D','a','epsilon_0','k_T','k','mu_1','mu_2','delta_t_e'
        ,'k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
        'n_0','l_0','spacing']

    params_AP_list = [value for key, value in params_AP.items()]
    
    for i in range(0,7):
        params[i] = params_AP_list[i]

    # for bounds of initial conditions
    u0,v0,T0,x0,x_dot0 = u_AP[0,0],v_AP[0,0],T_AP[0,0],x_BOCF[0,0],x_dot_BOCF[0,0]*12.9

    params_true = dict(zip(keys,params))
    params_low = {key: value - value*tol for key, value in params_true.items()}
    params_high = {key: value + value*tol for key, value in params_true.items()}
    params_low['k_ij_pad'],params_high['k_ij_pad'] = params_true['k_ij_pad'],params_true['k_ij_pad']
    params_low['k_a_pad'],params_high['k_a_pad'] = params_true['k_a_pad'],params_true['k_a_pad']

    kwargs_sys = {'size': kwargs_training['size'],
                'N_sys':kwargs_training['N_sys'],
                'par_tol': kwargs_training['tol'],
                'params_true': params_true}
    kwargs_adoptODE = {'epochs': kwargs_training['epochs'],'N_backups': kwargs_training['N_backups'],'lr':kwargs_training['lr'],
                'lower_b': params_low,'upper_b': params_high,
                'lr_y0':kwargs_training['lr_y0'],
                # 'lr_ip': kwargs_training['lr_ip'],
                'lower_b_y0': {'u':kwargs_training['u_low'],'v':kwargs_training['v_low'],'T':kwargs_training['T_low'],'x':x0,'x_dot':x_dot0},
                'upper_b_y0': {'u':kwargs_training['u_high'],'v':kwargs_training['v_high'],'T':kwargs_training['T_high'],'x':x0,'x_dot':x_dot0}}

    dataset_MSD = dataset_adoptODE(define_MSD_rec,
                                targets,
                                t_evals, 
                                kwargs_sys,
                                kwargs_adoptODE,
                                true_params=params_true)
    
    return dataset_MSD

def continue_dataset(dataset_MSD, kwargs_training,run):
    '''returns dataset for training'''
    keep_data = kwargs_training['keep_data']
    keep_params = kwargs_training['keep_params']
    tol = kwargs_training['tol']
    tol_AP = kwargs_training['tol_AP']
    
    t_evals = jnp.linspace(0, 50, 100)/12.9
    u_AP_init = u_AP
    targets = {}
    
    targets['x'] = x_BOCF[:,run*100:run*100+100]
    targets['x_dot'] = x_dot_BOCF[:,run*100:run*100+100]*12.9

    if keep_data == True:
        targets['u'] = jnp.broadcast_to(dataset_MSD.ys_sol['u'][0,-1],u_AP_init.shape)
        targets['v'] = jnp.broadcast_to(dataset_MSD.ys_sol['v'][0,-1],u_AP_init.shape)
        targets['T'] = jnp.broadcast_to(dataset_MSD.ys_sol['T'][0,-1],u_AP_init.shape)
    else:
        targets['u'] = jnp.full((u_AP_init).shape,.8)
        targets['v'] = jnp.full((u_AP_init).shape,.0)
        targets['T'] = jnp.full((u_AP_init).shape,1.)

    tol_AP = tol - .2*run
    if tol_AP < 0:
        tol_AP = 0

    if keep_params == True:
        params_true = dataset_MSD.params
        params_train = dataset_MSD.params_train
        par_tol = 0.1
    else:
        params_train = params_true
        par_tol = tol        

    u0,v0,T0,x0,x_dot0 = targets['u'][0,0],targets['v'][0,0],targets['T'][0,0],targets['x'][0,0],targets['x_dot'][0,0]*12.9

    N,size,params = read_config(['D','a','epsilon_0','k_T','k','mu_1','mu_2','delta_t_e'
        ,'k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
        'n_0','l_0','spacing'],mode = 'chaos')
    keys =['D','a','epsilon_0','k_T','k','mu_1','mu_2','delta_t_e'
        ,'k_ij','k_ij_pad','k_j','k_a','k_a_pad','c_a','m','c_damp',
        'n_0','l_0','spacing']
    params_AP_list = [value for key, value in params_AP.items()]
    
    for i in range(0,7):
        params[i] = params_AP_list[i]
    
    params_true = dict(zip(keys,params))
    params_low = {key: value - value*tol for key, value in params_true.items()}
    params_high = {key: value + value*tol for key, value in params_true.items()}
    params_low['k_ij_pad'],params_high['k_ij_pad'] = params_true['k_ij_pad'],params_true['k_ij_pad']
    params_low['k_a_pad'],params_high['k_a_pad'] = params_true['k_a_pad'],params_true['k_a_pad']
    keys_AP = ['D','a','epsilon_0','k','mu_1','mu_2']
    for key in keys_AP:
        params_low[key],params_high[key] = params_true[key] - params_true[key]*tol_AP,params_true[key] + params_true[key]*tol_AP

    kwargs_sys = {'size': kwargs_training['size'],
                'N_sys':kwargs_training['N_sys'],
                'par_tol': kwargs_training['tol'],
                'params_true': params_train}
    kwargs_adoptODE = {'epochs': kwargs_training['epochs'],'N_backups': kwargs_training['N_backups'],'lr':kwargs_training['lr'],
                'lower_b': params_low,'upper_b': params_high,
                'lr_y0':kwargs_training['lr_y0'],
                # 'lr_ip': kwargs_training['lr_ip'],
                'lower_b_y0': {'u':kwargs_training['u_low'],'v':kwargs_training['v_low'],'T':kwargs_training['T_low'],'x':x0,'x_dot':x_dot0},
                'upper_b_y0': {'u':kwargs_training['u_high'],'v':kwargs_training['v_high'],'T':kwargs_training['T_high'],'x':x0,'x_dot':x_dot0}}
    dataset_MSD_2 = dataset_adoptODE(define_MSD_rec,
                                    targets,
                                    t_evals, 
                                    kwargs_sys,
                                    kwargs_adoptODE,
                                    true_params=params_true)
    return dataset_MSD_2

def save_run(run, dataset,kwargs_training):
    '''prints and saves the results of the training'''
    print('Parameter:   True Value:   Recovered Value:')
    for key in dataset.params.keys():
        print(key+(16-len(key))*' '+'{:.3f}         {:-3f}'.format(dataset.params[key], dataset.params_train[key]),'rel err',np.abs(dataset.params[key]-dataset.params_train[key])/dataset.params[key])
    
    tol = kwargs_training['tol']
    keep_data = kwargs_training['keep_data']
    keep_params = kwargs_training['keep_params']
    eta_var = kwargs_training['eta_var']

    file_path = (
        f"../data/SpringMassModel/EtaSweep/"
        f"FullDomainBOCF__tol0{str(tol).split('.')[1]}_"
        f"keepdata{keep_data}_keepparams{keep_params}_etavar{eta_var}.h5"
    )
    if run == 0:
        with h5py.File(file_path, 'w') as f:
            group = f.create_group(f'run{run}')  # Create a group instead of a dataset
            group.create_dataset('u_aol', data=dataset.ys_sol['u'])
            group.create_dataset('u_BOCF',data=u_BOCF[:,run*100:run*100+100])
            group.create_dataset('v_sol', data=dataset.ys_sol['v'])
            group.create_dataset('v_BOCF',data=v_BOCF[:,run*100:run*100+100])
            group.create_dataset('T_sol', data=dataset.ys_sol['T'])
            group.create_dataset('T_BOCF',data=T_BOCF[:,run*100:run*100+100])
            group.create_dataset('x_sol', data=dataset.ys_sol['x'])
            group.create_dataset('x_BOCF',data=x_BOCF[:,run*100:run*100+100])
            params = group.create_group("params_train")  # Create a subgroup
            for key, value in dataset.params_train.items():
                params.attrs[key] = value  # Store values as attributes
    else:
        with h5py.File(file_path, 'a') as f:
            group = f.create_group(f'run{run}')
            group.create_dataset('u_sol', data=dataset.ys_sol['u'])
            group.create_dataset('u_BOCF',data=u_BOCF[:,run*100:run*100+100])
            group.create_dataset('v_sol', data=dataset.ys_sol['v'])
            group.create_dataset('v_BOCF',data=v_BOCF[:,run*100:run*100+100])
            group.create_dataset('T_sol', data=dataset.ys_sol['T'])
            group.create_dataset('T_BOCF',data=T_BOCF[:,run*100:run*100+100])
            group.create_dataset('x_sol', data=dataset.ys_sol['x'])
            group.create_dataset('x_BOCF',data=x_BOCF[:,run*100:run*100+100])
            params = group.create_group("params_train")  # Create a subgroup
            for key, value in dataset.params_train.items():
                params.attrs[key] = value
    print(f"Data for run '{run}' has been successfully added/updated.")

tol = 0.5

kwargs_training = {'epochs': 20,'N_backups': 5,
                    'lr': 2e-3,'lr_y0':2e-2, 
                    'u_low':0,'u_high':99,
                    'v_low':0,'v_high':np.max(v_AP),
                    'T_low':0,'T_high':np.max(T_AP),
                    'eta_var':False,
                    'N_sys':1,
                    'tol':tol,
                    'keep_data':True,
                    'keep_params':True,
                    'size':u_BOCF.shape[-1],}

#start initial Dataset and train

dataset_MSD= initial_dataset(kwargs_training)
print('start training')
params_final, losses, errors, params_history = train_adoptODE(dataset_MSD, print_interval=10, save_interval=10)

run = 0
save_run(run ,dataset_MSD,kwargs_training)

#continue training
for run in range(1,10):
    # overwrite old simulation and dataset with new one
    print(f'start run{run}')
    dataset_MSD,Simulation_MSD = continue_dataset(dataset_MSD,Simulation_MSD,kwargs_training,run)
    params_final, losses, errors, params_history = train_adoptODE(dataset_MSD, print_interval=10, save_interval=10)
    save_run(run, dataset_MSD, kwargs_training)

