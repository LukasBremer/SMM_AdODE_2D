#import jax and other libraries for computation
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.signal import convolve2d
from jax.flatten_util import ravel_pytree
from jax.experimental.ode import odeint
from jax import tree_util
from functools import partial
import numpy as np


@jit
def zero_out_edgesFD(force):
        force = force.at[0, :,:].set(0)   # Top row
        force = force.at[-1, :,:].set(0)  # Bottom row
        force = force.at[:, 0,:].set(0)   # Left column
        force = force.at[:, -1,:].set(0)  # Right column
        return force
@jit
def distance_y(xy_grid):
    return (xy_grid[:,1:] - xy_grid[:,:-1])
@jit
def distance_x(xy_grid):
    return (xy_grid[:,:, 1:] - xy_grid[:,:, :-1])
@jit 
def distance_axial(xy_cm_grid,axial_grid):
    return (xy_cm_grid - axial_grid)     
@jit
def zero_out_edges(force):
        force = force.at[:,0, :].set(0)   # Top row
        force = force.at[:,-1, :].set(0)  # Bottom row
        force = force.at[:,:, 0].set(0)   # Left column
        force = force.at[:,:, -1].set(0)  # Right column
        return force
@jit
def force_field_struct(xy_grid,T,par):
    '''calculates the structural forces on the grid; enforces zero force on the edges
    input: shape (2, n, m) array of xy coordinates
    output: shape (2, n, m) array of forces'''

    #y-direction -> 
    #x-direction ^
    
    # x -- x -- x    
    # |    |    |
    # x -- x -- x    
    # |    |    |
    # x -- x -- x    
    l_g = jnp.full(xy_grid.shape, par['l_0'])
    # k_struct = jnp.full(xy_grid.shape, par['k_ij'])

    pad = int((xy_grid.shape[1]-T.shape[0]) / 2 )  # Pad the extended grid equally on all sides
    # Pad the extended grid equally on all sides


    d_y = distance_y(xy_grid)#points to the top
    d_x = distance_x(xy_grid)#points to the right

    # Apply padding on vertical distances
    d_y_upward = jnp.pad(d_y, ((0, 0), (1, 0), (0, 0)),mode ='constant',constant_values = 1)  
    k_struct = jnp.full((1,T.shape[0],T.shape[1]+1), par['k_ij'])
    k_struct_pad= jnp.pad(k_struct, ((0,0), (pad, pad), (pad, pad)), mode="constant", constant_values=par['k_ij_pad'])
    k_y_upward = jnp.pad(k_struct_pad, ((0, 0), (1, 0), (0, 0)))

    d_y_downward = jnp.pad(d_y, ((0, 0), (0, 1), (0, 0)),mode ='constant',constant_values = 1)  
    k_struct = jnp.full((1,T.shape[0],T.shape[1]+1), par['k_ij'])
    k_struct_pad = jnp.pad(k_struct, ((0,0), (pad, pad), (pad, pad)), mode="constant", constant_values=par['k_ij_pad'])
    k_y_downward = jnp.pad(k_struct_pad, ((0, 0), (0, 1), (0, 0)))

    # Apply padding on horizontal distances
    d_x_right = jnp.pad(d_x, ((0, 0), (0, 0), (1, 0)),mode ='constant',constant_values = 1) 
    k_struct = jnp.full((1,T.shape[0]+1,T.shape[1]), par['k_ij'])
    k_struct_pad = jnp.pad(k_struct, ((0,0), (pad, pad), (pad, pad)), mode="constant", constant_values=par['k_ij_pad'])
    k_x_right = jnp.pad(k_struct_pad, ((0, 0), (0, 0), (1, 0)))
    
    d_x_left= jnp.pad(d_x, ((0, 0), (0, 0), (0, 1)),mode ='constant',constant_values = 1)  
    k_struct = jnp.full((1,T.shape[0]+1,T.shape[1]), par['k_ij'])
    k_struct_pad = jnp.pad(k_struct, ((0,0), (pad, pad), (pad, pad)), mode="constant", constant_values=par['k_ij_pad'])
    k_x_left = jnp.pad(k_struct_pad, ((0, 0), (0, 0), (0, 1)))
   

    y_force_upward = k_y_upward*(l_g-jnp.linalg.norm(d_y_upward,axis = 0)) * d_y_upward/jnp.linalg.norm(d_y_upward,axis = 0) #force points to -y-direction
    y_force_downward = -k_y_downward*(l_g-jnp.linalg.norm(d_y_downward,axis = 0)) * d_y_downward/jnp.linalg.norm(d_y_downward,axis = 0) #force points to +y-direction
    x_force_left = -k_x_left*(l_g-jnp.linalg.norm(d_x_left,axis = 0)) * d_x_left/jnp.linalg.norm(d_x_left,axis = 0) #force points to -x-direction
    x_force_right = k_x_right*(l_g-jnp.linalg.norm(d_x_right,axis = 0)) * d_x_right/jnp.linalg.norm(d_x_right,axis = 0) #force points to +x-direction
    # print(d_y_upward/jnp.linalg.norm(d_y_upward,axis = 0))
    return zero_out_edges(y_force_upward+ y_force_downward+ x_force_left+ x_force_right)
@jit
def force_field_active(xy_grid,T,par):
    '''calculates the active forces on the grid; enforces zero force on the edges
    input: xy grid shape (2, size+2pad+1, size+2pad+1) array of xy coordinates
        T shape (size, size) array of electric signal
        enforces two lazers of same T outside the simulatio grid and pads the rest of the T values with 0
    output: shape (2, size+2pad+1, size+2pad+1) array of forces
        enforces zero force on the edges'''
    attachment_right, attachment_left = interpolate_active(xy_grid, par['n_0'])
    x_cm = center_of_mass_neighbors(xy_grid)
    
    #all distances point to the center of mass points
    d_act_right_to_bottom = distance_axial(x_cm, attachment_right)
    d_act_right_to_bottom = jnp.pad(d_act_right_to_bottom, ((0, 0), (0, 1), (1, 0)),mode ='constant',constant_values = 1)
    d_act_right_to_top = distance_axial(x_cm, attachment_right)
    d_act_right_to_top = jnp.pad(d_act_right_to_top, ((0, 0), (1, 0), (1, 0)),mode ='constant',constant_values = 1)  
    d_act_left_to_bottom = distance_axial(x_cm, attachment_left)
    d_act_left_to_bottom = jnp.pad(d_act_left_to_bottom, ((0, 0), (0, 1), (0, 1)),mode ='constant',constant_values = 1)
    d_act_left_to_top = distance_axial(x_cm, attachment_left)
    d_act_left_to_top = jnp.pad(d_act_left_to_top, ((0, 0), (1, 0), (0, 1)),mode ='constant',constant_values = 1)
    # pad electric signal 
    T_padded = enforce_T_and_pad(T, xy_grid)#shape (size+2pad, size+2pad)
    # initiate params
    k_active = jnp.full((1,T.shape[0]+2,T.shape[1]+2), par['k_a'])
    # n = jnp.full(T.shape, par['n_0'])
    n = par['n_0']
    pad = int((xy_grid.shape[1]-k_active.shape[1]) / 2 )  # Pad the extended grid equally on all sides
    # Pad the extended grid equally on all sides
    k_active_pad = jnp.pad(k_active, ((0,0), (pad, pad), (pad, pad)), mode="constant", constant_values=par['k_a_pad'])
    l_a = jnp.full(T_padded.shape, ((par['n_0']-par['l_0']/2)**2+(par['l_0']/2)**2)**(1/2))
    
    l_a_effective = l_a / (1+par['c_a']*T_padded)#shape (size+2pad, size+2pad)
    # Compute the spring forces acting on the grid; division by zero on the edges gets nan --> zero out edges in the end
    force_right_to_bottom = -(1-n) * jnp.pad(k_active_pad,((0, 0), (0, 1), (1, 0)))*(jnp.pad(l_a_effective,((0, 0), (0, 1), (1, 0)))-jnp.linalg.norm(d_act_right_to_bottom,axis = 0)) * d_act_right_to_bottom/jnp.linalg.norm(d_act_right_to_bottom,axis = 0) 
    force_right_to_top = -n * jnp.pad(k_active_pad,((0, 0), (1, 0), (1, 0))) *(jnp.pad(l_a_effective,((0, 0), (1, 0), (1, 0))) -jnp.linalg.norm(d_act_right_to_top,axis = 0)) * d_act_right_to_top/jnp.linalg.norm(d_act_right_to_top,axis = 0) 
    force_left_to_bottom = -n * jnp.pad(k_active_pad,((0, 0), (0, 1), (0, 1)))*(jnp.pad(l_a_effective,((0, 0), (0, 1), (0, 1)))-jnp.linalg.norm(d_act_left_to_bottom,axis = 0)) * d_act_left_to_bottom/jnp.linalg.norm(d_act_left_to_bottom,axis = 0)       
    force_left_to_top = -(1-n) * jnp.pad(k_active_pad, ((0, 0), (1, 0), (0, 1)))*(jnp.pad(l_a_effective, ((0, 0), (1, 0), (0, 1)))-jnp.linalg.norm(d_act_left_to_top,axis = 0)) * d_act_left_to_top/jnp.linalg.norm(d_act_left_to_top,axis = 0) 
    
    return zero_out_edges(force_right_to_bottom+force_right_to_top+force_left_to_bottom+force_left_to_top)
@jit
def force_field_active_n_var(xy_grid,T,par,kwargs_sys):
    '''calculates the active forces on the grid; enforces zero force on the edges
    input: xy grid shape (2, size+2pad+1, size+2pad+1) array of xy coordinates
        T shape (size, size) array of electric signal
        enforces two lazers of same T outside the simulatio grid and pads the rest of the T values with 0
    output: shape (2, size+2pad+1, size+2pad+1) array of forces
        enforces zero force on the edges'''
     # pad electric signal 
    T_padded = enforce_T_and_pad(T, xy_grid)#shape (size+2pad, size+2pad)
    # initiate params
    k_active = jnp.full((1,T.shape[0]+2,T.shape[1]+2), par['k_a'])
    # n = jnp.reshape(get_n_dist(kwargs_sys['n_dict'],par),(1,100,100)) 
    n = jnp.reshape(kwargs_sys['n_dist'],(1,100,100))
    
    pad = int((xy_grid.shape[1]-k_active.shape[1]) / 2 )  # Pad the extended grid equally on all sides
    pad_n = int((xy_grid.shape[1]- n.shape[1]) / 2 )  # Pad the extended grid equally on all sides
    # Pad the extended grid equally on all sides
    k_active_pad = jnp.pad(k_active, ((0,0), (pad, pad), (pad, pad)), mode="constant", constant_values=par['k_a_pad'])
    n_pad = jnp.pad(n, ((0,0),(pad_n, pad_n), (pad_n, pad_n)), mode="constant", constant_values=.5)
    l_a = ((n_pad-par['l_0']/2)**2+(par['l_0']/2)**2)**(1/2)
    l_a_effective = l_a / (1+par['c_a']*T_padded)#shape (size+2pad, size+2pad)
    
    attachment_right, attachment_left = interpolate_active(xy_grid, n_pad)
    x_cm = center_of_mass_neighbors(xy_grid)
    
    #all distances point to the center of mass points
    d_act_right_to_bottom = distance_axial(x_cm, attachment_right)
    d_act_right_to_bottom = jnp.pad(d_act_right_to_bottom, ((0, 0), (0, 1), (1, 0)),mode ='constant',constant_values = 1)
    d_act_right_to_top = distance_axial(x_cm, attachment_right)
    d_act_right_to_top = jnp.pad(d_act_right_to_top, ((0, 0), (1, 0), (1, 0)),mode ='constant',constant_values = 1)  
    d_act_left_to_bottom = distance_axial(x_cm, attachment_left)
    d_act_left_to_bottom = jnp.pad(d_act_left_to_bottom, ((0, 0), (0, 1), (0, 1)),mode ='constant',constant_values = 1)
    d_act_left_to_top = distance_axial(x_cm, attachment_left)
    d_act_left_to_top = jnp.pad(d_act_left_to_top, ((0, 0), (1, 0), (0, 1)),mode ='constant',constant_values = 1)
   
    # Compute the spring forces acting on the grid; division by zero on the edges gets nan --> zero out edges in the end
    force_right_to_bottom = -jnp.pad((1-n_pad),((0, 0), (0, 1), (1, 0))) * jnp.pad(k_active_pad,((0, 0), (0, 1), (1, 0)))*(jnp.pad(l_a_effective,((0, 0), (0, 1), (1, 0)))-jnp.linalg.norm(d_act_right_to_bottom,axis = 0)) * d_act_right_to_bottom/jnp.linalg.norm(d_act_right_to_bottom,axis = 0) 
    force_right_to_top = jnp.pad(-n_pad,((0,0),(1,0),(1,0))) * jnp.pad(k_active_pad,((0, 0), (1, 0), (1, 0))) *(jnp.pad(l_a_effective,((0, 0), (1, 0), (1, 0))) -jnp.linalg.norm(d_act_right_to_top,axis = 0)) * d_act_right_to_top/jnp.linalg.norm(d_act_right_to_top,axis = 0) 
    force_left_to_bottom = jnp.pad(-n_pad,((0, 0), (0, 1), (0, 1)))* jnp.pad(k_active_pad,((0, 0), (0, 1), (0, 1)))*(jnp.pad(l_a_effective,((0, 0), (0, 1), (0, 1)))-jnp.linalg.norm(d_act_left_to_bottom,axis = 0)) * d_act_left_to_bottom/jnp.linalg.norm(d_act_left_to_bottom,axis = 0)       
    force_left_to_top = jnp.pad(-(1-n_pad), ((0, 0), (1, 0), (0, 1))) * jnp.pad(k_active_pad, ((0, 0), (1, 0), (0, 1)))*(jnp.pad(l_a_effective, ((0, 0), (1, 0), (0, 1)))-jnp.linalg.norm(d_act_left_to_top,axis = 0)) * d_act_left_to_top/jnp.linalg.norm(d_act_left_to_top,axis = 0) 
    
    return zero_out_edges(force_right_to_bottom+force_right_to_top+force_left_to_bottom+force_left_to_top)
@partial(jit, static_argnames=('n_gaussians',))
def force_field_active_n_var_rec(xy_grid,T,par,n_gaussians):
    '''calculates the active forces on the grid; enforces zero force on the edges
    input: xy grid shape (2, size+2pad+1, size+2pad+1) array of xy coordinates
        T shape (size, size) array of electric signal
        enforces two lazers of same T outside the simulatio grid and pads the rest of the T values with 0
    output: shape (2, size+2pad+1, size+2pad+1) array of forces
        enforces zero force on the edges'''
     # pad electric signal 
    T_padded = enforce_T_and_pad(T, xy_grid)#shape (size+2pad, size+2pad)
    # initiate params
    k_active = jnp.full((1,T.shape[0]+2,T.shape[1]+2), par['k_a'])
    n = jnp.reshape(sum_gaussian(n_gaussians, par, sigma = 1),(1,100,100))
    
    pad = int((xy_grid.shape[1]-k_active.shape[1]) / 2 )  # Pad the extended grid equally on all sides
    pad_n = int((xy_grid.shape[1]- n.shape[1]) / 2 )  # Pad the extended grid equally on all sides
    # Pad the extended grid equally on all sides
    k_active_pad = jnp.pad(k_active, ((0,0), (pad, pad), (pad, pad)), mode="constant", constant_values=par['k_a_pad'])
    n_pad = jnp.pad(n, ((0,0),(pad_n, pad_n), (pad_n, pad_n)), mode="constant", constant_values=.5)
    l_a = ((n_pad-par['l_0']/2)**2+(par['l_0']/2)**2)**(1/2)
    l_a_effective = l_a / (1+par['c_a']*T_padded)#shape (size+2pad, size+2pad)
    
    attachment_right, attachment_left = interpolate_active(xy_grid, n_pad)
    x_cm = center_of_mass_neighbors(xy_grid)
    
    #all distances point to the center of mass points
    d_act_right_to_bottom = distance_axial(x_cm, attachment_right)
    d_act_right_to_bottom = jnp.pad(d_act_right_to_bottom, ((0, 0), (0, 1), (1, 0)),mode ='constant',constant_values = 1)
    d_act_right_to_top = distance_axial(x_cm, attachment_right)
    d_act_right_to_top = jnp.pad(d_act_right_to_top, ((0, 0), (1, 0), (1, 0)),mode ='constant',constant_values = 1)  
    d_act_left_to_bottom = distance_axial(x_cm, attachment_left)
    d_act_left_to_bottom = jnp.pad(d_act_left_to_bottom, ((0, 0), (0, 1), (0, 1)),mode ='constant',constant_values = 1)
    d_act_left_to_top = distance_axial(x_cm, attachment_left)
    d_act_left_to_top = jnp.pad(d_act_left_to_top, ((0, 0), (1, 0), (0, 1)),mode ='constant',constant_values = 1)
   
    # Compute the spring forces acting on the grid; division by zero on the edges gets nan --> zero out edges in the end
    force_right_to_bottom = -jnp.pad((1-n_pad),((0, 0), (0, 1), (1, 0))) * jnp.pad(k_active_pad,((0, 0), (0, 1), (1, 0)))*(jnp.pad(l_a_effective,((0, 0), (0, 1), (1, 0)))-jnp.linalg.norm(d_act_right_to_bottom,axis = 0)) * d_act_right_to_bottom/jnp.linalg.norm(d_act_right_to_bottom,axis = 0) 
    force_right_to_top = jnp.pad(-n_pad,((0,0),(1,0),(1,0))) * jnp.pad(k_active_pad,((0, 0), (1, 0), (1, 0))) *(jnp.pad(l_a_effective,((0, 0), (1, 0), (1, 0))) -jnp.linalg.norm(d_act_right_to_top,axis = 0)) * d_act_right_to_top/jnp.linalg.norm(d_act_right_to_top,axis = 0) 
    force_left_to_bottom = jnp.pad(-n_pad,((0, 0), (0, 1), (0, 1)))* jnp.pad(k_active_pad,((0, 0), (0, 1), (0, 1)))*(jnp.pad(l_a_effective,((0, 0), (0, 1), (0, 1)))-jnp.linalg.norm(d_act_left_to_bottom,axis = 0)) * d_act_left_to_bottom/jnp.linalg.norm(d_act_left_to_bottom,axis = 0)       
    force_left_to_top = jnp.pad(-(1-n_pad), ((0, 0), (1, 0), (0, 1))) * jnp.pad(k_active_pad, ((0, 0), (1, 0), (0, 1)))*(jnp.pad(l_a_effective, ((0, 0), (1, 0), (0, 1)))-jnp.linalg.norm(d_act_left_to_top,axis = 0)) * d_act_left_to_top/jnp.linalg.norm(d_act_left_to_top,axis = 0) 
    
    return zero_out_edges(force_right_to_bottom+force_right_to_top+force_left_to_bottom+force_left_to_top)
@jit
def force_field_passive(xy_grid,par):
    '''calculates the passive forces on the grid; enforces zero force on the edges
    input: shape (2, size+2pad, size+2pad) array of xy coordinates
    output: shape (2, size+2pad, size+2pad) array of forces
        enforces zero force on the edges'''
    attachment_top, attachment_bottom = interpolate_passive(xy_grid, par['n_0'])
    x_cm = center_of_mass_neighbors(xy_grid)
    d_pass_top_to_right = distance_axial(x_cm,attachment_top)
    d_pass_top_to_right = jnp.pad(d_pass_top_to_right, ((0, 0), (1, 0), (1, 0)),mode ='constant',constant_values = 1)
    d_pass_top_to_left = distance_axial(x_cm,attachment_top)
    d_pass_top_to_left = jnp.pad(d_pass_top_to_left, ((0, 0), (1, 0), (0, 1)),mode ='constant',constant_values = 1)
    d_pass_bottom_to_right = distance_axial(x_cm,attachment_bottom)
    d_pass_bottom_to_right = jnp.pad(d_pass_bottom_to_right, ((0, 0), (0, 1), (1, 0)),mode ='constant',constant_values = 1)
    d_pass_bottom_to_left = distance_axial(x_cm,attachment_bottom)
    d_pass_bottom_to_left = jnp.pad(d_pass_bottom_to_left, ((0, 0), (0, 1), (0, 1)),mode ='constant',constant_values = 1)
    # initiate params
    k_passive = jnp.full(xy_grid.shape, par['k_T'])
    l_p = jnp.full(xy_grid.shape, ((par['n_0']-par['l_0']/2)**2+(par['l_0']/2)**2)**(1/2))
    n = par['n_0']
    
    # Compute the spring forces acting on the grid; division by zero on the edges gets nan --> zero out edges in the end
    force_top_to_right = -n * k_passive*(l_p-jnp.linalg.norm(d_pass_top_to_right,axis = 0)) * d_pass_top_to_right/jnp.linalg.norm(d_pass_top_to_right,axis = 0) 
    force_top_to_left = -(1-n) * k_passive*(l_p-jnp.linalg.norm(d_pass_top_to_left,axis = 0)) * d_pass_top_to_left/jnp.linalg.norm(d_pass_top_to_left,axis = 0)
    force_bottom_to_right = -(1-n) * k_passive*(l_p-jnp.linalg.norm(d_pass_bottom_to_right,axis = 0)) * d_pass_bottom_to_right/jnp.linalg.norm(d_pass_bottom_to_right,axis = 0)
    force_bottom_to_left = -n * k_passive*(l_p-jnp.linalg.norm(d_pass_bottom_to_left,axis = 0)) * d_pass_bottom_to_left/jnp.linalg.norm(d_pass_bottom_to_left,axis = 0)
    
    return zero_out_edges(force_top_to_right+force_top_to_left+force_bottom_to_right+force_bottom_to_left)

@jit
def force_field_passive_n_var(xy_grid,par,kwargs_sys):
    '''calculates the passive forces on the grid; enforces zero force on the edges
    input: shape (2, size+2pad, size+2pad) array of xy coordinates
    output: shape (2, size+2pad, size+2pad) array of forces
        enforces zero force on the edges'''
     # initiate params
    k_passive = jnp.full(xy_grid.shape, par['k_T'])
    # n = jnp.reshape(get_n_dist(kwargs_sys['n_dict'],par),(1,100,100)) 
    n = jnp.reshape(kwargs_sys['n_dist'],(1,100,100))
    pad_n = int((xy_grid.shape[1]- n.shape[1]) / 2 )
    n_pad = jnp.pad(n, ((0,0),(pad_n, pad_n), (pad_n, pad_n)), mode="constant", constant_values=.5)
    l_p = ((n_pad-par['l_0']/2)**2+(par['l_0']/2)**2)**(1/2)
    
    attachment_top, attachment_bottom = interpolate_passive(xy_grid, n_pad)
    x_cm = center_of_mass_neighbors(xy_grid)
    d_pass_top_to_right = distance_axial(x_cm,attachment_top)
    d_pass_top_to_right = jnp.pad(d_pass_top_to_right, ((0, 0), (1, 0), (1, 0)),mode ='constant',constant_values = 1)
    d_pass_top_to_left = distance_axial(x_cm,attachment_top)
    d_pass_top_to_left = jnp.pad(d_pass_top_to_left, ((0, 0), (1, 0), (0, 1)),mode ='constant',constant_values = 1)
    d_pass_bottom_to_right = distance_axial(x_cm,attachment_bottom)
    d_pass_bottom_to_right = jnp.pad(d_pass_bottom_to_right, ((0, 0), (0, 1), (1, 0)),mode ='constant',constant_values = 1)
    d_pass_bottom_to_left = distance_axial(x_cm,attachment_bottom)
    d_pass_bottom_to_left = jnp.pad(d_pass_bottom_to_left, ((0, 0), (0, 1), (0, 1)),mode ='constant',constant_values = 1)
   
    # Compute the spring forces acting on the grid; division by zero on the edges gets nan --> zero out edges in the end
    force_top_to_right = -jnp.pad(n_pad, ((0, 0), (1, 0), (1, 0))) * k_passive*(jnp.pad(l_p, ((0, 0), (1, 0), (1, 0)))-jnp.linalg.norm(d_pass_top_to_right,axis = 0)) * d_pass_top_to_right/jnp.linalg.norm(d_pass_top_to_right,axis = 0) 
    force_top_to_left = -jnp.pad((1-n_pad), ((0, 0), (1, 0), (0, 1))) * k_passive*(jnp.pad(l_p, ((0, 0), (1, 0), (0, 1)))-jnp.linalg.norm(d_pass_top_to_left,axis = 0)) * d_pass_top_to_left/jnp.linalg.norm(d_pass_top_to_left,axis = 0)
    force_bottom_to_right = -jnp.pad((1-n_pad), ((0, 0), (0, 1), (1, 0)))* k_passive*(jnp.pad(l_p, ((0, 0), (0, 1), (1, 0)))-jnp.linalg.norm(d_pass_bottom_to_right,axis = 0)) * d_pass_bottom_to_right/jnp.linalg.norm(d_pass_bottom_to_right,axis = 0)
    force_bottom_to_left = -jnp.pad(n_pad, ((0, 0), (0, 1), (0, 1)))* k_passive*(jnp.pad(l_p, ((0, 0), (0, 1), (0, 1)))-jnp.linalg.norm(d_pass_bottom_to_left,axis = 0)) * d_pass_bottom_to_left/jnp.linalg.norm(d_pass_bottom_to_left,axis = 0)
    return zero_out_edges(force_top_to_right+force_top_to_left+force_bottom_to_right+force_bottom_to_left)

@partial(jit, static_argnames=('n_gaussians',))
def force_field_passive_n_var_rec(xy_grid,par,n_gaussians):
    '''calculates the passive forces on the grid; enforces zero force on the edges
    input: shape (2, size+2pad, size+2pad) array of xy coordinates
    output: shape (2, size+2pad, size+2pad) array of forces
        enforces zero force on the edges'''
     # initiate params
    k_passive = jnp.full(xy_grid.shape, par['k_T'])
    n = jnp.reshape(sum_gaussian(n_gaussians, par, sigma = 1),(1,100,100))
    pad_n = int((xy_grid.shape[1]- n.shape[1]) / 2 )
    n_pad = jnp.pad(n, ((0,0),(pad_n, pad_n), (pad_n, pad_n)), mode="constant", constant_values=.5)
    l_p = ((n_pad-par['l_0']/2)**2+(par['l_0']/2)**2)**(1/2)
    
    attachment_top, attachment_bottom = interpolate_passive(xy_grid, n_pad)
    x_cm = center_of_mass_neighbors(xy_grid)
    d_pass_top_to_right = distance_axial(x_cm,attachment_top)
    d_pass_top_to_right = jnp.pad(d_pass_top_to_right, ((0, 0), (1, 0), (1, 0)),mode ='constant',constant_values = 1)
    d_pass_top_to_left = distance_axial(x_cm,attachment_top)
    d_pass_top_to_left = jnp.pad(d_pass_top_to_left, ((0, 0), (1, 0), (0, 1)),mode ='constant',constant_values = 1)
    d_pass_bottom_to_right = distance_axial(x_cm,attachment_bottom)
    d_pass_bottom_to_right = jnp.pad(d_pass_bottom_to_right, ((0, 0), (0, 1), (1, 0)),mode ='constant',constant_values = 1)
    d_pass_bottom_to_left = distance_axial(x_cm,attachment_bottom)
    d_pass_bottom_to_left = jnp.pad(d_pass_bottom_to_left, ((0, 0), (0, 1), (0, 1)),mode ='constant',constant_values = 1)
   
    # Compute the spring forces acting on the grid; division by zero on the edges gets nan --> zero out edges in the end
    force_top_to_right = -jnp.pad(n_pad, ((0, 0), (1, 0), (1, 0))) * k_passive*(jnp.pad(l_p, ((0, 0), (1, 0), (1, 0)))-jnp.linalg.norm(d_pass_top_to_right,axis = 0)) * d_pass_top_to_right/jnp.linalg.norm(d_pass_top_to_right,axis = 0) 
    force_top_to_left = -jnp.pad((1-n_pad), ((0, 0), (1, 0), (0, 1))) * k_passive*(jnp.pad(l_p, ((0, 0), (1, 0), (0, 1)))-jnp.linalg.norm(d_pass_top_to_left,axis = 0)) * d_pass_top_to_left/jnp.linalg.norm(d_pass_top_to_left,axis = 0)
    force_bottom_to_right = -jnp.pad((1-n_pad), ((0, 0), (0, 1), (1, 0)))* k_passive*(jnp.pad(l_p, ((0, 0), (0, 1), (1, 0)))-jnp.linalg.norm(d_pass_bottom_to_right,axis = 0)) * d_pass_bottom_to_right/jnp.linalg.norm(d_pass_bottom_to_right,axis = 0)
    force_bottom_to_left = -jnp.pad(n_pad, ((0, 0), (0, 1), (0, 1)))* k_passive*(jnp.pad(l_p, ((0, 0), (0, 1), (0, 1)))-jnp.linalg.norm(d_pass_bottom_to_left,axis = 0)) * d_pass_bottom_to_left/jnp.linalg.norm(d_pass_bottom_to_left,axis = 0)
    return zero_out_edges(force_top_to_right+force_top_to_left+force_bottom_to_right+force_bottom_to_left)
@jit 
def center_of_mass_neighbors(xy_grid):
    '''calculates the center of mass for each block of 4 neighbors 
    input: shape (2, n, m) array of xy coordinates
    output: shape (2, n-1, m-1) array of xy coordinates'''
    x_coords = xy_grid[0, :, :]
    y_coords = xy_grid[1, :, :]
    
    # Calculate the center of mass for each block of 4 neighbors
    x_com = (x_coords[:-1, :-1] + x_coords[1:, :-1] + x_coords[:-1, 1:] + x_coords[1:, 1:]) / 4
    y_com = (y_coords[:-1, :-1] + y_coords[1:, :-1] + y_coords[:-1, 1:] + y_coords[1:, 1:]) / 4
    
    return jnp.stack([x_com, y_com], axis=0)
@jit
def interpolate_active(xy_grid, n):
    '''interpolates the active spring positions on the right and left walls
    input: shape (2, n, m) array of xy coordinates
    output: 2 arrays of shape (2, n-1, m-1) array of xy coordinates'''
    # Calculate vertical distance between neighboring points in the z-direction
    d_y = distance_y(xy_grid)

    # Compute the attachment point on the right wall using fraction n
    attachment_right = xy_grid[:, :-1, 1:] + n * d_y[:,:,1:]

    # Compute the attachment point on the left wall using fraction (1-n)
    attachment_left = xy_grid[:, :-1, :-1] + (1 - n) * d_y[:, :,:-1]

    # Return the x and z attachment points on the right and left walls respectively
    return attachment_right[:, :, :], attachment_left[:, :, :]
@jit
def interpolate_passive(xy_grid, n):
    '''Interpolates the active spring positions on the top and bottom walls.
    Input: shape (2, n, m) array of xy coordinates
    Output: 2 arrays of shape (2, n-1, m-1) array of xy coordinates'''
    
    d_x = distance_x(xy_grid)  # Compute horizontal distances

    # Top wall interpolation (fraction `n`)
    attachment_top = xy_grid[:, 1:, :-1] + (1 - n) * d_x[:, 1:, :]

    # Bottom wall interpolation (fraction `1 - n`)
    attachment_bottom = xy_grid[:, :-1, :-1] + n * d_x[:, :-1, :]

    return attachment_top, attachment_bottom
@jit
def enforce_T_and_pad(T, xy_grid):
    """Expands T by enforcing single-layer boundary values on left/right & top/bottom, then pads equally.
    
    Args:
        T: jnp.array of shape (size, size) (original simulation grid)
        xy_grid: Coordinates grid  of shape (2, size+2*pad+1, size+2*pad+1)
        pad: Total zero padding (added equally on all sides)
    
    Returns:
        jnp.array of shape (size + 2*pad, size + 2*pad)
    """
    i = 1
    # Enforce left and right boundaries using ONLY the first and last column (not two)
    left_extension = jnp.concatenate([T[:, i:i+1], T[:, i:i+1]], axis=1)  
    right_extension = jnp.concatenate([T[:, -i-1:-i], T[:, -i-1:-i]], axis=1)  
    T_extended_lr = jnp.concatenate([left_extension, T[:,i:-i], right_extension], axis=1)  
    # Enforce top and bottom boundaries using ONLY the first and last row (not two)
    top_extension = jnp.concatenate([T_extended_lr[i:i+1, :], T_extended_lr[i:i+1, :]], axis=0)  
    bottom_extension = jnp.concatenate([T_extended_lr[-i-1:-i, :], T_extended_lr[-i-1:-i, :]], axis=0)  
    T_extended_tb = jnp.concatenate([top_extension, T_extended_lr[i:-i,:], bottom_extension], axis=0)  
    # Assuming that xy_grid size should match the size of T, the padding calculation becomes simpler
    pad = int((xy_grid.shape[1]-T_extended_tb.shape[0]) / 2 )  
    # Pad the extended grid equally on all sides
    T_padded = jnp.pad(T_extended_tb, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    
    return jnp.reshape(T_padded, (1,T_padded.shape[0], T_padded.shape[1]))

#for fiber orientation
@jit
def gaussian_2d(x0, y0, amplitude, sigma = 1):
    """2D Gaussian function"""
    
    x, y = jnp.meshgrid( jnp.linspace(-1, 1, 100),  jnp.linspace(-1, 1, 100))

    return amplitude * jnp.exp(-(((x - x0) / sigma) ** 2 + ((y - y0) / sigma) ** 2))

@partial(jit, static_argnames=('n_gaussians',))
def sum_gaussian(n_gaussians, params, sigma = 1):
    """2D Gaussian function"""
    centers = jnp.linspace(-1, 1, n_gaussians)
    sum = 0 
    
    for x0 in range(3):
        for y0 in range(3):
            sum += gaussian_2d(centers[x0], centers[y0], params['Amp'+str(x0)+str(y0)], sigma)

    return sum

def coeffs_to_params(coeffs):
    indx=0
    params_gaussian = {}
    for i in range(3):
        for j in range(3):
            params_gaussian = params_gaussian|{'Amp'+str(i)+str(j):coeffs[indx]}
            indx+=1
    return params_gaussian
def get_n_dist(n_dict,params):
    filtered_keys = n_dict.keys()
    
    indices = jnp.array([(int(k[1:3]), int(k[3:5])) for k in filtered_keys])
    values = jnp.array([params[k] for k in filtered_keys])

    # Initialize an empty JAX array
    n_distr_reconstructed = jnp.zeros((100, 100))

    # Populate the array using indexed assignment
    n_distr_reconstructed = n_distr_reconstructed.at[indices[:, 0], indices[:, 1]].set(values)
    return n_distr_reconstructed

#for loss and visualization 

@jit
def laplacian_filter(time_series):
    """Applies a Laplacian filter to an entire time series (T, domain_size, domain_size)."""
    # Define Laplacian kernel
    kernel = jnp.array([[0,  1,  0],
                         [1, -4,  1],
                         [0,  1,  0]])

    def laplacian_filter_2d(image):
        """Applies a Laplacian filter to a single 2D frame."""
        # Manually pad image with edge replication
        padded_image = jnp.pad(image, pad_width=1, mode='edge')

        # Apply convolution (JAX only supports 'fill' boundary mode)
        return convolve2d(padded_image, kernel, mode="valid", boundary="fill", fillvalue=0)

    # Vectorize over time dimension (T)
    return vmap(laplacian_filter_2d, in_axes=0, out_axes=0)(time_series)

@jit
def compute_dA(x, A_undeformed):
    """
    Computes the relative area change for a time series of quadrilateral meshes.

    Parameters:
    x : ndarray of shape (T, 2, H, M)
        Time series of x and y coordinate grids.
    A_undeformed : float
        The reference (undeformed) area.

    Returns:
    dA : ndarray of shape (T, H-1, M-1)
        The relative area change over time.
    """

    # Compute edge lengths (vectorized over time T)
    a = jnp.linalg.norm(x[:, :, 1:, 1:] - x[:, :, 1:, :-1], axis=1)  # Right edge
    b = jnp.linalg.norm(x[:, :, 1:, 1:] - x[:, :, :-1, 1:], axis=1)  # Top edge
    c = jnp.linalg.norm(x[:, :, :-1, 1:] - x[:, :, :-1, :-1], axis=1)  # Left edge
    d = jnp.linalg.norm(x[:, :, 1:, :-1] - x[:, :, :-1, :-1], axis=1)  # Bottom edge

    # Compute diagonal lengths
    diagonal1 = jnp.linalg.norm(x[:, :, :-1, 1:] - x[:, :, 1:, :-1], axis=1)  # Top-left to bottom-right
    diagonal2 = jnp.linalg.norm(x[:, :, 1:, 1:] - x[:, :, :-1, :-1], axis=1)  # Top-right to bottom-left

    # Compute helper term
    hlp = (b**2 + d**2 - a**2 - c**2)

    # Compute deformed area using the determinant formula
    A_deformed = jnp.sqrt(4 * diagonal1**2 * diagonal2**2 - hlp**2) / 4

    # Compute relative area change
    dA = A_deformed / A_undeformed - 1

    return dA