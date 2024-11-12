import jax.numpy as jnp
from jax import jit
from jax.flatten_util import ravel_pytree
from HelperAndMechanics.DataReading import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.interpolate

# grid points xj(0 to 3): Mass point x
#      xj0 
#
# xj3  x   xj1
#
#      xj2
#
# shape(x_i)  = (N,2,1)
# shape(x_j) = (N,2,4)
# shape(x_cm) = (N,2,4)
# shape(l_a) = (N,4) different l_a for each x_cm.

# active/passive axial points q 0 to 3; Mass point x; eta > 1/2
#
#              0
#      \      q0a\    
#       \xcm0\    \xcm1
#         /   \q1a  /  
#  3   q0/ /q1 xi q2   /q3  1 
#         /   q2a\    /  
#        xcm3\    \xcm2     
#             \q3a
#              2
#
# shape(q)  = (N,2,4)

#pure mechanics
@jit
def total_force(x, x_j, x_cm, l_a, t, params):
    """defines the total force of the system"""
    f = jnp.zeros(2)
    f = grid_force(x, x_j, params) + axial_force_a(x, x_j, x_cm, l_a, params) + axial_force_p(x, x_j, x_cm, params)
    return f

@jit
def grid_force(x,x_j,params):
    """defines the force that is given by the grid of the system"""
    local_force = jnp.zeros(2)
    k_g = params['k_g']
    l_g = params['l_g']

    for i in range(4):
        #local_force = local_force.at[:].set(local_force + f_ij(x, x_j[i,:], k_g, l_g)) 
        local_force += f_ij(x, x_j[i,:], k_g, l_g)
        
    return local_force

@jit
def axial_force_p(x, x_j, x_cm, params):
    """defines the force that is given by the active axial springs of the system"""
    local_force = jnp.zeros(2)
    eta = params['eta']

    k_p = params['k_p']
    l_p = ((params['eta']-1/2)**2+1/2**2)**(1/2)

    q_j = interpolate_q_p(x_j, x, eta)

    local_force += f_ij(q_j[0,:], x_cm[0,:], k_p, l_p)*(1-eta)
    local_force += f_ij(q_j[1,:], x_cm[3,:], k_p, l_p)*(eta)
    local_force += f_ij(q_j[2,:], x_cm[1,:], k_p, l_p)*(eta)
    local_force += f_ij(q_j[3,:], x_cm[2,:], k_p, l_p)*(1-eta)

    return local_force
    
@jit 
def axial_force_a(x, x_j, x_cm,l_a, params):
    """defines the force that is given by the passive axial springs of the system"""
    local_force = jnp.zeros(2)
    eta = params['eta']
    
    k_a = params['k_a']
    
    q_j = interpolate_q_a(x_j, x, eta)

    local_force += f_ij(q_j[0,:], x_cm[1,:], k_a, l_a[1])*(1-eta)
    local_force += f_ij(q_j[1,:], x_cm[0,:], k_a, l_a[0])*(eta)
    local_force += f_ij(q_j[2,:], x_cm[2,:], k_a, l_a[2])*(eta)
    local_force += f_ij(q_j[3,:], x_cm[3,:], k_a, l_a[3])*(1-eta)

    return local_force

@jit
def interpolate_q_a(x_j, x, eta):
    """interpolates the intersection point of the axial springs q"""
    q = jnp.zeros((4,2))
    
    q = q.at[0,:].set(x + (x_j[0,:]-x) * eta)
    q = q.at[1,:].set(x + (x_j[0,:]-x) * (1-eta))
    q = q.at[2,:].set(x + (x_j[2,:]-x) * (1-eta))
    q = q.at[3,:].set(x + (x_j[2,:]-x) * eta) 

    return q

@jit
def interpolate_q_p(x_j, x, eta):  
    """interpolates the intersection point of the axial springs q"""
    q = jnp.zeros((4,2))

    q = q.at[0,:].set(x + (x_j[3,:]-x) * eta)
    q = q.at[1,:].set(x + (x_j[3,:]-x) * (1-eta))
    q = q.at[2,:].set(x + (x_j[1,:]-x) * (1-eta))
    q = q.at[3,:].set(x + (x_j[1,:]-x) * eta)
    
        
    return q

@jit
def f_ij(x_i, x_j, k, l_0):
    """returns the force from acting on i"""
    return k*(jnp.linalg.norm(x_j-x_i)-l_0)*e_i_to_j(x_i, x_j)

@jit        
def e_i_to_j(x_i, x_j):
    """"returns normal vector from x_i to x_j"""
    return (x_j-x_i)/jnp.linalg.norm(x_j-x_i)



#
#euler Method to test the mechnical implementation
#

def euler_step(xy,t,real_params,x_cm_interp,x_j_interp,l_a_interp,t_interp, dt):

    dxy = sm_eom(xy,t,real_params,x_cm_interp,x_j_interp,l_a_interp,t_interp)
    
    x1 = xy['x1'] + dxy['x1'] * dt
    x2 = xy['x2'] + dxy['x2'] * dt
    y1 = xy['y1'] + dxy['y1'] * dt   
    y2 = xy['y2'] + dxy['y2'] * dt

    xy = {'x1':x1,'x2':x2,'y1':y1,'y2':y2,'f_x':dxy['f_x']}
    t = t + dt

    return xy, t

@jit
def sm_eom(xy, t, params,x_cm_interp,x_j_interp,l_a_interp,t_interp):
    x = jnp.array([xy['x1'], xy['x2']])
    # get interpolated parameters at corresponding time
    x_cm_temp = t_to_value_x(x_cm_interp,t_interp,t)
    x_j_temp = t_to_value_x(x_j_interp,t_interp,t)
    l_a_temp = t_to_value_l(l_a_interp,t_interp,t)

    #initialize total force
    f = total_force(x, x_j_temp, x_cm_temp, l_a_temp, t, params)
    

    #initialize eom
    dx1 = xy['y1']
    dx2 = xy['y2']
    dy1 = 1/params['m'] * (f[0] - params['nu'] * xy['y1'])
    dy2 = 1/params['m'] * (f[1] - params['nu'] * xy['y2'])

    return {'x1':dx1, 'x2':dx2, 'y1':dy1, 'y2':dy2,'f_x':f[0]}

def euler(xy0,iterations,real_params,x_cm_interp,x_j_interp,l_a_interp,t_interp,dt):
#returns a list of all iterations of the system after iterating the euler method
    xy = xy0
    t = 0
    xy_list = [xy['x1']]
    t_list = [t]
    f_x_list = [0]
    for i in range(iterations):
        xy, t = euler_step(xy,t,real_params,x_cm_interp,x_j_interp,l_a_interp,t_interp,dt)
        t_list.append(t)
        xy_list.append(xy['x1'])
        f_x_list.append(xy['f_x'])
    return xy_list,t_list,f_x_list