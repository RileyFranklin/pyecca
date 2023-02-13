import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import SF2D

import symforce
from symforce.values import Values
from symforce.opt.factor import Factor
import symforce.symbolic as sf
from symforce.opt.optimizer import Optimizer


def g(x: np.array, u: np.array, dt: float):
    """
    Vehicle dynamics propagation.
    
    @param x: vehicle state [1X3]
    @param u: vehicle input  [U,0]
    @param dt: time step
    """
    return x + u*dt

def data_association(x: np.array, z: np.array, landmarks: np.array):
    """
    Associates measurement with known landmarks using maximum likelihood
    
    @param x: state of vehicle
    @param z: measurement [RANGE,BEARING,PITCH]
    @param landmarks: map of known landmarks
    """
    dm = landmarks - x
    rng_pred = np.linalg.norm(dm, axis=1)
    bearing_pred = np.arctan2(dm[:, 1], dm[:, 0]) 
    pitch_pred = np.arctan2(dm[:, 2], np.sqrt(dm[:, 0]**2+dm[:, 1]**2))
    z_error_list = np.array([rng_pred, bearing_pred,pitch_pred]).T - np.array([z])
    # TODO: Make an actual number. Covariance of the measurement
    Q = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]])
    Q_I = np.linalg.inv(Q)
    J_list = []
    for z_error in z_error_list:
        J_list.append(z_error.T@Q_I@z_error)
    J_list = np.array(J_list)
    i = np.argmin(J_list)
    return i

def h(x, m, noise=None):
    """
    Predicts the measurements of a landmark at a given state. Returns None 
    if out of range.
    
    @param x: vehicle static
    @param m: landmark
    @param noise: bool to control noise
    """
    dm = m - x
    d = np.linalg.norm(dm)
    m_range = d
    m_bearing = np.arctan2(dm[1], dm[0])
    m_pitch =np.arctan2(dm[2], np.sqrt(dm[0]**2+dm[1]**2))
    if noise is not None:
        m_pitch += noise['bearing_std']*(np.random.randn()-.5)/.5
        m_bearing += noise['bearing_std']*(np.random.randn()-.5)/.5#Is the flat addition of a random variable appropriate or should we scale with magnitude of bearing? also we changed so that the bearing noise is positive or negative
            
        m_range += d*noise['range_std']*np.random.randn() # This may shrink large range to be below range_max. Need to fix
    return [m_range, m_bearing, m_pitch]

def measure_landmarks(x, landmarks, noise=None, range_max=4):
    """
    Predicts all measurements at a given state
    
    @param x: vehicle static
    @param landmarks: list of existing landmarks
    @param noise: bool to control noise
    """
    z_list = []
    for m in landmarks:
        z = h(x, m, noise=noise)
        if z[0] < range_max:
            z_list.append(z)
    return np.array(z_list)

def measure_odom(x, x_prev, noise=None):
    dx = x - x_prev
    d = np.linalg.norm(dx)
    theta = np.arctan2(dx[1], dx[0])
    odom = dx
    if noise is not None:
        #for R matrix, make it 3x3
        R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        odom += d*(noise['odom_std']*np.random.randn(3) + R@np.array([noise['odom_bx_bias'], noise['odom_by_bias'], noise['odom_bz_bias']]))#right now this is only positive noise, only overestimates 
        #TODO add z noise
    return list(odom)

def simulate(noise=None, plot=False, symf=False, tf=10):
    xi = np.array([0.0, 0.0,0.0])
    xh = np.array([xi])
    xi_prev = xi
    dt = 2
    # landmarks
    l = np.array([
        [0, 2,1],
        [1, 1,2],
        [4, 6,.5],
        [9, 1,3],
        [4,2,2],
        [9, 1.5,.5],
        [9,10,.25],
        [-5,15,5],
        # [-7,15],
    ])
    lh = l # + np.random.randn(*l.shape)*0.01

    hist = {
        't': [],
        'x': [xi],
        'u': [],
        'odom': [],
        'z': [],
        'xh': [xh[0,:]],
        'lh': [],
        'J': [],
        'assoc': [],
    }
    # hist['xh'].append(xi)

    # l = truth
    # lh = estimate
    # lh_sym = symbolic estimate of landmarks
    
    J = 0
    # Symbolic estimated states and landmark
    xh_sym = ca.SX.sym('xh0', 1, 3)
    
    lh_sym = ca.SX.sym('lh0', 1, 3)
    for i in range(len(l)-1):
        lh_sym = ca.vertcat(lh_sym, ca.SX.sym('lh'+str(i+1), 1, 3))

    t_vect = np.arange(0,tf,dt)
    for i, ti in enumerate(t_vect):
        
        # measure landmarks
        z = measure_landmarks(xi, l, noise=noise)
        # propagate
        if ti > 60:
            ui = np.array([np.cos(ti/10), -np.sin(ti/10),0])
        else:
            ui = np.array([np.cos(ti/10), np.sin(ti/10),0])
        xi = g(xi_prev, ui, dt)
        # odometry
        odom = measure_odom(xi, xi_prev, noise=noise)

        xh = np.vstack([xh, xh[-1,:]+odom])
        
        # cost
        
        # Call graph slam
        # Need to have previous xstar vector (history of x)
        # odom, z, assoc
        
        xh_sym = ca.vertcat(xh_sym, ca.SX.sym('xh'+str(i+1), 1, 3))
        # For now assume previous associations are good
        # TODO: Need to fix. Cannot use truth landmark locations. Really should be lh
        
        assoc = [ data_association(xh[-2,:], zi, lh) for zi in z ]

        J += build_cost(
            odom=odom,
            lh=lh,
            z1=z, 
            assoc=assoc,
            xh0=xh_sym[-2, :],
            xh1=xh_sym[-1, :],
            lh_sym = lh_sym)
        J_land = build_cost_land(
            odom=odom,
            lh=lh,
            z1=z, 
            assoc=assoc,
            xh0=xh_sym[-2, :],
            xh1=xh_sym[-1, :],
            lh_sym = lh_sym)

        # ca.Function('f_J', [x, l], [J], ['x', 'l'], ['J'])

        # Setup and run optimizer
        # Symbols/expressions
        nlp = {}                 # NLP declaration
        # print(xh_sym.shape)
        # print(xh_sym.shape[0]*2)
        x_temp = xh_sym.T.reshape((xh_sym.shape[0]*3,1))
        l_temp = lh_sym.T.reshape((lh_sym.shape[0]*3,1))
        nlp['x']= ca.vertcat(x_temp, l_temp)       # decision vars
        nlp['f'] = J+J_land      # objective
        nlp['g'] = 0             # constraints

        # Create solver instance
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.sb': 'yes'}
        F = ca.nlpsol('F','ipopt',nlp,opts);

        # Solve the problem using a guess
        # This uses original landmark/measure association (associates which landmark we think the measurement is measuring)
        x_input = np.hstack([np.array(xh).reshape(-1), lh.reshape(-1)])
        optim = F(x0=x_input)
        n_t = len(xh)
        n_l = len(l)
        xh = np.reshape(optim['x'][0:3*n_t], [n_t,3])    # Best estimate of all states for all times at time i
        lh = np.reshape(optim['x'][3*n_t:None], [n_l,3]) # Best estimate of all landmarks at time i
        curr_cost = float(optim['f'])
        xi_prev=xi #set previous state for next loop

        # Simulated data history
        hist['t'].append(ti)      # History of current time
        hist['x'].append(xi)      # History of current state at each time
        hist['u'].append(ui)      # History of current input at each time
        hist['odom'].append(np.hstack([odom, i]))     # History of current odometry reading at each time
        for zi in z:
            hist['z'].append(np.hstack([zi, i]))      # History of measurements recorded at each time step
            
        # Estimator history
        hist['xh'].append(xh[-1,:])    # History of current state estimate at each time
        hist['lh'].append(lh)     # History of location landmark estimate at each time
        hist['J'].append(curr_cost)   # History of minimized cost at each time
        for i in range(len(assoc)):
            hist['assoc'].append(assoc[i])
            

    for key in hist.keys():
        hist[key] = np.array(hist[key])
    
    # Print optimized xh and odom for comparison
    # print(hist['xh'])
    # Convert odom to cartesian coordinates
    xodom = np.zeros([len(hist['xh']),3])
    for i in range(len(hist['odom'])):
        xodom[i+1,:] = xodom[i,:] + hist['odom'][i,0:3]
#     print(xodom)
#     print(hist['xh']-xodom)
    
#     print(l)
#     print(hist['lh'])

    if plot:
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(l[:, 0], l[:, 1],l[:,2], 'bo', label='landmarks')
        ax.plot(hist['x'][:, 0], hist['x'][:, 1], hist['x'][:,2], 'r.', label='states', markersize=10)

#         #plot odom
        x_odom = np.array([0, 0, 0], dtype=float)
        x_odom_hist = [x_odom]
        for odom in hist['odom']:
            x_odom = np.array(x_odom) + np.array(odom[:3])
            x_odom_hist.append(x_odom)
        x_odom_hist = np.array(x_odom_hist)
        ax.plot(x_odom_hist[:, 0], x_odom_hist[:, 1], x_odom_hist[:,2], 'g.', linewidth=3, label='odom')
        
         # plot best estimate history from each time step
#         plt.plot(hist['xh'][:,0], hist['xh'][:,1], 'm.', linewidth=3, label='optimal x for each time')
        
#         # plot best estimate from the final time step
        ax.plot(xh[:,0], xh[:,1], xh[:,2], 'y.', linewidth=3, label='optimal x for all time')
        
#         # plot best estimate landmarks
        ax.plot(hist['lh'][-1,:, 0], hist['lh'][-1,:, 1], hist['lh'][-1,:,2], 'ko', label='lh')
        #plt.plot(hist['lh'][-1])
        
        # plot measurements
        for rng, bearing, pitch, xi in hist['z']:
            xi = int(xi)
            x = hist['xh'][xi, :]
            rng_xy = np.linalg.norm(x[0:1])
            ax.quiver(x[0], x[1], x[2], rng_xy*np.cos(bearing) , rng_xy*np.sin(bearing), rng*np.sin(pitch), normalize = False)
        

#         # # plot measurements
#         # for rng, bearing, xi in hist['z']:
#         #     xi = int(xi)
#         #     x = x_odom_hist[xi, :]
#         #     plt.arrow(x[0], x[1], rng*np.cos(bearing) , rng*np.sin(bearing), width=0.1,
#         #                   length_includes_head=True)

#         # plt.axis([0, 10, 0, 10])
#         plt.axis([5, 10, 0, 2])
#         plt.grid()
#         plt.legend()
#         plt.axis('equal');

    return locals()

def J_graph_slam(hist,x_meas,landmarks):
    J = 0
    
    Q = np.eye(3)  # meas covariance  ##look into more realistice covariance
    R = np.eye(3)  # odom covariance
    R_I = np.linalg.inv(R)
    Q_I = np.linalg.inv(Q)
    
    n_x = len(hist['x'])
    odom_pred= hist['odom_pred']
    for i in range(n_x):        
        # compute odom cost
        odom = hist['odom'][i]
        e_x = np.array(odom[:3]) - np.array(odom_pred[i][:3])
        J += e_x.T@R_I@e_x
    
    n_m = len(hist['z'])
    
    for i in range(n_m):
        # compute measurement cost
        rng, brg,pitch, xi = hist['z'][i]
        z_i = np.array([rng, brg,pitch])
        c_i = data_association(x_meas[int(xi)], z_i, landmarks)#this should use x predicted based off of our input
        z_i_predicted = h(x_meas[i], landmarks[c_i])#this should use x predicted based off of our input
        e_z = np.array(z_i) - np.array(z_i_predicted)
        J += e_z.T@Q_I@e_z
        return J
    

def build_cost(odom, lh, z1, assoc, xh0, xh1, lh_sym):
    """
    @param odom : [delta_x, delta_y]
    @param z : [rng, bearing] vertically stacked
    @param assoc : [ li ] landmark  associations for z, vertically stacked
    @param xh_sym : symbolic states
    @param lh_sym : symbolic landmarks
    """
    # constants
    # -------------------------------------------------------------

    
    # build_cost(prev_J, prev_xstar, curr_odom, curr_z, 
    # prev_J needs to be ca.Function
    
    # covariance for measurement
    Q = ca.SX(3, 3) 
    rng_std = .5
    bearing_std = .5
    pitch_std = .5
    Q[0, 0] = rng_std**2
    Q[1, 1] = bearing_std**2
    Q[2 ,2] = pitch_std**2
    Q_I = ca.inv(Q)

    # covariance for odometry
    R = ca.SX(3, 3) 
    odom_x_std = 3
    odom_y_std = 3
    odom_z_std = 3
    
    R[0, 0] = odom_x_std**2
    R[1, 1] = odom_y_std**2
    R[2, 2] = odom_z_std**2
    R_I = ca.inv(R)
    
    # # covariance for landmark
    # Ql = ca.SX(3, 3) 
    # land_x_std = 1
    # land_y_std = 1
    # Ql[0, 0] = land_x_std**2
    # Ql[1, 1] = land_y_std**2
    # Ql_I = ca.inv(Ql)
                        
    # compute cost
    # -------------------
    # for odometry measurement
    odom_pred = xh1 - xh0
    e_x = ca.SX.zeros(1,3)
    e_x[0] = odom[0] - odom_pred[0]
    e_x[1] = odom[1] - odom_pred[1]
    e_x[2] = odom[2] - odom_pred[2]
    J = e_x@R_I@e_x.T

    # for each (rng, bearing) measurement
    for j in range(len(z1)):
        rng, bearing, pitch = z1[j, :]
        li = assoc[j]
        
        # predicted measurement
        z_pred = ca.SX(3, 1);
        dm = lh_sym[li, :] - xh0
        z_pred[0] = ca.norm_2(dm)  # range
        z_pred[1] = ca.arctan2(dm[1], dm[0]) # bearing
        z_pred[2] = ca.arctan2(dm[2], ca.norm_2(dm[0:1])) # pitch

        # error
        e_z = z1[j, :3] - z_pred
        # cost
        J += e_z.T@Q_I@e_z
        
    # # Landmark moving cost
    # for j in range(len(lh)):
    #     l = lh[j,:]
    #     l_sym = lh_sym[j,:]
    #     # error
    #     e_l = ca.SX.zeros(1,2)
    #     e_l[0] = l[0] - l_sym[0]
    #     e_l[1] = l[1] - l_sym[1]
    #     # cost
    #     J += e_l@Ql_I@e_l.T
    
        
    return J

def build_cost_land(odom, lh, z1, assoc, xh0, xh1, lh_sym):
    """
    @param odom : [delta_x, delta_y]
    @param z : [rng, bearing] vertically stacked
    @param assoc : [ li ] landmark  associations for z, vertically stacked
    @param xh_sym : symbolic states
    @param lh_sym : symbolic landmarks
    """
    # constants
    # -------------------------------------------------------------

    # build_cost(prev_J, prev_xstar, curr_odom, curr_z, 
    # prev_J needs to be ca.Function
    
    # covariance for landmark
    Ql = ca.SX(3, 3) 
    land_x_std = 1
    land_y_std = 1
    land_z_std = 1
    Ql[0, 0] = land_x_std**2
    Ql[1, 1] = land_y_std**2
    Ql[2, 2] = land_z_std**2
    Ql_I = ca.inv(Ql)
                        
    # compute cost
    J = ca.SX.zeros(1,1)
        
    # Landmark moving cost
    for j in range(len(lh)):
        l = lh[j,:]
        l_sym = lh_sym[j,:]
        # error
        e_l = ca.SX.zeros(1,3)
        e_l[0] = l[0] - l_sym[0]
        e_l[1] = l[1] - l_sym[1]
        e_l[2] = l[2] - l_sym[2]
        # cost
        J += e_l@Ql_I@e_l.T
        
    return J