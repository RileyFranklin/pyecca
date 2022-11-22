import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

def g(x: np.array, u: np.array, dt: float):
    """
    Vehicle dynamics propagation.
    
    @param x: vehicle state
    @param u: vehicle input
    @param dt: time step
    """
    return x + u*dt

def data_association(x: np.array, z: np.array, landmarks: np.array):
    """
    Associates measurement with known landmarks using maximum likelihood
    
    @param x: state of vehicle
    @param z: measurement
    @param landmarks: map of known landmarks
    """
    dm = landmarks - x
    rng_pred = np.linalg.norm(dm, axis=1)
    bearing_pred = np.arctan2(dm[:, 1], dm[:, 0])    
    z_error_list = np.array([rng_pred, bearing_pred]).T - np.array([z])
    # TODO: Make an actual number. Covariance of the measurement
    Q = np.array([
        [1, 0],
        [0, 1]])
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
    if noise is not None:
        m_bearing += noise['bearing_std']*(np.random.randn()-.5)/.5 #Is the flat addition of a random variable appropriate or should we scale with magnitude of bearing? also we changed so that the bearing noise is positive or negative
            
        m_range += d*noise['range_std']*np.random.randn() # This may shrink large range to be below range_max. Need to fix
    return [m_range, m_bearing]

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
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        odom += d*(noise['odom_std']*np.random.randn(2) + R@np.array([noise['odom_bx_bias'], noise['odom_by_bias']]))#right now this is only positive noise, only overestimates
    return list(odom)

def simulate(noise=None, plot=False):
    xi = np.array([0.0, 0.0])
    xhi = xi
    xi_prev = xi
    dt = 2

    # landmarks
    l = np.array([
        [0, 2],
        [4, 6],
        [9, 1],
        [9, 1.5],
        [9,10],
        # [-7,15],
    ])
    lhi = l # + np.random.randn(*l.shape)*0.01

    hist = {
        't': [],
        'x': [],
        'u': [],
        'odom': [],
        'z': [],
        #'xh': [xhi],     # .append() seems to overwrite to same xhi every iteration
        'xh': [],
        'lh': [],
        'J': []
    }
    hist['xh'].append(xhi.tolist())

    # l = truth
    # lh = estimate
    # lh_sym = symbolic estimate of landmarks
    
    J = 0
    # Symbolic estimated states and landmark
    xh_sym = ca.SX.sym('xh0', 1, 2)
    
    lh_sym = ca.SX.sym('lh0', 1, 2)
    for i in range(len(l)-1):
        lh_sym = ca.vertcat(lh_sym, ca.SX.sym('lh'+str(i+1), 1, 2))

    for i in range(10):
        ti = dt*i

        # measure landmarks
        z = measure_landmarks(xi, l, noise=noise)

        # propagate
        ui = np.array([np.cos(ti/10), np.sin(ti/10)])
        xi = g(xi_prev, ui, dt)

        # odometry
        odom = measure_odom(xi, xi_prev, noise=noise)
        
        xhi += odom
                
        hist['t'].append(ti)
        hist['x'].append(xi)
        hist['u'].append(ui)
        hist['odom'].append(np.hstack([odom, i]))
        hist['xh'].append(xhi.tolist())
        hist['lh'].append(lhi)
            
        # cost
        
        # Call graph slam
        # Need to have previous xstar vector (history of x)
        # odom, z, assoc
        
        xh_sym = ca.vertcat(xh_sym, ca.SX.sym('xh'+str(i+1), 1, 2))
        # For now assume previous associations are good
        # TODO: Need to fix. Cannot use truth landmark locations. Really should be lh
        
        assoc = [ data_association(hist['xh'][-2], zi, lhi) for zi in z ]
        J += build_cost(
            odom=odom,
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
        x_temp = xh_sym.T.reshape((xh_sym.shape[0]*2,1))
        l_temp = lh_sym.T.reshape((lh_sym.shape[0]*2,1))
        nlp['x']= ca.vertcat(x_temp, l_temp)       # decision vars
        nlp['f'] = J           # objective
        nlp['g'] = 0             # constraints

        # Create solver instance
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.sb': 'yes'}
        F = ca.nlpsol('F','ipopt',nlp,opts);
        
        # Solve the problem using a guess
        # This uses original landmark/measure association (associates which landmark we think the measurement is measuring)
        x_input = np.hstack([np.array(hist['xh']).reshape(-1), lhi.reshape(-1)])
        optim = F(x0=x_input)
        hist['J'].append(optim['f'])

        n_x = len(hist['x'])+1
        n_l = len(l)
        xh = np.reshape(optim['x'][0:2*n_x], [n_x,2])
        lh = np.reshape(optim['x'][2*n_x:None], [n_l,2])
        
        xi_prev = xi
        xhi = xh[-1,:]
        
        hist['xh'] = xh.tolist()
        hist['lh'] = lh.tolist() 

        for zi in z:
            hist['z'].append(np.hstack([zi, i]))

    for key in hist.keys():
        hist[key] = np.array(hist[key])
    
    # Print optimized xh and odom for comparison
    # print(hist['xh'])
    # Convert odom to cartesian coordinates
    xodom = np.zeros([len(hist['xh']),2])
    for i in range(len(hist['odom'])):
        xodom[i+1,:] = xodom[i,:] + hist['odom'][i,0:2]
#     print(xodom)
#     print(hist['xh']-xodom)
    
#     print(l)
#     print(hist['lh'])

    if plot:
        fig = plt.figure(1)
        plt.plot(l[:, 0], l[:, 1], 'bo', label='landmarks')
        plt.plot(hist['x'][:, 0], hist['x'][:, 1], 'r.', label='states', markersize=10)

        # plot odom
        x_odom = np.array([0, 0], dtype=float)
        x_odom_hist = [x_odom]
        for odom in hist['odom']:
            x_odom = np.array(x_odom) + np.array(odom[:2])
            x_odom_hist.append(x_odom)
        x_odom_hist = np.array(x_odom_hist)
        plt.plot(x_odom_hist[:, 0], x_odom_hist[:, 1], 'g.', linewidth=3, label='odom')
        
        # plot best estimate
        plt.plot(hist['xh'][:,0], hist['xh'][:,1], 'm.', linewidth=3, label='xh')
        
        # plot best estimate landmarks
        plt.plot(hist['lh'][:, 0], hist['lh'][:, 1], 'ko', label='lh')
        
        # plot measurements
        for rng, bearing, xi in hist['z']:
            xi = int(xi)
            x = hist['xh'][xi, :]
            plt.arrow(x[0], x[1], rng*np.cos(bearing) , rng*np.sin(bearing), width=0.1,
                          length_includes_head=True)
        

        # # plot measurements
        # for rng, bearing, xi in hist['z']:
        #     xi = int(xi)
        #     x = x_odom_hist[xi, :]
        #     plt.arrow(x[0], x[1], rng*np.cos(bearing) , rng*np.sin(bearing), width=0.1,
        #                   length_includes_head=True)

        # plt.axis([0, 10, 0, 10])
        plt.axis([5, 10, 0, 2])
        plt.grid()
        plt.legend()
        plt.axis('equal');

    return locals()

def J_graph_slam(hist,x_meas,landmarks):
    J = 0
    
    Q = np.eye(2)  # meas covariance  ##look into more realistice covariance
    R = np.eye(2)  # odom covariance
    R_I = np.linalg.inv(R)
    Q_I = np.linalg.inv(Q)
    
    n_x = len(hist['x'])
    odom_pred= hist['odom_pred']
    for i in range(n_x):        
        # compute odom cost
        u = hist['u'][i]
        odom = hist['odom'][i]
        e_x = np.array(odom[:2]) - np.array(odom_pred[i][:2])
        J += e_x.T@R_I@e_x
    
    n_m = len(hist['z'])
    
    for i in range(n_m):
        # compute measurement cost
        rng, brg, xi = hist['z'][i]
        z_i = np.array([rng, brg])
        c_i = data_association(x_meas[int(xi)], z_i, landmarks)#this should use x predicted based off of our input
        z_i_predicted = h(x_meas[i], landmarks[c_i])#this should use x predicted based off of our input
        e_z = np.array(z_i) - np.array(z_i_predicted)
        J += e_z.T@Q_I@e_z
        return J
    

def build_cost(odom, z1, assoc, xh0, xh1, lh_sym):
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
    Q = ca.SX(2, 2) 
    rng_std = 1
    bearing_std = 1
    Q[0, 0] = rng_std**2
    Q[1, 1] = bearing_std**2
    Q_I = ca.inv(Q)

    # covariance for odometry
    R = ca.SX(2, 2) 
    odom_x_std = 1
    odom_y_std = 1
    R[0, 0] = odom_x_std**2
    R[1, 1] = odom_y_std**2
    R_I = ca.inv(R)
                        
    # compute cost
    # -------------------
    # for odometry measurement
    odom_pred = xh1 - xh0
    e_x = ca.SX.zeros(1,2)
    e_x[0] = odom[0] - odom_pred[0]
    e_x[1] = odom[1] - odom_pred[1]
    J = e_x@R_I@e_x.T

    # for each (rng, bearing) measurement
    for j in range(len(z1)):
        rng, bearing = z1[j, :]
        li = assoc[j]
        
        # predicted measurement
        z_pred = ca.SX(2, 1);
        dm = lh_sym[li, :] - xh0
        z_pred[0] = ca.norm_2(dm)  # range
        z_pred[1] = ca.arctan2(dm[1], dm[0]) # bearing

        # error
        e_z = z1[j, :2] - z_pred
        # cost
        J += e_z.T@Q_I@e_z
        
    return J

def plot_me(sim):
    hist = sim['hist']
    landmarks = sim['landmarks']
    fig = plt.figure(1)
    plt.plot(landmarks[:, 0], landmarks[:, 1], 'bo', label='landmarks')
    plt.plot(hist['x'][:, 0], hist['x'][:, 1], 'r.', label='states', markersize=10)
    # plot odom
    x_odom = np.array([0, 0], dtype=float)
    x_odom_hist = [x_odom]
    for odom in hist['odom']:
        x_odom = np.array(x_odom) + np.array(odom[:2])
        x_odom_hist.append(x_odom)
    x_odom_hist = np.array(x_odom_hist)
    plt.plot(x_odom_hist[:, 0], x_odom_hist[:, 1], 'g.', linewidth=3, label='odom')
    # plot measurements
    for rng, bearing, xi in hist['z']:
        xi = int(xi)
        x = x_odom_hist[xi, :]
        plt.arrow(x[0], x[1], rng*np.cos(bearing) , rng*np.sin(bearing), width=0.1,
                      length_includes_head=True)
        
        
    # plt.axis([0, 10, 0, 10])
    plt.axis([5, 10, 0, 2])
    plt.grid()
    plt.legend()
    plt.axis('equal');
    return