import numpy as np
import symforce
symforce.set_epsilon_to_symbol()
from symforce.values import Values
from symforce.opt.factor import Factor
import symforce.symbolic as sf
from symforce.opt.optimizer import Optimizer

def meas_residual(
        pose: sf.V2, landmark: sf.V2, meas: sf.V2, epsilon: sf.Scalar, 
    ) -> sf.V1:

        Q = np.array([[0.0,0.0],[0.0,0.0]]) 
        rng_std = .5
        bearing_std = .5
        Q[0, 0] = rng_std**2
        Q[1, 1] = bearing_std**2

        Q_I = np.linalg.inv(Q)

        rng, bearing = meas
        # predicted measurement
        dm = sf.V2(landmark - pose)
        z_pred = sf.V2([[dm.norm(epsilon=epsilon)],[sf.atan2(dm[1], dm[0],epsilon=epsilon)]])

        # error
        e_z = meas - z_pred
        # cost
        J = e_z@Q_I@e_z.T

        return sf.V1(J)

def odometry_residual(
        pose_0: sf.V2, pose_1: sf.V2, odom: sf.V2, epsilon: sf.Scalar, 
    ) -> sf.V1:

        R = np.array([[0.0,0.0],[0.0,0.0]]) 
        odom_x_std = 3
        odom_y_std = 3
        R[0, 0] = odom_x_std**2
        R[1, 1] = odom_y_std**2
        R_I = np.linalg.inv(R)

        odom_pred = pose_1-pose_0    # error
        e_x = sf.V2(odom-odom_pred)
        # cost
        J = e_x@R_I@e_x.T
        return sf.V1(J)
    
def update_init_values(initial_values, x, lm, odom, z):
    # initial_values: Previous initial value dictionary generated
    # x: Newest x,y coordinate from rover np.array([x, y]) that will be appended to old array
    # lm: Newest SET of landmarks np.array([[x1,y1],[x2,y2],...]) that will replace old landmarks
    # odom: Newest x,y odom from rover np.array([x, y]) that will be appended to old array
    # z: Newest range, bearing, pose, landmark np.array([]) that will be appended to old array
    
    x_new = np.vstack([np.array(initial_values['poses']), x])
    lm_new = lm
    odom_new = np.vstack([np.array(initial_values['odom']), odom])
    z_new = np.vstack([np.array(initial_values['meas']), z[:,0:2]])
    
    initial_values = Values(
        poses=[sf.V2(i,j) for i,j in x_new],
        landmarks=[sf.V2(i,j) for i,j in lm_new],
        odom=[sf.V2(i,j) for i,j in odom_new],
        meas=[sf.V2(i,j) for i,j in z_new],
        epsilon=sf.numeric_epsilon,
    )
    
    return initial_values

def update_factor_graph(factors, x, lm, odom, z):
    # factors: current factor graph to be updated
    # x: Newest x,y coordinate from rover np.array([x, y]) that will be appended to old array
    # lm: Newest SET of landmarks np.array([[x1,y1],[x2,y2],...]) that will replace old landmarks
    # odom: Newest x,y odom from rover np.array([x, y]) that will be appended to old array
    # z: Newest range, bearing, pose, landmark np.array([]) that will be appended to old array
    for j in range(len(z)):
        factors.append(Factor(
            residual=meas_residual,
            keys=[f"poses[{int(z[j][2])}]", f"landmarks[{int(z[j][3])}]", f"meas[{j}]", "epsilon"],
        ))

    # Odometry factors
    factors.append(Factor(
        residual=odometry_residual,
        keys=[f"poses[{len(x)-1}]", f"poses[{len(x)}]", f"odom[{len(x)-1}]", "epsilon"],
    ))
    return factors

def optimize(factors,initial_values):
    params = Optimizer.Params(verbose=False)
    optimizer = Optimizer(
        factors=factors,
        optimized_keys=[f"poses[{i}]" for i in range(2)],
        params=params,
    )
    #len(initial_values['poses'])
    result = optimizer.optimize(initial_values)

    return result