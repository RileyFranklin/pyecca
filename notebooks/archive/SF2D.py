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

def landmark_residual(
        landmark: sf.V2, lm: sf.V2, epsilon: sf.Scalar,
    ) -> sf.V1:
    #     landmark: sf.M(8,2), lm: sf.M(8,2), epsilon: sf.Scalar,
    # ) -> sf.V1:
        # covariance for landmark
        Ql = np.array([[0.0,0.0],[0.0,0.0]])
        land_x_std = 1
        land_y_std = 1
        Ql[0, 0] = land_x_std**2
        Ql[1, 1] = land_y_std**2
        Ql_I = np.linalg.inv(Ql)
        
        J = sf.V1()
        # Landmark moving cost
        # print(len(landmark))
        # for j in range(int(len(landmark)/2)):
        #     print(j)
        #     # error
        #     e_l = lm[j,:] - landmark[j,:]
        #     # cost
        #     J += e_l@Ql_I@e_l.T
        #     print("land error ", e_l)
        # print("land cost(j)", J)
        
        # error
        e_l = lm - landmark
        # cost
        J += e_l@Ql_I@e_l.T
        return sf.V1(J)
    
def update_init_values(initial_values, xh, lm, odom, z):
    # initial_values: Previous initial value dictionary generated
    # x: Newest x,y coordinate from rover np.array([x, y]) that will be appended to old array
    # lm: Newest SET of landmarks np.array([[x1,y1],[x2,y2],...]) that will replace old landmarks
    # odom: Newest x,y odom from rover np.array([x, y]) that will be appended to old array
    # z: Newest range, bearing, pose, landmark np.array([]) that will be appended to old array
    
    xh_new = xh
    lm_new = lm
    odom_new = np.vstack([np.array(initial_values['odom']), odom])
    if len(z):
        z_new = np.vstack([np.array(initial_values['meas']), z[:,0:2]])
    else:
        z_new = np.array(initial_values['meas'])
    
    # initial_values = Values(
    #     poses=[sf.V2(i,j) for i,j in xh_new],
    #     landmarks=[sf.V2(i,j) for i,j in lm_new],
    #     odom=[sf.V2(i,j) for i,j in odom_new],
    #     meas=[sf.V2(i,j) for i,j in z_new],
    #     epsilon=sf.numeric_epsilon,
    # )
    

    initial_values = Values(
        poses=[sf.V2(i,j) for i,j in xh_new],
        landmarks=[sf.V2(i,j) for i,j in lm_new],
        lm=[sf.V2(i,j) for i,j in lm_new],
        odom=[sf.V2(i,j) for i,j in odom_new],
        meas=[sf.V2(i,j) for i,j in z_new],
        epsilon=sf.numeric_epsilon,
    )
    
    return initial_values

def update_factor_graph(factors, xh, lm, odom, z):
    # factors: current factor graph to be updated
    # x: Newest x,y coordinate from rover np.array([x, y]) that will be appended to old array
    # lm: Newest SET of landmarks np.array([[x1,y1],[x2,y2],...]) that will replace old landmarks
    # odom: Newest x,y odom from rover np.array([x, y]) that will be appended to old array
    # z: Newest range, bearing, pose, landmark np.array([]) that will be appended to old array
    
    # Run only once when updating factor graph for the first time
    if factors == []:
        for lcv in range(len(lm)):
            factors.append(Factor(
                residual=landmark_residual,
                keys=[f"landmarks[{lcv}]", f"lm[{lcv}]", "epsilon"],
                ))
    
    z_num = 0
    for lcv in range(len(factors)):
        if factors[lcv].name == 'meas_residual':
            z_num += 1

    for j in range(len(z)):
        factors.append(Factor(
            residual=meas_residual,
            keys=[f"poses[{int(z[j][2])}]", f"landmarks[{int(z[j][3])}]", f"meas[{z_num+j}]", "epsilon"],
        ))

    # Odometry factors
    factors.append(Factor(
        residual=odometry_residual,
        keys=[f"poses[{len(xh)-2}]", f"poses[{len(xh)-1}]", f"odom[{len(xh)-2}]", "epsilon"],
    ))
    return factors

def optimize(factors,initial_values):
    poses_keys = [f"poses[{i}]" for i in range(len(initial_values['poses']))]
    landmark_keys = [f"landmarks[{j}]" for j in range(len(initial_values['landmarks']))]
    params = Optimizer.Params(verbose=False)
    optimizer = Optimizer(
        factors=factors,
        optimized_keys=poses_keys + landmark_keys,
        params=params,
    )
    #len(initial_values['poses'])
    result = optimizer.optimize(initial_values)

    return result