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

def optimize(x,lm,odom,z):
    print(x)
    print(odom)
    print(z)
    initial_values = Values(
        poses=[sf.V2(i,j) for i,j in x],
        landmarks=[sf.V2(i,j) for i,j in lm],
        odom=[sf.V2(i,j) for i,j in odom],
        meas=[sf.V2(i,j) for i,j,k,m in z],
        epsilon=sf.numeric_epsilon,
    )

    factors = []

    # Bearing factors

    for j in range(len(z)):
        factors.append(Factor(
            residual=meas_residual,
            keys=[f"poses[{int(z[j][2])}]", f"landmarks[{int(z[j][3])}]", f"meas[{j}]", "epsilon"],
        ))

    # Odometry factors
    for i in range(len(x) - 1):
        factors.append(Factor(
            residual=odometry_residual,
            keys=[f"poses[{i}]", f"poses[{i + 1}]", f"odom[{i}]", "epsilon"],
        ))

    optimizer = Optimizer(
        factors=factors,
        optimized_keys=[f"poses[{i}]" for i in range(len(x))],
    )

    result = optimizer.optimize(initial_values)

    return result