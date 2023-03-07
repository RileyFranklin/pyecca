import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import time


def euler2rot(psi, theta, phi):
    # Assumes yaw, pitch, roll sequence (3-2-1)
    
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1],
                  ])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)],
                  ])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)],
                  ])
    
    R = np.matmul(np.matmul(Rz, Ry), Rx)
    
    return R


def euler2rot_casadi(psi, theta, phi):
    # Assumes yaw, pitch, roll sequence (3-2-1)
    
    Rz = ca.SX.zeros(3,3)
    Rz[0,0] = ca.cos(psi)
    Rz[0,1] = -ca.sin(psi)
    Rz[1,0] = ca.sin(psi)
    Rz[1,1] = ca.cos(psi)
    Rz[2,2] = 1.0
    # Rz = np.array([[ca.cos(psi), -ca.sin(psi), 0],
    #                [ca.sin(psi), ca.cos(psi), 0],
    #                [0, 0, 1],
    #               ])
    
    
    Ry = ca.SX.zeros(3,3)
    Ry[0,0] = ca.cos(theta)
    Ry[0,2] = ca.sin(theta)
    Ry[1,1] = 1.0
    Ry[2,0] = -ca.sin(theta)
    Ry[2,2] = ca.cos(theta)
    # Ry = np.array([[ca.cos(theta), 0, ca.sin(theta)],
    #                [0, 1, 0],
    #                [-ca.sin(theta), 0, ca.cos(theta)],
    #               ])
    
    Rx = ca.SX.zeros(3,3)
    Rx[0,0] = 1.0
    Rx[1,1] = ca.cos(phi)
    Rx[1,2] = -ca.sin(phi)
    Rx[2,1] = ca.sin(phi)
    Rx[2,2] = ca.cos(phi)
    # Rx = np.array([[1, 0, 0],
    #                [0, ca.cos(phi), -ca.sin(phi)],
    #                [0, ca.sin(phi), ca.cos(phi)],
    #               ])
    
    R_sym = Rz@Ry@Rx
    
    return R_sym


def applyT(points, T):
    # Applies a tranformation matrix to an array of [x,y,z] points
    new_points = np.zeros([len(points), 3])
    for lcv, point in enumerate(points):
        P_k = np.expand_dims(np.hstack([point, 1]), axis=1)
        P_k_trans = np.matmul(T, P_k).T
        new_points[lcv, :] = P_k_trans[0][0:3]
    
    return new_points


def build_cost(points_0, points_1, all_sym):
    """
    @param points_0 : n x 3 numpy matrix of points observed at pose 0
    @param points_1 : n x 3 numpy matrix of points observed at pose 1
    @param all_sym : all symbolic
    """
    
    # covariance for points
    Q = ca.SX(4, 4) 
    std = 1
    Q[0, 0] = std**2
    Q[1, 1] = std**2
    Q[2, 2] = std**2
    Q[3, 3] = std**2
    Q_I = ca.inv(Q)
    
    # Form symbolic transformation matrix
    # Rotation matrix, SO(3)
    R_sym = euler2rot_casadi(all_sym[0], all_sym[1], all_sym[2])
    
    # Translation
    t_sym = ca.SX.zeros(3,1)
    t_sym[0] = all_sym[3]
    t_sym[1] = all_sym[4]
    t_sym[2] = all_sym[5]
    
    # Last row
    last_row = ca.SX.zeros(1,4)
    last_row[0,3] = 1.0
    
    # Combine into tranformation matrix, SE(3)
    T_sym = ca.vertcat(ca.horzcat(R_sym, t_sym), last_row)
                        
    # compute cost
    J = ca.SX.zeros(1,1)
        
    # Landmark moving cost
    for k in range(len(points_0)):
        # Create P_0
        P_0 = ca.SX.zeros(4,1)
        P_0[0,0] = points_0[k,0]
        P_0[1,0] = points_0[k,1]
        P_0[2,0] = points_0[k,2]
        P_0[3,0] = 1.0
        # Create P_1
        P_1 = ca.SX.zeros(4,1)
        P_1[0,0] = points_1[k,0]
        P_1[1,0] = points_1[k,1]
        P_1[2,0] = points_1[k,2]
        P_1[3,0] = 1.0
        # error
        e = P_1 - T_sym@P_0
        # cost
        J += e.T@Q_I@e
        
    return J