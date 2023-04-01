import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import time
import sys
sys.path.insert(0, '../')
from pyecca.lie import se3, so3, matrix_lie_group


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
    
    R = np.matmul(np.matmul(Rx, Ry), Rz)
    
    return R


def euler2rot_casadi(psi, theta, phi):
    # Assumes yaw, pitch, roll sequence (3-2-1)
    
    Rz = ca.SX.zeros(3,3)
    Rz[0,0] = ca.cos(psi)
    Rz[0,1] = -ca.sin(psi)
    Rz[1,0] = ca.sin(psi)
    Rz[1,1] = ca.cos(psi)
    Rz[2,2] = 1.0
    
    
    Ry = ca.SX.zeros(3,3)
    Ry[0,0] = ca.cos(theta)
    Ry[0,2] = ca.sin(theta)
    Ry[1,1] = 1.0
    Ry[2,0] = -ca.sin(theta)
    Ry[2,2] = ca.cos(theta)
    
    Rx = ca.SX.zeros(3,3)
    Rx[0,0] = 1.0
    Rx[1,1] = ca.cos(phi)
    Rx[1,2] = -ca.sin(phi)
    Rx[2,1] = ca.sin(phi)
    Rx[2,2] = ca.cos(phi)
    
    R_sym = Rx@Ry@Rz
    
    return R_sym


def applyT(points, T):
    # Applies a tranformation matrix to an array of [x,y,z] points
    new_points = np.zeros([len(points), 3])
    for lcv, point in enumerate(points):
        P_k = np.expand_dims(np.hstack([point, 1]), axis=1)
        P_k_trans = np.matmul(T, P_k).T
        new_points[lcv, :] = P_k_trans[0][0:3]
    
    return new_points

def build_cost_barfoot(Top , p , y, assoc, weight, epsilon):
    """
    Top: Transformation Matrix
    p: landmarks 1,..,j observed at time k
    y: measurement 1,...,j at time k
    """
    
    # Form symbolic transformation matrix using pyecca
    SE3 = se3._SE3()
    T_sym = SE3.exp(SE3.wedge(epsilon))
                        
    # initiaeliz cost
    J = ca.SX.zeros(1,1)
        
    for j in range(len(y)):
        li = assoc[j] #index for landmarks
        
        #homogeneous landmark j 
        pj = ca.SX.zeros(4,1)
        pj[0,0] = p[li,0]
        pj[1,0] = p[li,1]
        pj[2,0] = p[li,2]
        pj[3,0] = 1.0
        
        #homogeneous measurment j
        yj = ca.SX.zeros(4,1)
        yj[0,0] = y[j,0]
        yj[1,0] = y[j,1]
        yj[2,0] = y[j,2]
        yj[3,0] = 1.0
        e_y = ca.SX.zeros(1,3)
        
        #weight of landmark j
        wj = weight[li]
        
        # compute operating point zj
        zj = Top@pj
        
        # compute error (Barfoot 8.97)
        e = (yj - zj) - SE3.wedge(epsilon)@zj
        
        # compute cost
        J += 1/2*wj*e.T@e
        
    return J


def build_constraint(R):
    
    #Constraint 1: det(R) = 1
    G1 = ca.SX.zeros(1,1)
    a = R[0,0]
    b = R[1,0]
    c = R[2,0]
    d = R[0,1]
    e = R[1,1]
    f = R[2,1]
    g = R[0,2]
    h = R[1,2]
    i = R[2,2]
    G1 = a*e*i + d*h*c + g*b*f - (c*e*g + f*h*a + i*b*d) - 1
    
    #Constraint 2: R^{-1} == R^{T}
    Ainv = ca.inv(R)
    Atrans = R.T
    G2 = ca.SX.zeros(9,1)
    G2[0] = Ainv[0,0] - Atrans[0,0]
    G2[1] = Ainv[0,1] - Atrans[0,1]
    G2[2] = Ainv[0,2] - Atrans[0,2]
    G2[3] = Ainv[1,0] - Atrans[1,0]
    G2[4] = Ainv[1,1] - Atrans[1,1]
    G2[5] = Ainv[1,2] - Atrans[1,2]
    G2[6] = Ainv[2,0] - Atrans[2,0]
    G2[7] = Ainv[2,1] - Atrans[2,1]
    G2[8] = Ainv[2,2] - Atrans[2,2]
    G = ca.vertcat(G1 , G2)
    
    return G



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


def measure_landmark(Rot, x , landmarks, noise=None, range_max=4):
    """
    @param
    Rot: rotation matrix of the vehicle's orientation
    Trans: Translation  of the vehicle's frame
    x: state of the vehicle in inertial frame
    m: all landmarks in inertial frame
    
    @return
    z_list: list of landmarks observed by the vehicle at time k in body frame
    
    """
    
    z_list = []
    for m in landmarks:
        dm = m - x
        if noise is not None:
            dm[0] += noise['x_std']*np.random.randn()
            dm[1] += noise['y_std']*np.random.randn()
            dm[2] += noise['z_std']*np.random.randn()
        
        dist = np.linalg.norm(dm , 1)
        
        #Measurement in body frame
        dm_body = np.matmul(Rot.T , dm)
        
        if dist <= range_max:
            z_list.append(dm_body)
            
    return np.array(z_list)


def data_association(x: np.array, z: np.array, landmarks: np.array, Rot):
    """
    Associates measurement with known landmarks using maximum likelihood
    
    @param x: state of vehicle
    @param z: measurement [x,y,z]
    @param landmarks: map of known landmarks
    """
    dm = landmarks - x
    z_error_list = np.array(dm) - np.array([np.matmul(Rot, z.T)])
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

def noisy_sig(sig, k, dt):
    return np.fft.ifft(np.fft.fft(sig)*dt + np.fft.fft(np.random.normal(scale= k*np.max(sig), size=len(sig)))*dt)/dt
            
    
def Ad(T):
    #Initialize Lie Group
    SO3 = so3._Dcm()
    
    C = T[:3,:3]
    r = T[:3,3]
    return ca.vertcat(ca.horzcat(C, SO3.wedge(r)@C), ca.horzcat(ca.SX.zeros(3,3),C))

def barfoot_solve(Top, p, y, assoc, weight):
    #Initialize Lie Group
    SE3 = se3._SE3()
    SO3 = so3._Dcm()
    
    #the incorporated weights assume that every landmark is observed len(y) = len(w) = len(p)
    Tau = Ad(Top)
    Cop = Top[:3,:3]
    rop = (-Cop.T@Top[:3,3])
    
    w = np.sum(weight)
    P = ca.SX(np.average(p,axis=0, weights=weight))
    Y = ca.SX(np.average(y,axis=0, weights=weight))
    
    I = 0
    for j in range(len(p)):
        pint0=(p[j] - P)
        wj = weight[j]
        I += wj*SO3.wedge(pint0)@SO3.wedge(pint0)
    I=-I/w
    
    M1 = ca.vertcat(ca.horzcat(ca.SX.eye(3), ca.SX.zeros(3,3)), ca.horzcat(SO3.wedge(P),ca.SX.eye(3)))
    M2 = ca.vertcat(ca.horzcat(ca.SX.eye(3), ca.SX.zeros(3,3)), ca.horzcat(ca.SX.zeros(3,3),I))
    M3 = ca.vertcat(ca.horzcat(ca.SX.eye(3), -SO3.wedge(P)), ca.horzcat(ca.SX.zeros(3,3),ca.SX.eye(3)))
    M=M1@M2@M3
    
    W = 0
    for j in range(len(y)):
        li = assoc[j]
        pj = p[li]
        yj = y[j]
        wj = weight[li]
        
        W += wj*(yj-Y)@(pj-P).T  
    W = W/w
    
    b=ca.SX.zeros(1,3)
    b[0] = ca.trace(SO3.wedge([1,0,0])@Cop@W.T)
    b[1] = ca.trace(SO3.wedge([0,1,0])@Cop@W.T)
    b[2] = ca.trace(SO3.wedge([0,0,1])@Cop@W.T) 

    a=ca.vertcat(Y-Cop@(P-rop),b.T-SO3.wedge(Y)@Cop@(P-rop))
    
    #Optimizied pertubation point
    eopt=Tau@ca.inv(M)@Tau.T@a
    
    return eopt
    
    