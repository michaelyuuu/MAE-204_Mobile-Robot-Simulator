import modern_robotics as mr
import numpy as np

def get_Jacobian(arm_config):
    """
    根據目前的機械臂狀態，計算 6x9 總雅可比矩陣 Je
    """
    r = 0.0475 
    l = 0.47 / 2
    w = 0.15 
    
    # 底盤到輪子的矩陣 F6
    F6 = (r/4) * np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
        [1, 1, 1, 1],
        [-1, 1, -1, 1],
        [0, 0, 0, 0]
    ])
    
    T_b0 = np.array([
        [1, 0, 0, 0.1662],
        [0, 1, 0, 0],
        [0, 0, 1, 0.0026],
        [0, 0, 0, 1]
    ])
    M_0e = np.array([
        [1, 0, 0, 0.033],
        [0, 1, 0, 0],
        [0, 0, 1, 0.6546],
        [0, 0, 0, 1]
    ])
    Blist = np.array([
        [0, 0, 1, 0, 0.033, 0],
        [0, -1, 0, -0.5076, 0, 0],
        [0, -1, 0, -0.3526, 0, 0],
        [0, -1, 0, -0.2176, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ]).T
    
    # 計算正運動學 T_0e
    T_0e = mr.FKinBody(M_0e, Blist, arm_config)
    
    # 計算底盤雅可比
    Adj_te = mr.Adjoint(mr.TransInv(T_0e) @ mr.TransInv(T_b0))
    J_base = Adj_te @ F6
    
    # 計算手臂雅可比
    J_arm = mr.JacobianBody(Blist, arm_config)
    
    return J_base, J_arm

def FeedbackControl(T_se, T_sd, T_sd_next, Kp, Ki, dt, J_arm, J_base, Xerr_sum):

    Vd = mr.se3ToVec((1/dt) * mr.MatrixLog6(np.dot(mr.TransInv(T_sd), T_sd_next)))
    
    Xerr = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(T_se), T_sd)))
    Xerr_sum += Xerr * dt
    
    adj_xxd = mr.Adjoint(np.dot(mr.TransInv(T_se), T_sd))
    V = adj_xxd @ Vd + Kp @ Xerr + Ki @ Xerr_sum
    
    Je = np.hstack((J_base, J_arm))
    control = np.linalg.pinv(Je, rcond=1e-3) @ V
    
    wheel_speed = np.clip(control[:4], -10, 10)        
    joint_speed = np.clip(control[4:], -np.pi, np.pi)
    
    return V, wheel_speed, joint_speed, Xerr_sum, Xerr