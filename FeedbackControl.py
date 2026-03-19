import modern_robotics as mr
import numpy as np
# FeedbackControl.py
# Computes the control twist for the robot given the current and desired end-effector configurations.
# Inputs:
# - T_se: current end-effector configuration (SE(3))
# - T_sd: desired end-effector configuration at current time step (SE(3))
# - T_sd_next: desired end-effector configuration at next time step (SE(3))
# - Kp: proportional gain matrix (6x6)
# - Ki: integral gain matrix (6x6)
# - dt: time step duration
# - J_arm: arm Jacobian at current configuration (6x5)
# - J_base: base Jacobian at current configuration (6x4)
# - Xerr_sum: accumulated error for integral control (6x1)
# Outputs:
# - V: control twist (6x1)
# - wheel_speed: commanded wheel speeds (4x1)
# - joint_speed: commanded joint speeds (5x1)
# - Xerr_sum: updated accumulated error for integral control (6x1)
#Run this file to test the FeedbackControl function with a sample input and print the resulting control twist, wheel speeds, and joint speeds for analysis.
def FeedbackControl(T_se, T_sd, T_sd_next, Kp, Ki, dt, J_arm, J_base, Xerr_sum):
    # Compute the feedforward reference twist Vd
    Vd = mr.se3ToVec((1/dt) * mr.MatrixLog6(np.dot(mr.TransInv(T_sd), T_sd_next)))
    
    # Compute the error twist Xerr
    Xerr = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(T_se), T_sd)))
    
    # Numerical integration of the error
    Xerr_sum += Xerr * dt
    
    # Compute the control twist with feedforward and feedback terms
    adj_xxd = mr.Adjoint(np.dot(mr.TransInv(T_se), T_sd))
    V = adj_xxd @ Vd + Kp @ Xerr + Ki @ Xerr_sum
    # Combine base and arm Jacobians
    Je = np.hstack((J_base, J_arm))
    # Compute the joint and wheel speeds using pseudo-inverse 
    # rcond=1e-3 is used to handle singularities without dropping matrix rank manually
    control = np.linalg.pinv(Je, rcond=1e-3) @ V
    
    # Clip speeds to prevent unrealistic extreme values
    wheel_speed = np.clip(control[:4], -50, 50)        
    joint_speed = np.clip(control[4:], -np.pi, np.pi)
    
    return V, wheel_speed, joint_speed, Xerr_sum, Xerr
if __name__ == "__main__":
    T_sd = np.array([
    [0, 0, 1, 0.5],
    [0, 1, 0, 0],
    [-1, 0, 0, 0.5],
    [0, 0, 0, 1]
    ])
    r = 0.0475 # wheel radius
    L = 0.47/2 # distance between the center of the robot and the wheel
    w = 0.15 # distance between the left and right wheels
    arm_config = np.array([0, 0, 0.2, -1.6, 0])
    Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                        [0, -1, 0, -0.5076, 0, 0],
                        [0, -1, 0, -0.3526, 0, 0],
                        [0, -1, 0, -0.2176, 0, 0],
                        [0, 0, 1, 0, 0, 0]]).T
    J_arm = mr.JacobianBody(Blist, arm_config)
    Tb0 = np.array([[1, 0, 0, 0.1662],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.0026],
                        [0, 0, 0, 1]])
    M0e = np.array([[1, 0, 0, 0.033],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.6546],
                    [0, 0, 0, 1]])
    T0e = mr.FKinBody(M0e, Blist, arm_config)

    Adj_t0e_tb0 = mr.Adjoint(mr.TransInv(T0e) @ mr.TransInv(Tb0))
    F6 = r/4 * np.array([[0,0,0,0],
                        [0,0,0,0],
            [-1/(L+w), 1/(L+w), 1/(L+w), -1/(L+w)],
            [1, 1, 1, 1],
            [-1, 1, -1, 1],  
            [0,0,0,0] 
        ])
    J_base = Adj_t0e_tb0 @ F6
    T_sd_next = np.array([
        [0, 0, 1, 0.6],
        [0, 1, 0, 0],
        [-1, 0, 0, 0.3],
        [0, 0, 0, 1]
    ])
    Kp = np.eye(6) * 0
    Ki = np.eye(6) * 0
    dt = 0.01
    T_se = np.array([
        [0.170, 0, 0.985, 0.387],
        [0, 1, 0, 0],
        [-0.985, 0, 0.170, 0.570],
        [0, 0, 0, 1]
    ])
    # chassis base to arm base transformation
    theta = 0
    x = 0
    y = 0
    z = 0.0963 # height of the robot chassis frame b center
    Tsb = np.array([[np.cos(theta), -np.sin(theta), 0, x],
                    [np.sin(theta), np.cos(theta), 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]])
    T0e = np.array([[1, 0, 0, 0.033],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.6546],
                    [0, 0, 0, 1]])
    V_e, wheel_speed, joint_speed, Xerr_sum, Xerr = FeedbackControl(T_se, T_sd, T_sd_next, Kp, Ki, dt, J_arm, J_base, np.zeros(6))
    print("V_e: ", V_e)
    print("Xerr: ", Xerr)
    print("wheel_speed: ", wheel_speed)
    print("joint_speed: ", joint_speed)
