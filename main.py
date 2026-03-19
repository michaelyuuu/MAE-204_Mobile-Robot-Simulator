import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt

import Nextstate as ns
import FeedbackControl as fc
import TrajectoryGenerator as tg
# Main file to run the mobile manipulator simulation.
# This file sets up the robot parameters, generates a trajectory for the end-effector to follow,
# and runs a control loop to compute the necessary joint and wheel velocities to follow the trajectory.
# The resulting states are saved to "overall_trajectory.csv" for analysis and playback in the simulator Scene 6.
# The control loop also computes and stores the error and manipulability metrics for further analysis.
# To run the three different cases ("best case", "overshoot" "New Task"), adjust the proportional gain and
# integral gain accordingly, and comment out the undesired initial / goal pose of the cube.


# Simulation setup
dt = 0.01
Kp = np.eye(6) * 15   # proportional gain (tuned manually)
Ki = np.eye(6) * 0.1  # small integral term to reduce steady-state error


z = 0.0963     # base frame centerheight
z_cube = 0.025 # cube center height


"""Please comment out the undesired case(s)' Initial / goal pose of the cube"""


# # Initial / goal pose of the cube (x, y, yaw) ("Best Case" and "Overshoot")
# block_init = np.array([1, 0, 0])
# block_end  = np.array([0, -1, -np.pi/2])


# Initial / goal pose of the cube (x, y, yaw) ("New Task")
block_init = np.array([0.8, 0.5, np.pi/2])
block_end  = np.array([1, -1, -np.pi/2])

# Convert cube pose to SE(3)
cube_init_T = np.array([
    [np.cos(block_init[2]), -np.sin(block_init[2]), 0, block_init[0]],
    [np.sin(block_init[2]),  np.cos(block_init[2]), 0, block_init[1]],
    [0, 0, 1, z_cube],
    [0, 0, 0, 1]
])

cube_end_T = np.array([
    [np.cos(block_end[2]), -np.sin(block_end[2]), 0, block_end[0]],
    [np.sin(block_end[2]),  np.cos(block_end[2]), 0, block_end[1]],
    [0, 0, 1, z_cube],
    [0, 0, 0, 1]
])

# Base kinematics parameters
r = 0.0475 
L = 0.47/2 
w = 0.15 

# Mapping wheel speeds to body twist
F6 = (r/4) * np.array([
    [0,0,0,0],
    [0,0,0,0],
    [-1/(L+w), 1/(L+w), 1/(L+w), -1/(L+w)],
    [1, 1, 1, 1],
    [-1, 1, -1, 1],  
    [0,0,0,0] 
])

# Fixed transform from base to arm base
Tb0 = np.array([
    [1, 0, 0, 0.1662],
    [0, 1, 0, 0],
    [0, 0, 1, 0.0026],
    [0, 0, 0, 1]
])

# Home configuration of end-effector
M0e = np.array([
    [1, 0, 0, 0.033],
    [0, 1, 0, 0],
    [0, 0, 1, 0.6546],
    [0, 0, 0, 1]
])

# Screw axes in body frame
Blist = np.array([
    [0, 0, 1, 0, 0.033, 0],
    [0, -1, 0, -0.5076, 0, 0],
    [0, -1, 0, -0.3526, 0, 0],
    [0, -1, 0, -0.2176, 0, 0],
    [0, 0, 1, 0, 0, 0]
]).T

# Initial robot configuration
# [theta, x, y, arm joints (5), wheel angles (4)]
config = np.array([0, -0.5, 0.2, 0.5, -0.5, -0.1, -0.5, 0.2, 0, 0, 0, 0])

# Gripper poses relative to cube
T_ce_grasp = np.array([
    [0, 0, 1, 0.025],
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1]
])

T_ce_standoff = np.array([
    [0, 0, 1, 0.025],
    [0, 1, 0, 0],
    [-1, 0, 0, 0.2],
    [0, 0, 0, 1]
])

# Convert current state to end-effector pose and Jacobians
def state_to_T(state):
    theta, x, y = state[0], state[1], state[2]
    arm_config = state[3:8]

    Tsb = np.array([
        [np.cos(theta), -np.sin(theta), 0, x],
        [np.sin(theta),  np.cos(theta), 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    
    T0e = mr.FKinBody(M0e, Blist, arm_config)
    Tse = Tsb @ Tb0 @ T0e
    
    # body Jacobian of arm
    J_arm = mr.JacobianBody(Blist, arm_config)
    
    # base Jacobian mapped to end-effector frame
    Adj = mr.Adjoint(mr.TransInv(T0e) @ mr.TransInv(Tb0))
    J_base = Adj @ F6 
    
    return Tse, J_arm, J_base

# Starting pose for trajectory
T_se_init = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0.5],
    [0, 0, 0, 1]
])

# Generate reference trajectory
k = 1
Traj = tg.TrajectoryGenerator(
    T_se_init, cube_init_T, cube_end_T,
    T_ce_grasp, T_ce_standoff, k
)

Xerr_sum = np.zeros(6)
config_list = []
Xerr_list = []
mu1_w_list = []
mu1_v_list = []

# save initial state (important for simulator playback)
config_list.append(np.append(config, Traj[0][12]))
# Main loop
for i in range(len(Traj)-1):
    # current desired pose
    T_sd = np.eye(4)
    T_sd[:3, :3] = Traj[i][:9].reshape(3, 3)
    T_sd[:3, 3]  = Traj[i][9:12]
    # next desired pose (for feedforward)
    T_sd_next = np.eye(4)
    T_sd_next[:3, :3] = Traj[i+1][:9].reshape(3, 3)
    T_sd_next[:3, 3]  = Traj[i+1][9:12]
    # current robot pose
    T_se, J_arm, J_base = state_to_T(config)
    # compute commanded twist
    V, wheel_speed, joint_speed, Xerr_sum, Xerr = fc.FeedbackControl(
        T_se, T_sd, T_sd_next, Kp, Ki, dt, J_arm, J_base, Xerr_sum
    )
    # propagate state forward
    combine_speed = np.concatenate([joint_speed, wheel_speed])
    config = ns.NextState(config, combine_speed, dt, 20.0) # max velocity of 20 rad/s for safety
    # store state + gripper command
    config_list.append(np.append(config, Traj[i+1][12])) 
    Xerr_list.append(Xerr)
    # compute manipulability (angular + linear separately)
    Je = np.hstack((J_base, J_arm))
    Jw = Je[:3, :]
    Jv = Je[3:, :]
    sigma_w = np.linalg.svd(Jw, compute_uv=False)
    sigma_v = np.linalg.svd(Jv, compute_uv=False)
    # avoid division by very small numbers
    mu1_w = sigma_w[0] / sigma_w[-1] if sigma_w[-1] > 1e-4 else 1000
    mu1_v = sigma_v[0] / sigma_v[-1] if sigma_v[-1] > 1e-4 else 1000
    mu1_w_list.append(mu1_w)
    mu1_v_list.append(mu1_v)


np.savetxt("overall_trajectory.csv", config_list, delimiter=",")
print("Simulation done. File saved.")


Xerr_array = np.array(Xerr_list)

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(Xerr_array[:, 0], label="wx")
plt.plot(Xerr_array[:, 1], label="wy")
plt.plot(Xerr_array[:, 2], label="wz")
plt.ylabel("rad/s")
plt.title("Angular error")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(Xerr_array[:, 3], label="vx")
plt.plot(Xerr_array[:, 4], label="vy")
plt.plot(Xerr_array[:, 5], label="vz")
plt.xlabel("time step")
plt.ylabel("m/s")
plt.title("Linear error")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(mu1_w_list, label="angular")
plt.plot(mu1_v_list, label="linear")
plt.xlabel("time step")
plt.ylabel("condition number")
plt.title("Manipulability")
plt.ylim(0, 80)
plt.grid(True)
plt.legend()
plt.show()