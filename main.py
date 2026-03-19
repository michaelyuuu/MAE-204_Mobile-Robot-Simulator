import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt

import Nextstate as ns
import FeedbackControl as fc
import TrajectoryGenerator as tg

# Simulation parameters
dt = 0.01
Kp = np.eye(6) * 15.0 # Reduced from 50 to avoid aggressive oscillation
Ki = np.eye(6) * 0.1  # Reduced from 20 

z = 0.0963 # height of the robot chassis frame b center
z_cube = 0.025 # height of the cube

# Setup initial and final configurations of the cube
block_init = np.array([1, 0, 0])
block_end = np.array([0, -1, -np.pi/2])

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

# Robot parameters for base Jacobian
r = 0.0475 
L = 0.47/2 
w = 0.15 
F6 = (r/4) * np.array([
    [0,0,0,0],
    [0,0,0,0],
    [-1/(L+w), 1/(L+w), 1/(L+w), -1/(L+w)],
    [1, 1, 1, 1],
    [-1, 1, -1, 1],  
    [0,0,0,0] 
])

Tb0 = np.array([
    [1, 0, 0, 0.1662],
    [0, 1, 0, 0],
    [0, 0, 1, 0.0026],
    [0, 0, 0, 1]
])

M0e = np.array([
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

# Initial configuration: [theta, x, y, arm joint angles (5), wheel angles (4)]
config = np.array([0, -0.5, 0.2, 0.5, -0.5, -0.1, -0.5, 0.2, 0, 0, 0, 0])

# Pre-define gripper poses relative to the cube
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

# Function to convert state to end-effector pose, arm Jacobian, and base Jacobian
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
    Adj_t0e_tb0 = mr.Adjoint(mr.TransInv(T0e) @ mr.TransInv(Tb0)) 
    Tse = Tsb @ Tb0 @ T0e 
    J_arm = mr.JacobianBody(Blist, arm_config) 
    J_base = Adj_t0e_tb0 @ F6 
    return Tse, J_arm, J_base

# Desired initial reference configuration for the trajectory
T_se_init = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0.5],
    [0, 0, 0, 1]
])

# Generate Reference Trajectory
k = 1
Traj = tg.TrajectoryGenerator(T_se_init, cube_init_T, cube_end_T, T_ce_grasp, T_ce_standoff, k)

Xerr_sum = np.zeros(6)
config_list = []
Xerr_list = []
mu1_w_list = []
mu1_v_list = []

# Append the t=0 state to ensure correct starting position in CoppeliaSim
config_list.append(np.append(config, Traj[0][12]))

# Main Control Loop
for i in range(len(Traj)-1):
    # Current desired state
    T_sd = np.eye(4)
    T_sd[:3, :3] = Traj[i][:9].reshape(3, 3)
    T_sd[:3, 3]  = Traj[i][9:12]

    # Next desired state
    T_sd_next = np.eye(4)
    T_sd_next[:3, :3] = Traj[i+1][:9].reshape(3, 3)
    T_sd_next[:3, 3]  = Traj[i+1][9:12]
    
    # Calculate current actual state matrices
    T_se, J_arm, J_base = state_to_T(config)
    
    # Calculate Control
    V, wheel_speed, joint_speed, Xerr_sum, Xerr = fc.FeedbackControl(
        T_se, T_sd, T_sd_next, Kp, Ki, dt, J_arm, J_base, Xerr_sum
    )
    
    # Integrate to next state (NextState expects [5 arm joints, 4 wheels])
    combine_speed = np.concatenate([joint_speed, wheel_speed])
    config = ns.NextState(config, combine_speed, dt, 20.0)
    
    # Append the state with the corresponding gripper state
    config_list.append(np.append(config, Traj[i+1][12])) 
    Xerr_list.append(Xerr)
    
    # Manipulability calculation
    Je = np.hstack((J_base, J_arm))
    Jw = Je[:3, :]
    Jv = Je[3:, :]
    
    sigma_w = np.linalg.svd(Jw, compute_uv=False)
    sigma_v = np.linalg.svd(Jv, compute_uv=False)
    
    # Replace np.inf with 1000 to prevent plot scaling issues
    mu1_w = sigma_w[0] / sigma_w[-1] if sigma_w[-1] > 1e-4 else 1000
    mu1_v = sigma_v[0] / sigma_v[-1] if sigma_v[-1] > 1e-4 else 1000
    
    mu1_w_list.append(mu1_w)
    mu1_v_list.append(mu1_v)

# Save results using standard decimal format (CoppeliaSim struggles with scientific notation)
np.savetxt("overall_trajectory.csv", config_list, delimiter=",", fmt="%.6f")
print("✅ Simulation complete. Saved to overall_trajectory.csv")

# Plot Error Twist over time
Xerr_array = np.array(Xerr_list)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(Xerr_array[:, 0], label="Xerr[0] (wx)")
plt.plot(Xerr_array[:, 1], label="Xerr[1] (wy)")
plt.plot(Xerr_array[:, 2], label="Xerr[2] (wz)")
plt.ylabel("Error [rad/s]")
plt.title("Angular Velocity Error")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(Xerr_array[:, 3], label="Xerr[3] (vx)")
plt.plot(Xerr_array[:, 4], label="Xerr[4] (vy)")
plt.plot(Xerr_array[:, 5], label="Xerr[5] (vz)")
plt.xlabel("Time step")
plt.ylabel("Error [m/s]")
plt.title("Linear Velocity Error")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot Manipulability over time
plt.figure(figsize=(10, 4))
plt.plot(mu1_w_list, label="mu_1(A_w) Angular", color='blue')
plt.plot(mu1_v_list, label="mu_1(A_v) Linear", color='orange')
plt.xlabel("Time step")
plt.ylabel("Manipulability Factor")
plt.title("Manipulability over Time")
plt.ylim(0, 80) # Bound the Y axis so the normal values are readable
plt.grid(True)
plt.legend()
plt.show()