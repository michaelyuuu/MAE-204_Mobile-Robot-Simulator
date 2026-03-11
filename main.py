import Nextstate as ns
import numpy as np
import modern_robotics as mr
import FeedbackControl as fc
import TrajectoryGenerator as tg
import matplotlib.pyplot as plt

joint_limits = np.array([-np.pi/2, np.pi/2])
dt = 0.01
Kp = np.eye(6) * 2
Ki = np.eye(6) * 0
z = 0.0963 # height of the robot chassis frame b center
z_cube = 0.025 # height of the cube
block_init = np.array([1, 0 ,0])
block_end = np.array([0, -1, -np.pi/2])
cube_init_T = np.array([[np.cos(block_init[2]), -np.sin(block_init[2]), 0, block_init[0]],
                      [np.sin(block_init[2]), np.cos(block_init[2]), 0, block_init[1]],
                      [0, 0, 1, z_cube],
                      [0, 0, 0, 1]])
cube_end_T = np.array([[np.cos(block_end[2]), -np.sin(block_end[2]), 0, block_end[0]],
                      [np.sin(block_end[2]), np.cos(block_end[2]), 0, block_end[1]],
                      [0, 0, 1, z_cube],
                      [0, 0, 0, 1]])
# cube_init_T = np.array([[0, 0, 1, 0.5],
#     [0, 1, 0, 0],
#     [-1, 0, 0, 0.5],
#     [0, 0, 0, 1]])


Tb0 = np.array([[1, 0, 0, 0.1662],
                [0, 1, 0, 0],
                [0, 0, 1, 0.0026],
                [0, 0, 0, 1]])
config = np.array([0, 0, 0, 0, 0, -0.5, -1, 2, 0, 0, 0, 0])
M0e = np.array([[1, 0, 0, 0.033],
                [0, 1, 0, 0],
                [0, 0, 1, 0.6546],
                [0, 0, 0, 1]])
Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                    [0, -1, 0, -0.5076, 0, 0],
                    [0, -1, 0, -0.3526, 0, 0],
                    [0, -1, 0, -0.2176, 0, 0],
                    [0, 0, 1, 0, 0, 0]]).T
def state_to_T(state):
    theta, x, y = state[0], state[1], state[2]
    arm_config = state[3:8]
    Tsb = np.array([[np.cos(theta), -np.sin(theta), 0, x],
                [np.sin(theta), np.cos(theta), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1]])
    T0e = mr.FKinBody(M0e, Blist, arm_config)
    Adj_t0e_tb0 = mr.Adjoint(mr.TransInv(T0e) @ mr.TransInv(Tb0))
    Tse = Tsb @ Tb0 @ T0e
    J_arm = mr.JacobianBody(Blist, arm_config)
    J_base = Adj_t0e_tb0 @ fc.F6
    return Tse, J_arm, J_base
Xerr_sum = np.zeros(6)
pose_init, _, _ = state_to_T(config)
Traj = tg.TrajectoryGenerator(pose_init, cube_init_T, cube_end_T, tg.T_ce_grasp, tg.T_ce_standoff, tg.k)
config_list = []
Xerr_list = []
for i in range(len(Traj)-1):
    T_sd = np.eye(4)
    T_sd[:3, :3] = Traj[i][:9].reshape(3, 3)
    T_sd[:3, 3] = Traj[i][9:12]

    # T_sd_next
    T_sd_next = np.eye(4)
    T_sd_next[:3, :3] = Traj[i+1][:9].reshape(3, 3)
    T_sd_next[:3, 3] = Traj[i+1][9:12]
    # print("T_sd: ", T_sd)
    
    T_se, J_arm, J_base = state_to_T(config)
    # print("pass1")
    V, wheel_speed, joint_speed, Xerr_sum = fc.FeedbackControl(T_se, T_sd, T_sd_next, Kp, Ki, dt, J_arm, J_base, Xerr_sum)
    # print("pass2")
    #joint limits: [-pi/2, pi/2], wheel speed limits: [-50, 50]
    wheel_speed = np.clip(wheel_speed, -50, 50)
    joint_speed = np.clip(joint_speed, -np.pi, np.pi)
    combine_speed = np.concatenate([joint_speed, wheel_speed])
    # print("joint_speed: ", joint_speed)
    next_state = ns.NextState(config, combine_speed, dt, 20)
    # print("pass3")
    next_state = np.append(next_state, Traj[i][12]) # update the gripper state
    config_list.append(next_state)
    config = next_state
    Xerr_list.append(np.linalg.norm(Xerr_sum))
np.savetxt("overall_trajectory.csv", config_list, delimiter=",")
#plot Xerr over time


   
plt.plot(Xerr_list)
plt.xlabel("Time step")
plt.ylabel("Cumulative Error")
plt.title("Cumulative Error over Time")
plt.show()
    