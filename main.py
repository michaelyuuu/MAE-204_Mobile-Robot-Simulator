import Nextstate as ns
import numpy as np
import modern_robotics as mr
import FeedbackControl as fc
import TrajectoryGenerator as tg
import matplotlib.pyplot as plt

joint_limits = np.array([[-np.pi/2, np.pi/2]])
dt = 0.01
Kp = np.eye(6) * 15
Ki = np.eye(6) * 0.1
z = 0.0963 # height of the robot chassis frame b center
z_cube = 0.025 # height of the cube

#set up inital and final configuration of the cube
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

r = 0.0475 # wheel radius
L = 0.47/2 # distance between the center of the robot and the wheel
w = 0.15 # distance between the left and right wheels
F6 = r/4 * np.array([[0,0,0,0],
                     [0,0,0,0],
        [-1/(L+w), 1/(L+w), 1/(L+w), -1/(L+w)],
        [1, 1, 1, 1],
        [-1, 1, -1, 1],  
        [0,0,0,0] 
    ])

Tb0 = np.array([[1, 0, 0, 0.1662],
                [0, 1, 0, 0],
                [0, 0, 1, 0.0026],
                [0, 0, 0, 1]])

#initial configuration: [theta, x, y, arm joint angles (5), wheel angles (4)]
config = np.array([0, -0.5, 0.2, 0.5, -0.5, -0.1, -0.5, 0.2, 0, 0, 0, 0])
M0e = np.array([[1, 0, 0, 0.033],
                [0, 1, 0, 0],
                [0, 0, 1, 0.6546],
                [0, 0, 0, 1]])
Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                    [0, -1, 0, -0.5076, 0, 0],
                    [0, -1, 0, -0.3526, 0, 0],
                    [0, -1, 0, -0.2176, 0, 0],
                    [0, 0, 1, 0, 0, 0]]).T

# function to convert state to end-effector pose, arm Jacobian and base Jacobian
def state_to_T(state):
    theta, x, y = state[0], state[1], state[2]
    arm_config = state[3:8]
    Tsb = np.array([[np.cos(theta), -np.sin(theta), 0, x],
                [np.sin(theta), np.cos(theta), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1]])
    
    T0e = mr.FKinBody(M0e, Blist, arm_config) # compute the end-effector pose in the arm base frame
    Adj_t0e_tb0 = mr.Adjoint(mr.TransInv(T0e) @ mr.TransInv(Tb0)) 
    Tse = Tsb @ Tb0 @ T0e # compute the end-effector pose in the world frame
    J_arm = mr.JacobianBody(Blist, arm_config) # compute the arm Jacobian
    J_base = Adj_t0e_tb0 @ F6 # compute the base Jacobian
    return Tse, J_arm, J_base

T_se = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],  
                        [0, 0, 1, 0.5],
                        [0, 0, 0, 1]])

Xerr_sum = np.zeros(6)
pose_init, _, _ = state_to_T(config)
Traj = tg.TrajectoryGenerator(T_se, cube_init_T, cube_end_T, tg.T_ce_grasp, tg.T_ce_standoff, tg.k)

config_list = []
Xerr_list = []
mu1_w_list = []
mu1_v_list = []

# 【修正 1】：先把第一秒的初始狀態放入清單，以免缺少 t=0 的資料
config_list.append(np.append(config, Traj[0][12]))

for i in range(len(Traj)-1):
    T_sd = np.eye(4)
    T_sd[:3, :3] = Traj[i][:9].reshape(3, 3)
    T_sd[:3, 3] = Traj[i][9:12]

    # T_sd_next
    T_sd_next = np.eye(4)
    T_sd_next[:3, :3] = Traj[i+1][:9].reshape(3, 3)
    T_sd_next[:3, 3] = Traj[i+1][9:12]
    
    T_se, J_arm, J_base = state_to_T(config)
    
    # 【修正 2】：正確接住 FeedbackControl 傳回來的 5 個變數，並傳入正確的參數
    V, wheel_speed, joint_speed, Xerr_sum, Xerr = fc.FeedbackControl(T_se, T_sd, T_sd_next, Kp, Ki, dt, J_arm, J_base, Xerr_sum)
    
    # 將關節與輪子速度合併 (注意 NextState 吃的是 [5個關節速度, 4個輪子速度])
    joint_and_wheel_vels = np.concatenate((joint_speed, wheel_speed))
    
    # 預測下一個狀態
    config = ns.NextState(config, joint_and_wheel_vels, dt, 20.0)
    
    # 把下一個狀態與下一個夾爪狀態結合並存檔
    config_list.append(np.append(config, Traj[i+1][12])) 
    Xerr_list.append(Xerr)
    
    # Combine base and arm Jacobians to get the full 6x9 end-effector Jacobian
    Je = np.hstack((J_base, J_arm))
    Jw = Je[:3, :]
    Jv = Je[3:, :]
    
    sigma_w = np.linalg.svd(Jw, compute_uv=False)
    sigma_v = np.linalg.svd(Jv, compute_uv=False)
    
    # 【修正 3】：避免使用 np.inf 導致畫圖 Y 軸爆掉，把無窮大設為 1000
    mu1_w = sigma_w[0] / sigma_w[-1] if sigma_w[-1] > 1e-4 else 1000 
    mu1_v = sigma_v[0] / sigma_v[-1] if sigma_v[-1] > 1e-4 else 1000 
    
    mu1_w_list.append(mu1_w)
    mu1_v_list.append(mu1_v)

# 【修正 4】：加上 fmt="%.6f" 讓 CoppeliaSim 看得懂
np.savetxt("overall_trajectory.csv", config_list, delimiter=",", fmt="%.6f")
print("✅ CSV 檔案已成功生成！總步數：", len(config_list))

Xerr_array = np.array(Xerr_list)

#plot Xerr over time
plt.figure(figsize=(10, 6))
#subplot for angular velocity error
plt.subplot(2, 1, 1)
plt.plot(Xerr_array[:, 0], label="Xerr[0] (wx)")
plt.plot(Xerr_array[:, 1], label="Xerr[1] (wy)")
plt.plot(Xerr_array[:, 2], label="Xerr[2] (wz)")
plt.ylabel("Error[rad]")
plt.title("Angular Velocity Error")
plt.legend()
plt.grid(True)

#subplot for linear velocity error
plt.subplot(2, 1, 2)
plt.plot(Xerr_array[:, 3], label="Xerr[3] (vx)")
plt.plot(Xerr_array[:, 4], label="Xerr[4] (vy)")
plt.plot(Xerr_array[:, 5], label="Xerr[5] (vz)")
plt.xlabel("Time step")
plt.ylabel("Error[m]")
plt.title("Linear Velocity Error")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#plot manipulability factor over time
plt.figure()
plt.plot(mu1_w_list, label="mu_1(A_w) Angular", color='blue')
plt.plot(mu1_v_list, label="mu_1(A_v) Linear", color='orange')
plt.xlabel("Time step")
plt.ylabel("Manipulability Factor")
plt.title("Manipulability over Time")
plt.ylim(0, 100) # 限制 Y 軸高度，讓大部分的波動能夠被看清楚
plt.grid()
plt.legend()
plt.show()