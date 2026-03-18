import numpy as np

def NextState(current_state, joint_velocity, dt, max_vel):
    """
    计算机器人在 dt 时间后的状态。
    current_state: 12-vector (3 个底盘变量, 5 个机械臂关节, 4 个轮子角度)
    joint_velocity: 9-vector (5 个机械臂关节速度, 4 个轮子速度)
    """
    # 限制最大速度以防超限
    joint_velocity = np.clip(joint_velocity, -max_vel, max_vel)
    
    arm_vel = joint_velocity[:5]
    wheel_vel = joint_velocity[5:9]
    
    chassis_state = current_state[:3]
    arm_state = current_state[3:8]
    wheel_state = current_state[8:12]
    
    # 简单的欧拉积分更新机械臂和轮子角度
    arm_state_next = arm_state + arm_vel * dt
    wheel_state_next = wheel_state + wheel_vel * dt
    
    # youBot 麦克纳姆轮的几何参数
    r = 0.0475
    l = 0.47 / 2
    w = 0.15
    
    # 轮速到底盘速度的转换矩阵 F
    F = (r / 4) * np.array([
        [-1/(l+w),  1/(l+w),  1/(l+w), -1/(l+w)],
        [ 1,        1,        1,        1      ],
        [-1,        1,       -1,        1      ]
    ])
    
    # 计算底盘的 Twist (Vb)
    Vb = F @ (wheel_vel * dt)
    wbz, vbx, vby = Vb[0], Vb[1], Vb[2]
    
    # 根据航迹推算 (Odometry) 更新底盘坐标
    if np.abs(wbz) < 1e-6:
        dq = np.array([0, vbx, vby])
    else:
        dq = np.array([
            wbz,
            (vbx * np.sin(wbz) + vby * (np.cos(wbz) - 1)) / wbz,
            (vby * np.sin(wbz) + vbx * (1 - np.cos(wbz))) / wbz
        ])
        
    phi = chassis_state[0]
    T_sb = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi),  np.cos(phi)]
    ])
    
    chassis_state_next = chassis_state + T_sb @ dq
    
    return np.concatenate((chassis_state_next, arm_state_next, wheel_state_next))