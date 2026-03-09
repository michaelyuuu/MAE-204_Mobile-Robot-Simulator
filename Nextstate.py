import modern_robotics as mr
import numpy as np
def NextState(current_state, joint_velocity, dt, joint_limits):
    # current_state: [x, y, theta]
    # control_input: [v, w]
    # dt: time step
    r = 0.0475 # wheel radius
    L = 0.47/2 # distance between the center of the robot and the wheel
    w = 0.15 # distance between the left and right wheels
    z = 0.0963 # height of the robot chassis frame b center
    # chassis base to arm base transformation
    Tb0 = np.array([[1, 0, 0, 0.1662],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.0026],
                    [0, 0, 0, 1]])
    M0e = np.array([[1, 0, 0, 0.033],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.6546],
                    [0, 0, 0, 1]])
    chassis_state = current_state[:3]
    arm_state = current_state[3:8]
    wheel_state = current_state[8:12]
    arm_velocity = joint_velocity[:5]
    arm_velocity = np.clip(arm_velocity, -np.pi/2, np.pi/2)
    wheel_velocity = joint_velocity[5:9]
    chassis_twist = r/4 * np.array([
        [-1/(L+w), 1/(L+w), 1/(L+w), -1/(L+w)],
        [1, 1, 1, 1],
        [-1, 1, -1, 1],  
    ]) @ (wheel_velocity*dt)
    yaw_rate = chassis_twist[0]
    # Update arm state
    arm_state_next = arm_state + arm_velocity * dt
    # Update wheel state
    wheel_state_next = wheel_state + wheel_velocity * dt
    # Update chassis state
    if abs(yaw_rate) < 1e-6:
        delta_chassis_state = np.array([
            0,
            chassis_twist[1],
            chassis_twist[2]
        ])
    else:
        delta_chassis_state = np.array([
            yaw_rate,
            (chassis_twist[1]*np.sin(yaw_rate) + chassis_twist[2]*(np.cos(yaw_rate)-1))/yaw_rate,
            (chassis_twist[2]*np.sin(yaw_rate) + chassis_twist[1]*(1-np.cos(yaw_rate)))/yaw_rate
        ])
    theta = chassis_state[0]
    delta_chassis_state_s = np.array([[1, 0, 0],
                                     [0, np.cos(theta), -np.sin(theta)],
                                     [0, np.sin(theta), np.cos(theta)]])
    chassis_state_next = chassis_state + delta_chassis_state_s @ delta_chassis_state
    # print ("chassis_state_next: ", chassis_state_next)
    # print(np.shape(chassis_state_next))
    # print ("arm_state_next: ", arm_state_next)
    # print ("wheel_state_next: ", wheel_state_next)
    return np.concatenate((chassis_state_next, arm_state_next, wheel_state_next))
if __name__ == "__main__":
    #Test NextState function
    current_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    joint_velocity = np.array([0.5, -1, 1, 0.5, 0.5, 30, 30, 10, 10])
    dt = 0.01
    gripper_state = 0 #gripper state, 1 for grasping, 0 for not grasping
    joint_limits = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])
    state = []
    for i in range(100):
        
        next_state = NextState(current_state, joint_velocity, dt, joint_limits)
        next_state = np.concatenate((next_state,[gripper_state]))
        current_state = next_state[:12]
        state.append(next_state)
    # print(next_state)
    np.savetxt("state.csv", state, delimiter=",")
    print(np.shape(state))

# vedio link
# https://drive.google.com/file/d/1lbgDFJGrlnpH5PzflTsHDh8_fr7KP37T/view?usp=sharing