import numpy as np
# Nextstate.py
# Computes the next state of the robot given the current state and control inputs.
# Inputs:
# - current_state: current state of the robot (12x1 vector: [theta, x, y, arm joints (5), wheel angles (4)])
# - joint_velocity: commanded joint velocities (9x1 vector: [arm joints (5), wheel angles (4)])
# - dt: time step for integration
# - max_vel: maximum allowed joint velocity for safety
# Outputs:
# - next_state: next state of the robot after applying the control inputs for one time step (12x1 vector)
# Run this file to test the NextState function with a sample input and save the resulting states to "state.csv" for analysis.
# input csv file to simulator Scene 6
def NextState(current_state, joint_velocity, dt, max_vel):
    r = 0.0475 # wheel radius
    L = 0.47/2 # distance between the center of the robot and the wheels (forward/backward)
    w = 0.15   # distance between the left and right wheels
    
    chassis_state = current_state[:3]
    arm_state = current_state[3:8]
    wheel_state = current_state[8:12]
    
    # Add small noise to avoid singularities on joint 3 and 4 when perfectly straight
    if np.linalg.norm(arm_state[2]) < 1e-3 and np.linalg.norm(arm_state[3]) < 1e-3:
        arm_state[2] += 0.001 
        arm_state[3] += 0.001 
        
    arm_velocity = joint_velocity[:5]
    arm_velocity = np.clip(arm_velocity, -max_vel, max_vel)
    wheel_velocity = joint_velocity[5:9]
    
    # Compute chassis twist in the body frame
    chassis_twist = (r/4) * np.array([
        [-1/(L+w), 1/(L+w), 1/(L+w), -1/(L+w)],
        [1, 1, 1, 1],
        [-1, 1, -1, 1],  
    ]) @ (wheel_velocity * dt)
    
    yaw_rate = chassis_twist[0]
    
    # Update arm and wheel states using simple Euler integration
    arm_state_next = arm_state + arm_velocity * dt
    wheel_state_next = wheel_state + wheel_velocity * dt
    if abs(yaw_rate) < 1e-6:
        # Straight line approximation if yaw rate is extremely small
        delta_chassis_state = np.array([
            yaw_rate,
            chassis_twist[1],
            chassis_twist[2]
        ])
    else:
        # Exact circular arc integration
        delta_chassis_state = np.array([
            yaw_rate,
            (chassis_twist[1]*np.sin(yaw_rate) + chassis_twist[2]*(np.cos(yaw_rate)-1))/yaw_rate,
            (chassis_twist[2]*np.sin(yaw_rate) + chassis_twist[1]*(1-np.cos(yaw_rate)))/yaw_rate
        ])
        
    theta = chassis_state[0]
    # Transform delta from body frame to space frame
    delta_chassis_state_s = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])
    
    chassis_state_next = chassis_state + delta_chassis_state_s @ delta_chassis_state
    
    return np.concatenate((chassis_state_next, arm_state_next, wheel_state_next))
if __name__ == "__main__":
#Test NextState function
    current_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    joint_velocity = np.array([0.5, -1, 1, 0.5, 0.5, 30, 30, 10, 10])
    dt = 0.01
    gripper_state = 0 #gripper state, 1 for grasping, 0 for not grasping
    max_joint_vel = 5.0
    state = []
    for i in range(100):
        
        next_state = NextState(current_state, joint_velocity, dt, max_joint_vel)
        next_state = np.concatenate((next_state,[gripper_state]))
        current_state = next_state[:12]
        state.append(next_state)
    # print(next_state)
    np.savetxt("state.csv", state, delimiter=",")
    print(np.shape(state))

# vedio link
# https://drive.google.com/file/d/1lbgDFJGrlnpH5PzflTsHDh8_fr7KP37T/view?usp=sharing