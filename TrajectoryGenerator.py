import modern_robotics as mr
import numpy as np
# TrajectoryGenerator.py
# Generates a trajectory of end-effector configurations for the robot to follow.
# Inputs:
# - T_init: initial end-effector configuration (SE(3))
# - Cube_init: initial cube configuration (SE(3))
# - Cube_end: desired cube configuration (SE(3))
# - T_ce_grasp: end-effector configuration relative to cube for grasping (SE(3))
# - T_ce_standoff: end-effector configuration relative to cube for standoff (SE(3))
# - k: number of trajectory reference configurations per 0.01 second
# Outputs:
# - Traj: list of end-effector configurations (SE(3)) with gripper state for each time step
# Run this file to test the TrajectoryGenerator function with sample inputs and save the resulting trajectory to "trajectory.csv" for analysis.
# Input csv file to simulator Scene 8
def TrajectoryGenerator(T_init, Cube_init, Cube_end, T_ce_grasp, T_ce_standoff, k):
    # Time step
    dt = 0.01 / k
    Traj = []
    
    # Calculate the 4 key configurations based on the cube's initial and final poses
    T_grasp_init = Cube_init @ T_ce_grasp
    T_standoff_init = Cube_init @ T_ce_standoff
    T_grasp_end = Cube_end @ T_ce_grasp
    T_standoff_end = Cube_end @ T_ce_standoff
    
    # Helper function to generate trajectory segments and accurately append the gripper state
    def append_segment(start_T, end_T, Tf, grasp_val):
        # Calculate number of steps based on the time for this specific segment
        N = max(int(Tf / dt), 1) 
        # 5 represents quintic time scaling
        segment = mr.ScrewTrajectory(start_T, end_T, Tf, N, 5) 
        for T in segment:
            Rot = np.array(T[:3, :3]).flatten()
            pos = np.array(T[:3, 3]).flatten()
            Traj.append(np.concatenate((Rot, pos, [grasp_val])))

    # Generate the 8 segments with individual appropriate time durations
    append_segment(T_init, T_standoff_init, 5.0, 0)          # 1. Move to standoff above init (open)
    append_segment(T_standoff_init, T_grasp_init, 2.0, 0)    # 2. Move down to grasp (open)
    append_segment(T_grasp_init, T_grasp_init, 0.63, 1)      # 3. Close gripper (wait ~63 steps)
    append_segment(T_grasp_init, T_standoff_init, 2.0, 1)    # 4. Move up back to standoff (closed)
    append_segment(T_standoff_init, T_standoff_end, 5.0, 1)  # 5. Move to target standoff (closed)
    append_segment(T_standoff_end, T_grasp_end, 2.0, 1)      # 6. Move down to target (closed)
    append_segment(T_grasp_end, T_grasp_end, 0.63, 0)        # 7. Open gripper (wait ~63 steps)
    append_segment(T_grasp_end, T_standoff_end, 2.0, 0)      # 8. Move up to target standoff (open)
    
    # Save standalone trajectory for testing purposes
    np.savetxt("trajectory.csv", Traj, delimiter=",")
    return Traj
if __name__ == "__main__":
    block_init = np.array([1, 0 ,0])
    block_end = np.array([0, -1, -np.pi/2])
    standoff = 0.2 # distance above the block[m]
    Tf = 15 # total time of the trajectory [s]
    k=1 # number of trajectory reference configurations per 0.01 second
    N = Tf*k/0.01 # number of trajectory reference configurations
    method = 5 # interpolation method 'cubic' 3  or 'quintic' 5
    z = 0.025 # height of the block
    pose_init = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],  
                            [0, 0, 1, 0.5],
                            [0, 0, 0, 1]])
    pose_init_standoff = np.array([[np.cos(block_init[2]), -np.sin(block_init[2]), 0, block_init[0]],
                        [np.sin(block_init[2]), np.cos(block_init[2]), 0, block_init[1]],
                        [0, 0, 1, standoff],
                        [0, 0, 0, 1]])
    cube_init_T = np.array([[np.cos(block_init[2]), -np.sin(block_init[2]), 0, block_init[0]],
                        [np.sin(block_init[2]), np.cos(block_init[2]), 0, block_init[1]],
                        [0, 0, 1, z],
                        [0, 0, 0, 1]])
    cube_end_T = np.array([[np.cos(block_end[2]), -np.sin(block_end[2]), 0, block_end[0]],
                        [np.sin(block_end[2]), np.cos(block_end[2]), 0, block_end[1]],
                        [0, 0, 1, z],
                        [0, 0, 0, 1]])
    pose_end_standoff = np.array([[np.cos(block_end[2]), -np.sin(block_end[2]), 0, block_end[0]],
                        [np.sin(block_end[2]), np.cos(block_end[2]), 0, block_end[1]],
                        [0, 0, 1, standoff],
                        [0, 0, 0, 1]])
    T_ce_grasp = np.array([[0, 0, 1, 0.025],
                            [0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 0, 1]])
    T_ce_standoff = np.array([[0, 0, 1, 0.025],
                            [0, 1, 0, 0],
                            [-1, 0, 0, standoff],
                            [0, 0, 0, 1]])
    k = 1 #numbe of trajectory reference configurations per 0.01 second

    grasp_state = np.zeros(10) #grasping state, 1 for grasping, 0 for not grasping
    TrajectoryGenerator(pose_init, cube_init_T, cube_end_T, T_ce_grasp, T_ce_standoff, k)