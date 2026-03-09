import modern_robotics as mr
import numpy as np
import time
block_init = np.array([1, 0 ,0])
block_end = np.array([0, -1, -np.pi/2])
standoff = 0.2 # distance above the block[m]
Tf = 5 # total time of the trajectory [s]
N = 100 # number of trajectory reference configurations
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
def TrajectoryGenerator(T_init, Cube_init, Cube_end , T_ce_grasp, T_ce_standoff, k):
    Traj = []
    T_grasp = Cube_init @ T_ce_grasp
    T_standoff_init = Cube_init @ T_ce_standoff
    T_standoff_end = Cube_end @ T_ce_standoff
    T_release = Cube_end @ T_ce_grasp
    M = mr.ScrewTrajectory(T_init, T_standoff_init, Tf, N, method)
    grasp_state = np.array([0]) #grasping state, 1 for grasping, 0 for not grasping
    M.extend(mr.ScrewTrajectory(T_standoff_init, T_grasp, Tf, N, method))
    grasp_state = np.append(grasp_state, 0) #grasping state, 1 for grasping, 0 for not grasping
    M.extend(mr.ScrewTrajectory(T_grasp, T_grasp, Tf, N, method))
    grasp_state = np.append(grasp_state, 1) #grasping state, 1 for grasping, 0 for not grasping
    M.extend(mr.ScrewTrajectory(T_grasp, T_standoff_init, Tf, N, method))
    grasp_state = np.append(grasp_state, 1) #grasping state, 1 for grasping, 0 for not grasping
    M.extend(mr.ScrewTrajectory(T_standoff_init, T_standoff_end, Tf, N, method))
    grasp_state = np.append(grasp_state, 1) #grasping state, 1 for grasping, 0 for not grasping
    M.extend(mr.ScrewTrajectory(T_standoff_end, T_release, Tf, N, method))
    grasp_state = np.append(grasp_state, 1) #grasping state, 1 for grasping, 0 for not grasping
    M.extend(mr.ScrewTrajectory(T_release, T_release, Tf, N, method))
    grasp_state = np.append(grasp_state, 0) #grasping state, 1 for grasping, 0 for not grasping
    M.extend(mr.ScrewTrajectory(T_release,T_init, Tf, N, method))
    grasp_state = np.append(grasp_state, 0) #grasping state, 1 for grasping, 0 for not grasping
    print (grasp_state)
    print (np.shape(M))
    print (T_release)
    # grasp = 0 #grasping state, 1 for grasping, 0 for not grasping
    #flatten the trajectory list and convert to numpy array
    cnt = 0
    for i in range(np.shape(M)[0]):
        if i%(N)==0 or i == 0:
            grasp = grasp_state[cnt]
            cnt += 1  
        Rot = np.array(M[i][:3,:3]).flatten()
        pos = np.array(M[i][:3,3]).flatten()
        Traj.append(np.concatenate((Rot, pos, [grasp])))
    print("Trajectory from initial pose to standoff pose generated.")
    print(Traj[-1])
    #save the trajectory to a csv file
    np.savetxt("trajectory.csv", Traj, delimiter=",")
    return Traj
TrajectoryGenerator(pose_init, cube_init_T, cube_end_T, T_ce_grasp, T_ce_standoff, k)


# Vedio link
# https://drive.google.com/file/d/1aPeGt26Qy-GkRPyg0EzCHgnP78hsPtz3/view?usp=sharing