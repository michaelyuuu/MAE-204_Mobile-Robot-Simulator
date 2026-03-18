# MAE-204 Mobile Robot Simulator

This repository contains a Python simulation of a mobile manipulator performing a pick-and-place task. The project generates a reference end-effector trajectory, tracks it with task-space feedback control, propagates the robot state forward in time, and saves the resulting motion history for visualization and analysis.

The code is organized around three main components:

- `TrajectoryGenerator.py` builds the reference pick-and-place trajectory.
- `FeedbackControl.py` computes the commanded chassis and arm velocities.
- `Nextstate.py` advances the robot configuration one time step.

`main.py` ties everything together and plots tracking error and manipulability over time.

## Project overview

The simulator models a mobile manipulator with:

- a mobile chassis,
- a 5-DOF robot arm,
- 4 wheels,
- a gripper state attached to the trajectory/state logs.

The task is to move a cube from an initial pose to a final pose using a multi-stage end-effector trajectory:

1. Move from the initial end-effector pose to an initial standoff pose.
2. Descend to the grasp pose.
3. Close the gripper.
4. Lift back to standoff.
5. Move to the goal standoff pose.
6. Descend to the release pose.
7. Open the gripper and retreat.

The controller uses a feedforward twist plus PI feedback in task space and maps that commanded twist to wheel and arm joint velocities using the combined end-effector Jacobian pseudoinverse.

## Repository contents

### Source files

- `main.py` — runs the full closed-loop simulation and displays plots.
- `TrajectoryGenerator.py` — generates the reference SE(3) trajectory and writes `trajectory.csv`.
- `FeedbackControl.py` — computes end-effector twist tracking commands and actuator velocities.
- `Nextstate.py` — propagates the robot configuration forward one time step and can generate `state.csv` in standalone mode.

### Generated data

- `trajectory.csv` — reference end-effector trajectory.
- `state.csv` — standalone state propagation output from `Nextstate.py`.
- `overall_trajectory.csv` — full simulation result from `main.py`.

### Media included in the repository

- `arm trajecotry.avi`
- `mobile robot control.avi`
- `Best plot.png`
- `Best plot_ manipulability factors.png`
- `oscillation plot.png`
- `Oscilation_ manipulability factors.png`
- `new_task.png`
- `new_task_manipulability_factor.png`
- `MAE204_WI26_Final_Project (1).pdf`

These appear to be sample outputs and project deliverables associated with the simulator.

## Dependencies

The code uses the following Python packages:

- `numpy`
- `matplotlib`
- `modern_robotics`

Install them in your environment before running the project.

## Setup

From the repository root, install the required packages in your Python environment:

```bash
pip install numpy matplotlib modern_robotics
```

## How to run

### Run the full simulator

```bash
python main.py
```

This will:

- generate the reference trajectory,
- simulate closed-loop tracking,
- save the state history to `overall_trajectory.csv`,
- display plots of tracking error and manipulability.

### Run the trajectory generator only

```bash
python TrajectoryGenerator.py
```

This writes `trajectory.csv`.

### Run the state propagator test only

```bash
python Nextstate.py
```

This writes `state.csv`.

## Simulation details

Important parameters defined in the code include:

- time step: `dt = 0.01`
- proportional gain: `Kp = 50 I`
- integral gain: `Ki = 20 I`
- wheel radius: `r = 0.0475 m`
- chassis geometry: `L = 0.47 / 2`, `w = 0.15`
- trajectory duration per screw segment: `Tf = 15 s`
- interpolation method: quintic (`method = 5`)

The initial robot configuration in `main.py` is stored as:

`[theta, x, y, arm1, arm2, arm3, arm4, arm5, wheel1, wheel2, wheel3, wheel4]`

The cube initial and final poses are also defined directly in `main.py` and `TrajectoryGenerator.py`.

## Output formats

### `trajectory.csv`

Each row contains 13 values:

`[r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper]`

Where:

- the first 9 entries are the rotation matrix flattened row-wise,
- the next 3 entries are the end-effector position,
- the final entry is the gripper state (`0` open, `1` closed).

### `state.csv` and `overall_trajectory.csv`

Each row contains 13 values:

`[theta, x, y, arm1, arm2, arm3, arm4, arm5, wheel1, wheel2, wheel3, wheel4, gripper]`

Where:

- `theta, x, y` describe the chassis configuration,
- `arm1` to `arm5` are the arm joint angles,
- `wheel1` to `wheel4` are the wheel angles,
- `gripper` is the gripper state.

## Control and kinematics summary

### Trajectory generation

`TrajectoryGenerator.py` creates a sequence of SE(3) waypoints using `modern_robotics.ScrewTrajectory` between the major pick-and-place poses.

### Feedback control

`FeedbackControl.py` computes:

- the feedforward twist from consecutive reference poses,
- the configuration error twist,
- the PI-corrected commanded end-effector twist,
- the wheel and joint rates from the full Jacobian pseudoinverse.

### State update

`Nextstate.py` integrates:

- arm joint motion,
- wheel motion,
- chassis body twist projected into the space frame.

## Notes

- Running `main.py` displays plots interactively using `matplotlib`.
- The output CSV files are overwritten when the corresponding scripts are run again.
- The repository already includes example plots and videos that can be used to review the robot behavior without rerunning the simulation immediately.

## Possible future improvements

Some implementation details you may want to refine if you continue developing the project:

- add a `requirements.txt` file,
- save plots automatically from `main.py`,
- clean up joint-limit handling in `FeedbackControl.py`,
- parameterize the task setup from a config file instead of hard-coding it.

## Summary

This project provides a compact end-to-end example of mobile manipulation:

- reference trajectory generation,
- task-space feedback control,
- kinematic state propagation,
- logging and visualization.

It is a solid base for experimentation with trajectory tracking, manipulability analysis, and controller tuning for a mobile manipulator.
