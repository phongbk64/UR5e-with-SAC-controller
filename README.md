# UR5e-with-SAC-controller
This project presents the development of a SAC based controller for the point tracking control of a 6DOF UR5E robotic manipulator
## Project structure

- `MATLAB/`: Simulink model, SAC agent initialization, training, evaluation, and trajectory export.
- `MUJOCO/universal_robots_ur5e/`: UR5e MuJoCo XML model and meshes.
- `MUJOCO/ur5_forward_kinematics/`: Python programs for replaying and recording the MATLAB joint trajectory.
## Train the SAC agent in MATLAB
Open MATLAB and make the `MATLAB` directory the current folder:
```matlab
cd('C:\path\to\UR5e-with-SAC-controller\MATLAB')
```
Start a new 1,000-episode training run with one command:
```matlab
train_UR5_SAC(1000)
```
This command creates a new agent, trains it from scratch, evaluates the saved
checkpoints without exploration, and selects a checkpoint that satisfies both
of the following tolerances:
- Position error: no more than `0.5 cm`
- End-effector direction error: no more than `1 degree`
Starting a new training run deletes the existing `MATLAB/final_training`
directory. Copy any result that must be retained before running the command.
The training outputs are written to `MATLAB/final_training/`, including:
- `trained_agent_final.mat`: selected trained agent
- `training_final.csv`: reward and episode history
- `training_curve_final.png`: training curve
- `checkpoint_evaluation.csv`: deterministic checkpoint evaluation
- `evaluation_final.csv`: result of the selected checkpoint
The training time depends on the computer. A 1,000-episode run can take several
hours, including checkpoint evaluation.
## Run a trained agent
After `trained_agent_final.mat` has been created, run:
```matlab
run_UR5_SAC
```
The program runs the selected agent without exploration and produces:
- Joint variables over time
- End-effector position and direction over time
- Position and direction tracking errors
- `joint_trajectory.xlsx`, containing time and the six joint angles for MuJoCo
- `execution_results.mat`, containing the complete execution result
All output files are stored in `MATLAB/final_training/`.
First run `run_UR5_SAC` in MATLAB so that `joint_trajectory.xlsx` exists. From
`MUJOCO/ur5_forward_kinematics`, run:

```powershell
.\.venv\Scripts\python.exe .\run_sac_trajectory.py --xlsx "..\..\MATLAB\final_training\joint_trajectory.xlsx"
```
## Replay the MATLAB trajectory in MuJoCo
The MuJoCo interface highlights the start position, target position, target
direction, current end-effector pose, and travelled path. Use the **Start**
button to replay from the beginning, **Pause / Resume** to pause, and **Close**
to exit. The Space key also pauses the replay, and `R` restarts it.
When a different MATLAB goal is used, update `GOAL_POSITION` and
`GOAL_DIRECTION` near the top of `run_sac_trajectory.py` so that the MuJoCo
highlight matches the MATLAB model.
