from robot_data import robot
import numpy as np
from forward_kinematics import forward_kinematics
import utility as ram
from animate import animate

#print(robot.body[1].name)

q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
sol,robot = forward_kinematics(q)

end_eff_pos = sol.end_eff_pos
end_eff_rot = sol.end_eff_rot

print("Position of the end-effector:")
print(end_eff_pos)
print('**********')

print("Rotation matrix of the end-effector:")
print(end_eff_rot)
print('**********')

print("Euler angles of the end-effector:")
end_eff_euler = ram.rotation2euler(end_eff_rot)
print(end_eff_euler)
print('**********')

end_eff_quat = ram.rotation2quat(end_eff_rot)
print("Quaternion of the end-effector:")
print(end_eff_quat)

#print(sol)
animate(robot)
