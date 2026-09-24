import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utility as ram

def animate(robot):

    end_eff_pos_local = robot.params.end_eff_pos_local
    end_eff_quat_local = robot.params.end_eff_quat_local
    R_end_eff = ram.quat2rotation(end_eff_quat_local )

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot Animation')
    ax.grid(True)

    # Set axis limits as [-1, 1, -1, 1, -1, 1]
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Hold the previous point
    prev = np.array([0, 0, 0])
    # Plot the robot links
    for i, body in robot.body.items():
        # Get the current global position
        curr = robot.body[i].H_global[:3, 3]  # Extract translation component from H_global

        # Plot the link as a line
        ax.plot([prev[0], curr[0]],[prev[1], curr[1]],[prev[2], curr[2]],linewidth=5, color='magenta')

        # Update the previous point
        prev = curr
        # Handle the end-effector if this is the last body
        if i == len(robot.body):
            # Calculate the global position of the end-effector
            end_eff_pos = robot.body[i].H_global @ np.append(end_eff_pos_local, 1)  # Homogeneous coordinates
            end_eff_pos = end_eff_pos[:3]  # Extract translation\
            end_eff_rot = robot.body[i].H_global[:3, :3] @ R_end_eff

            curr = end_eff_pos
            # Plot the link to the end-effector
            ax.plot([prev[0], curr[0]],[prev[1], curr[1]],[prev[2], curr[2]],linewidth=5, color='magenta')


    # Set equal aspect ratio for proper visualization
    ax.set_box_aspect([1, 1, 1])  # Equal scaling for x, y, z axes


    #print(ax.elev)  # Current elevation angle
    #print(ax.azim)  # Current azimuth angle
    ax.view_init(elev=18, azim=33)

    origin = end_eff_pos #np.array([0,0,0])
    R = end_eff_rot  #np.eye(3)
    arrow_length = 0.2
    dirn_x = np.array([1, 0, 0]); dirn_x = R.dot(dirn_x);
    dirn_y = np.array([0, 1, 0]); dirn_y = R.dot(dirn_y);
    dirn_z = np.array([0, 0, 1]); dirn_z = R.dot(dirn_z);
    ax.quiver(origin[0],origin[1],origin[2],dirn_x[0],dirn_x[1],dirn_x[2],
             length=arrow_length, arrow_length_ratio = .1,normalize=True,color='red')
    ax.quiver(origin[0],origin[1],origin[2],dirn_y[0],dirn_y[1],dirn_y[2],
             length=arrow_length, arrow_length_ratio = .1,normalize=True,color='green')
    ax.quiver(origin[0],origin[1],origin[2],dirn_z[0],dirn_z[1],dirn_z[2],
             length=arrow_length, arrow_length_ratio = .1,normalize=True,color='blue')
    ax.scatter(origin[0], origin[1], origin[2], color='orange', s=50, label='EndEff')  # `s` sets the size of the marker


    # Display the plot
    # plt.show(block=False)
    # plt.pause(5)
    # plt.close()
    #set view
    plt.show()
