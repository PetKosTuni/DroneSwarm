import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import signal
import json
from scipy.stats import multivariate_normal as mvn

def plot_trajectory_snapshot(ax, agent_trajectories, slice_left, slice_right, pdf_values, grids_x, grids_y, L_list, dt, colors):
    ax.set_aspect('equal')
    ax.set_xlim(0, L_list[0])
    ax.set_ylim(0, L_list[1])
    ax.set_xlabel('x [m]', fontsize = 30)
    ax.set_ylabel('y [m]', fontsize = 30)
    ax.set_title('Drone trajectories at t = ' + str(slice_right * dt) + 's')
    ax.contourf(grids_x, grids_y, pdf_values, cmap='Reds')
    i = 0

    for agent in agent_trajectories.keys():
        # Check if trajectory data exists before plotting
        if agent_trajectories[agent][0] and agent_trajectories[agent][1]:  # Check if lists are not empty
            ax.plot(agent_trajectories[agent][0][slice_right], agent_trajectories[agent][1][slice_right], linestyle='', marker='o', markersize=15, color=colors[i], alpha=1.0, label=f'Drone {i+1} end')
            if agent != 'cf3' or slice_left == 0:

                # remove following line if the legend is too crowded
                ax.plot(agent_trajectories[agent][0][slice_left], agent_trajectories[agent][1][slice_left], linestyle='', marker='o', markersize=10, color=colors[i], alpha=1.0, label=f'Drone {i+1} start')

                ax.plot(agent_trajectories[agent][0][slice_left:slice_right], agent_trajectories[agent][1][slice_left:slice_right], linestyle='--', marker='', color=colors[i], alpha=1, label=f'Drone {i+1} trajectory')
        else:
            print(f"Warning: No trajectory data for drone {agent}")
        i += 1

    ax.legend(loc='best', fontsize=14)

def pdf(x, clusters):

    # Define the required variables
    weights = []
    means = []
    covariances = []
    points_in_cluster = []
    total_points = 0

    for cluster in clusters:
        # We save the amount of points in each cluster for weight calculation (more points = bigger weight)
        points_in_cluster.append(len(cluster))
        # Total points for relative weights
        total_points += len(cluster)

        # Implementation 1:
        # We separate x and y coordinates from each point for calculating means.
        # There might be a way to do this better, not sure.
        x_coords = []
        y_coords = []

        for point in cluster:
            x_coords.append(point[0])
            y_coords.append(point[1])

        mean_x = np.mean(x_coords)
        mean_y = np.mean(y_coords)

        # Save mean values for calculating the return value.
        means.append(np.array([mean_x, mean_y]))

        # Save covariances to calculate the return value.
        cov = np.cov(x_coords, y_coords)
        covariances.append(cov)

        # Implementation 2:
        # This is the better way, if the result is not 2 coordinates, change axis to 0.
        #means.append(np.mean(cluster, axis=1))

        # If this does not work, use all of the stuff from the worse implementation 1.
        #covariances.append(np.cov(cluster))

    for amount in points_in_cluster:
        weights.append(amount / total_points)
    
    returnable = 0

    for index in range(len(weights)):
        # Test print to view the data values.
        # print(str(weights[index]) + " " + str(x) + " " + str(means[index]) + " " + str(covariances[index]))
        returnable += weights[index] * mvn.pdf(x, means[index], covariances[index], allow_singular=True)
    
    return returnable

def main(args=None):
    plt.rcParams['xtick.labelsize']=15
    plt.rcParams['ytick.labelsize']=15
    plt.rcParams['axes.titlesize']=24
    plt.rcParams['axes.labelsize']=24
    plt.rcParams['legend.fontsize']=20
    plt.rcParams['legend.framealpha']=0.6
    plt.rcParams['lines.linewidth']=2.8
    plt.rcParams['figure.titlesize']=18

    colors = ['C7', 'C4', 'C2', 'C2', 'C5', 'C6', 'C7', 'C8', 'C1']

    # Change to proper filepath
    with open('ergodic_test_record_jun19_video.json', 'r') as file:
        data = json.load(file)
    with open('points_jun19_video.json', 'r') as file2:
        points = json.load(file2)

    agent_trajectories = data["trajectories"]
    control_inputs = data["control_inputs"]
    ergodic_metric = data["ergodic_metrics"]
    pdf_values = data["pdf_values"]
    phi_recon = np.array(data["phi_recon"])
    grids_x = np.array(data["grids_x"])
    grids_y = np.array(data["grids_y"])
    ks = data["ks"]
    h_funcs = data["h_funcs"]
    L_list = data["parameters"]["L_list"]
    num_agents = data["parameters"]["num_agents"]
    dt = data["parameters"]["rate"]
    umax = data["parameters"]["umax"]
    safety_distance = data["parameters"]["safety_distance"]
    gamma = data["parameters"]["gamma"]
    h_pow = data["parameters"]["h_pow"]
    threat_radius = 0.6
    wall_distance = 0.17

    metric_logs = ergodic_metric[0:3*len(control_inputs["cf3"][0]):3] + ergodic_metric[3*len(control_inputs["cf3"][0])::2]
    max_time = 120
    locs = np.arange(0, max_time + 10, step=10)

    # Plot the ergodic metric for all agents
    plt.figure(figsize=(10, 6))
    plt.plot(metric_logs, label=f'Ergodic Metric over time')
    plt.xticks(locs / dt, [x for x in locs])
    plt.xlabel('t [s]')
    plt.ylabel('Ergodic Metric Ratio')
    plt.title('Ergodic Metric')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the control inputs for all agents
    fig, axes = plt.subplots(num_agents, 1, figsize=(10, 4*num_agents), sharex=True)

    i = 0
    for agent in control_inputs.keys():

        drone_index = str(agent).replace('cf', '')
        if drone_index == '3':
            drone_index = 2
        elif drone_index == '5':
            drone_index = 3

        if num_agents == 1:
            axes.plot(control_inputs[agent][0], label='u_x')
            axes.plot(control_inputs[agent][1], label='u_y')
            axes.set_ylabel(f'Drone {drone_index} control input')
            axes.legend()
            axes.grid(True)
            axes.set_xticks(locs / dt, [x for x in locs])
            axes.set_xlabel('t [s]')
        else:
            axes[i].plot(control_inputs[agent][0], label='u_x')
            axes[i].plot(control_inputs[agent][1], label='u_y')
            axes[i].set_ylabel(f'Drone {drone_index} control input')
            axes[i].legend()
            axes[i].grid(True)
            axes[i].set_xticks(locs / dt, [x for x in locs])
            axes[-1].set_xlabel('t [s]')
            i += 1

    plt.suptitle('Control Inputs Over Time')
    plt.tight_layout()
    plt.show()

    # Plot the control input magnitudes for all agents
    fig, axes = plt.subplots(num_agents, 1, figsize=(10, 4*num_agents), sharex=True)

    i = 0
    for agent in control_inputs.keys():

        drone_index = str(agent).replace('cf', '')
        if drone_index == '3':
            drone_index = 2
        elif drone_index == '5':
            drone_index = 3

        string = 'max'
        if num_agents == 1:
            axes.plot(np.linalg.norm(control_inputs[agent], axis = 0))
            axes.set_ylabel(f'||$u_{{{drone_index}}}$|| [m]')
            axes.grid(True)
            axes.set_xticks(locs / dt, [x for x in locs])
            axes.set_xlabel('t [s]')
            axes.axhline(y = umax, color = 'r', linestyle = 'dashed', label=f'$u_{{{string}}}$ = ' + str(umax) + 'm/s')
            axes.axvline(x = 1000, color = 'black', linestyle = 'dashed', label= 'Drone 1 landing at ' + str(1000 * dt) + 's')
            axes.legend()
        else:
            axes[i].plot(np.linalg.norm(control_inputs[agent], axis = 0), color=colors[i])
            axes[i].set_ylabel(f'||$u_{{{drone_index}}}$|| [m]')
            axes[i].grid(True)
            axes[i].set_xticks(locs / dt, [x for x in locs])
            axes[i].axhline(y = umax, color = 'r', linestyle = 'dashed', label=f'$u_{{{string}}}$ = ' + str(umax) + 'm/s')
            if i == 1:
                axes[i].axvline(x = 1150, color = 'black', linestyle = 'dashed', label= 'Drone 2 landing at ' + str(1150 * dt) + 's')
            axes[i].legend()

            axes[-1].set_xlabel('t [s]')
            i += 1

    plt.suptitle('Control Input Magnitude Over Time')
    plt.tight_layout()
    plt.show()

    # Plot xyz-coordinates for all agents
    fig, axes = plt.subplots(1, 1, figsize=(10, 4), sharex=True)

    for agent in agent_trajectories.keys():

        drone_index = str(agent).replace('cf', '')
        if drone_index == '3':
            drone_index = 2
        elif drone_index == '5':
            drone_index = 3

        """axes[0].plot(agent_trajectories[agent][0], label=f'Drone {drone_index}')
        axes[0].set_ylabel('X-coordinate')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(agent_trajectories[agent][1], label=f'Drone {drone_index}')
        axes[1].set_ylabel('Y-coordinate')
        axes[1].legend()
        axes[1].grid(True)"""

        axes.plot(agent_trajectories[agent][2], label=f'Drone {drone_index}')
        axes.set_ylabel('Altitude [m]')
        axes.legend()
        axes.grid(True)

        axes.set_xticks(locs / dt, [x for x in locs])
        axes.set_xlabel('t [s]')

    plt.suptitle('Drone z-coordinates over time')
    plt.tight_layout()
    plt.show()

    # Plot XYZ-distances between all agents
    fig, axes = plt.subplots(3, 1, figsize=(10, 4*num_agents), sharex=True)

    remove_from_comparison = []
    for agent in agent_trajectories.keys():

        for other_agents in agent_trajectories.keys():

            if agent != other_agents and other_agents not in remove_from_comparison:

                axes[0].plot(np.abs(np.array(agent_trajectories[agent][0]) - np.array(agent_trajectories[other_agents][0])), label=f'||x_{agent} - x_{other_agents}||')
                axes[0].set_ylabel('X-coordinate distances')
                axes[0].legend()
                axes[0].grid(True)

                axes[1].plot(np.abs(np.array(agent_trajectories[agent][1]) - np.array(agent_trajectories[other_agents][1])), label=f'||y_{agent} - y_{other_agents}||')
                axes[1].set_ylabel('Y-coordinate distances')
                axes[1].legend()
                axes[1].grid(True)

                axes[2].plot(np.abs(np.array(agent_trajectories[agent][2]) - np.array(agent_trajectories[other_agents][2])), label=f'||z_{agent} - z_{other_agents}||')
                axes[2].set_ylabel('Z-coordinate distances')
                axes[2].legend()
                axes[2].grid(True)

                axes[-1].set_xticks(locs / dt, [x for x in locs])
                axes[-1].set_xlabel('t [s]')

        remove_from_comparison.append(agent)
    
    plt.suptitle('Plotting of XYZ-distances between all drones')
    plt.tight_layout()
    plt.show()

    # Plot distances between all agents

    if num_agents > 1:
        plots = (num_agents * (num_agents - 1)) // 2
        fig, axes = plt.subplots(plots, 1, figsize=(10, 4*num_agents), sharex=True)

        remove_from_comparison = []
        i = 0
        
        for agent in agent_trajectories.keys():

            drone_index = str(agent).replace('cf', '')
            if drone_index == '3':
                drone_index = 2
            elif drone_index == '5':
                drone_index = 3
    
            for other_agents in agent_trajectories.keys():

                other_index = str(other_agents).replace('cf', '')
                if other_index == '3':
                    other_index = 2
                elif other_index == '5':
                    other_index = 3

                if agent != other_agents and other_agents not in remove_from_comparison:
                    
                    if plots == 1:
                        axes.plot(np.linalg.norm(np.array(agent_trajectories[agent][:-1]) - np.array(agent_trajectories[other_agents][:-1]), axis = 0))
                        axes.set_ylabel(f'||$p_{drone_index} - p_{other_index}$|| [m]')
                        axes.axhline(y = safety_distance, color = 'r', linestyle = 'dashed', label= f'$R_s$ = ' + str(safety_distance) + 'm')
                        axes.axvline(x = 1150, color = 'black', linestyle = 'dashed', label= 'landing at ' + 1150 * dt + 's')
                        axes.legend()
                        axes.grid(True)
                        axes.set_xticks(locs / dt, [x for x in locs])
                        axes.set_xlabel('t [s]')
                    else:
                        axes[i].plot(np.linalg.norm(np.array(agent_trajectories[agent][:-1]) - np.array(agent_trajectories[other_agents][:-1]), axis = 0))
                        axes[i].set_ylabel(f'||$p_{drone_index} - p_{other_index}$|| [m]')
                        axes[i].axhline(y = safety_distance, color = 'r', linestyle = 'dashed', label= f'$R_s$ = ' + str(safety_distance) + 'm')
                        if (other_agents) == "cf3" or agent == "cf3":
                            axes[i].axvline(x = 1150, color = 'black', linestyle = 'dashed', label= 'Drone 2 landing at ' + str(1150 * dt) + 's')
                        axes[i].legend()
                        axes[i].grid(True)
                        axes[-1].set_xticks(locs / dt, [x for x in locs])
                        axes[-1].set_xlabel('t [s]')
                        i += 1
        
            remove_from_comparison.append(agent)
        
        plt.suptitle('Distances between all drones')
        plt.tight_layout()
        plt.show()

    # Visualize the trajectories
    fig, axes = plt.subplots(1, 2, figsize=(15,7), dpi=70, tight_layout=True)

    ax = axes[0]
    ax.set_aspect('equal')
    ax.set_xlim(0, L_list[0])
    ax.set_ylim(0, L_list[1])
    ax.set_xlabel('x [m]', fontsize = 30)
    ax.set_ylabel('y [m]', fontsize = 30)
    ax.set_title('Original PDF')
    ax.contourf(grids_x, grids_y, pdf_values, cmap='Reds')
    i = 0
    for agent in agent_trajectories.keys():
        # Check if trajectory data exists before plotting
        if agent_trajectories[agent][0] and agent_trajectories[agent][1]:  # Check if lists are not empty
            ax.plot(agent_trajectories[agent][0], agent_trajectories[agent][1], linestyle='-', marker='', color=colors[i], alpha=0.4, label=f'Drone {i+1}')
            ax.plot(agent_trajectories[agent][0][0], agent_trajectories[agent][1][0], linestyle='', marker='o', markersize=15, color=colors[i], alpha=1.0, label=f'Drone {i+1} start')

            # remove following line if the legend is too crowded
            ax.plot(agent_trajectories[agent][0][-1], agent_trajectories[agent][1][-1], linestyle='', marker='o', markersize=10, color=colors[i], alpha=1.0, label=f'Drone {i+1} end')

        else:
            print(f"Warning: No trajectory data for drone {agent}")
        i += 1
    ax.legend(loc='best')

    ax = axes[1]
    ax.set_aspect('equal')
    ax.set_xlim(0, L_list[0])
    ax.set_ylim(0, L_list[1])
    ax.set_xlabel('x [m]', fontsize = 30)
    ax.set_ylabel('y [m]', fontsize = 30)
    ax.set_title('Empirical Distribution of the swarm trajectory', fontsize = 18)
    ax.contourf(grids_x, grids_y, phi_recon.reshape(grids_x.shape), cmap='Blues')
    i = 0
    for agent in agent_trajectories.keys():
        # Check if trajectory data exists before plotting
        if agent_trajectories[agent][0] and agent_trajectories[agent][1]:  # Check if lists are not empty
            ax.plot(agent_trajectories[agent][0], agent_trajectories[agent][1], linestyle='-', marker='', color=colors[i], alpha=0.4, label=f'Drone {i+1}')
            ax.plot(agent_trajectories[agent][0][0], agent_trajectories[agent][1][0], linestyle='', marker='o', markersize=15, color=colors[i], alpha=1.0, label=f'Drone {i+1} start')

            # remove following line if the legend is too crowded
            ax.plot(agent_trajectories[agent][0][-1], agent_trajectories[agent][1][-1], linestyle='', marker='o', markersize=10, color=colors[i], alpha=1.0, label=f'Drone {i+1} end')
            
        else:
            print(f"Warning: No trajectory data for drone {agent}")
        i += 1
    ax.legend(loc='best')

    plt.show()

    # Visualize the trajectories
    fig, axes = plt.subplots(1, 3, figsize=(25,9), dpi=70, tight_layout=True)

    ax1 = axes[0]
    plot_trajectory_snapshot(ax1, agent_trajectories, 0, 1150, pdf_values, grids_x, grids_y, L_list, dt, colors)

    ax2 = axes[1]
    plot_trajectory_snapshot(ax2, agent_trajectories, 1150, 2000, pdf_values, grids_x, grids_y, L_list, dt, colors)

    ax3 = axes[2]
    plot_trajectory_snapshot(ax3, agent_trajectories, 2000, 3000, pdf_values, grids_x, grids_y, L_list, dt, colors)

    plt.show()

    # Visualize the pdf

    fig, axes = plt.subplots(1, 1, figsize=(10,7), dpi=70, tight_layout=True)
    
    grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
    pdf_cluster = np.array([pdf(point, points[:-2]) for point in grids]).reshape(grids_x.shape)

    flattened_points = [item for sublist in points for item in sublist]

    ax = axes
    ax.set_aspect('equal')
    ax.set_xlim(0, L_list[0])
    ax.set_ylim(0, L_list[1])
    ax.set_xlabel('x [m]', fontsize = 30)
    ax.set_ylabel('y [m]', fontsize = 30)
    ax.set_title('Target points and the transformed PDF')
    ax.contourf(grids_x, grids_y, pdf_cluster, cmap='Reds')
    x_coords = [p[0] for p in flattened_points[:-10]]
    y_coords = [p[1] for p in flattened_points[:-10]]
    ax.plot(x_coords, y_coords, linestyle='', marker='o', markersize=10, alpha=1.0)

    plt.show()

if __name__ == '__main__':
    main()