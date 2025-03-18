import numpy as np
import rclpy
import rclpy.duration
import rclpy.logging
from rclpy.node import Node
import rclpy.publisher
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, Twist, PointStamped, Point32, PolygonStamped 
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
from motion_capture_tracking_interfaces.msg import NamedPoseArray
import cvxopt
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import signal
import json

from visualization_msgs.msg import MarkerArray, Marker

# Define the probability density function
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

def calculate_QP_solver(H, b, all_h, error, safety_distance, constraint_index, gamma, h_pow):
    dist = np.linalg.norm(error)
    h_func = dist**2 - safety_distance**2
    all_h[constraint_index] = h_func

    H[constraint_index] = -2 * error
    b[constraint_index] = gamma * np.power(h_func, h_pow)
    return H, b, all_h

# Control Barrier Function
def compute_control_input(agent_states, current_agent, desired_velocity, safety_distance, gamma, h_pow, umax, drone_dict, L_list):

    current_state = agent_states[current_agent][:-1]
    threat_radius = 0.6
    wall_safety_distance = 0.15
    
    # Nominal control (constant desired velocity)
    u_nom = desired_velocity
    
    # Set up optimization problem
    Q_mat = 2 * cvxopt.matrix(np.eye(2), tc='d')
    c_mat = -2 * cvxopt.matrix(u_nom, tc='d')
    
    wall_up = np.array([current_state[0], L_list[1]])
    wall_left = np.array([0, current_state[1]])
    wall_right = np.array([L_list[0], current_state[1]])
    wall_down = np.array([current_state[0], 0])
    walls = [wall_up, wall_left, wall_right, wall_down]

    constraint_index = 0
    number_of_threats = 1 #1 if velocity limit
    threat_indexes = []

    wall_dist_temp = []
    for wall in walls:
        dist = current_state - wall
        if threat_radius >= np.linalg.norm(dist):
            wall_dist_temp.append(dist)
            number_of_threats += 1
    
    wall_dist = np.array(wall_dist_temp)

    for agent in agent_states.keys():

        if agent != current_agent and drone_dict[agent] == "active":
            
            if (np.linalg.norm(np.array(agent_states[agent][:-1]) - np.array(current_state))) <= threat_radius:
                number_of_threats += 1
                threat_indexes.append(agent)
    
    # Inequality constraints (CBF)
    H = np.zeros([number_of_threats, 2])
    b = np.zeros([number_of_threats, 1])
    all_h = np.zeros(number_of_threats)

    for agent in agent_states.keys():
        if agent != current_agent and agent in threat_indexes:
            print("CBF activated", agent, current_agent)
            H, b, all_h = calculate_QP_solver(H, b, all_h, np.array(current_state) - np.array(agent_states[agent][:-1]), safety_distance, constraint_index, gamma, h_pow)
            constraint_index += 1

    for j in range(len(wall_dist)):
        print("Wall avoided", current_agent)
        H, b, all_h = calculate_QP_solver(H, b, all_h, wall_dist[j], wall_safety_distance, constraint_index, gamma, h_pow)
        constraint_index += 1

    H[constraint_index] = np.linalg.norm(u_nom)
    b[constraint_index] = umax

    # Convert to cvxopt matrices
    H_mat = cvxopt.matrix(H, tc='d')
    b_mat = cvxopt.matrix(b, tc='d')

    # Solve optimization problem
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(Q_mat, c_mat, H_mat, b_mat, verbose=False)
    
    # Extract solution
    current_input = np.array([sol['x'][0], sol['x'][1]])
        
    return current_input, u_nom, all_h.tolist()

class CFController(Node):

    def __init__(self):
        super().__init__(f'cf_controller_example')
        self.get_logger().info("Initializing controller node...")
        self.declare_parameter('cfname', 'cfX')
        self.cfname = self.get_parameter('cfname').get_parameter_value().string_value

        self.stop = False

        # TODO: Remove
        # self.first_goal = np.array([1, 1])

        self.position = np.array([.0, .0, .0])
        self.other_poses = {}
        self.rate = 0.02
        self.it = 0
        self.true_timestep = 0

        self.has_taken_off = False

        self.CF_DICT = {
            "cf1": "active", 
            #"cf2": "active",
            "cf3": "active",
            #"cf4": "active",
            "cf5": "active",
            #"cf6": "active",
            #"cfa": "active"
        }

        drone_indexes = list(self.CF_DICT.keys())

        # Other drones poses subscriber
        self.other_poses_sub = self.create_subscription(
            NamedPoseArray,
            '/poses',
            self.all_poses_callback,
            qos_profile=qos_profile_sensor_data
        )

        # Just to prevent unused variable warning
        self.other_poses_sub
        self.ros_pubs = {}

        # ----------------------------Publisher setup for RViz--------------------------------------

        self.broken_sensors_pub = self.create_publisher(MarkerArray, 'nebolab/cluster_0/targeted_points', 1)

        self.cf_pos = []

        self.pos_pub_dict = {}

        # -------------------------------------------------------------------------------------------

        # Takeoff and land clients
        # USE IF NEEDED (not necessary)
        self.takeoff_client = self.create_client(Trigger, f"nebolab/allcfs/takeoff")
        self.land_client = self.create_client(Trigger, f"nebolab/allcfs/land")

        self.landing_dict = {}

        if "cf1" in self.CF_DICT:
            pos_pub1 = self.create_publisher(Marker, f'nebolab/cf1/sensing_range', 1)
            land_client_cf1 = self.create_client(Trigger, f"nebolab/cf1/land")
            self.pos_pub_dict["cf1"] = pos_pub1
            self.landing_dict["cf1"] = land_client_cf1
        if "cf2" in self.CF_DICT:
            pos_pub2 = self.create_publisher(Marker, f'nebolab/cf2/sensing_range', 1)
            land_client_cf2 = self.create_client(Trigger, f"nebolab/cf2/land")
            self.pos_pub_dict["cf2"] = pos_pub2
            self.landing_dict["cf2"] = land_client_cf2
        if "cf3" in self.CF_DICT:
            pos_pub3 = self.create_publisher(Marker, f'nebolab/cf3/sensing_range', 1)
            land_client_cf3 = self.create_client(Trigger, f"nebolab/cf3/land")
            self.pos_pub_dict["cf3"] = pos_pub3
            self.landing_dict["cf3"] = land_client_cf3
        if "cf4" in self.CF_DICT:
            pos_pub4 = self.create_publisher(Marker, f'nebolab/cf4/sensing_range', 1)
            land_client_cf4 = self.create_client(Trigger, f"nebolab/cf4/land")
            self.pos_pub_dict["cf4"] = pos_pub4
            self.landing_dict["cf4"] = land_client_cf4
        if "cf5" in self.CF_DICT:
            pos_pub5 = self.create_publisher(Marker, f'nebolab/cf5/sensing_range', 1)
            land_client_cf5 = self.create_client(Trigger, f"nebolab/cf5/land")
            self.pos_pub_dict["cf5"] = pos_pub5
            self.landing_dict["cf5"] = land_client_cf5
        if "cf6" in self.CF_DICT:
            pos_pub6 = self.create_publisher(Marker, f'nebolab/cf6/sensing_range', 1)
            land_client_cf6 = self.create_client(Trigger, f"nebolab/cf6/land")
            self.pos_pub_dict["cf6"] = pos_pub6
            self.landing_dict["cf6"] = land_client_cf6

        self.land = True
        self. h_funcs = []

        self.get_logger().info("Waiting for takeoff & land services to be available...")
        if not self.takeoff_client.wait_for_service(10.0):
            self.get_logger().warn(f"Waited for takeoff service: nebolab/{self.cfname}/takeoff, and could not reach it.")
        if not self.land_client.wait_for_service(10.0):
            self.get_logger().warn(f"Waited for land service: nebolab/{self.cfname}/land, and could not reach it.")
        self.get_logger().info("Done waiting")

        # Publishers (choose one to use)
        for cfname in drone_indexes:
            # create cmd_vel publisher
            self.get_logger().info(f'Creating cmd_vel publisher: /nebolab/{cfname}/cmd_vel')
            self.ros_pubs[cfname] = self.create_publisher(Twist, f'/nebolab/{cfname}/cmd_vel', 1)

        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Area: x: 2.5m, y: 1.4m (*2)

        # Define a 1-by-1 2D search space
        self.L_list = np.array([2.5, 2.5]) # Set total range for x and y axis 2 (-1 to 1)
        
        # Discretize the search space
        grids_x, grids_y = np.meshgrid(np.linspace(0, self.L_list[0], 100), np.linspace(0, self.L_list[1], 100))
        grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
        dx = self.L_list[0] / 99  # We change this part to adjust with new configuration
        dy = self.L_list[1] / 99

        self.grids_x = grids_x
        self.grids_y = grids_y

        # shifted_grids_x, shifted_grids_y = np.meshgrid(np.linspace(-self.L_list[0] / 2, self.L_list[0], 100), np.linspace(0, self.L_list[1], 100))
        # shifted_grids = np.array([shifted_grids_x.ravel(), shifted_grids_y.ravel()]).T

        self.num_agents = len(self.CF_DICT)

        self.trajectories = {}
        self.control_inputs = {}
        for agent in self.CF_DICT.keys():
            self.trajectories[agent] = [[],[],[]]
            self.control_inputs[agent] = [[],[]]

        # Open and read the JSON file
        with open('/home/localadmin/points.json', 'r') as file:
            clusters = json.load(file)

        # Print the data
        print(clusters, "Points forming the PDF")

        # Calculate the PDF values for the grid points
        pdf_values = np.array([pdf(point, clusters) for point in grids])
        self.pdf_values = pdf_values.reshape(grids_x.shape)

        time = self.get_clock().now().to_msg()
        plot_rviz(self.grids_x, self.grids_y, self.pdf_values, self.broken_sensors_pub, time, self.L_list)

        # Configure the index vectors
        self.num_k_per_dim = 20
        self.ks_dim1, self.ks_dim2 = np.meshgrid(np.arange(self.num_k_per_dim), np.arange(self.num_k_per_dim))
        self.ks = np.array([self.ks_dim1.ravel(), self.ks_dim2.ravel()]).T

        # Pre-processing lambda_k and h_k
        self.lamk_list = np.power(1.0 + np.linalg.norm(self.ks, axis=1), -3/2.0)
        self.hk_list = np.zeros(self.ks.shape[0])
        for i, k_vec in enumerate(self.ks):
            fk_vals = np.prod(np.cos(np.pi * k_vec / self.L_list * grids), axis=1)
            hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
            self.hk_list[i] = hk

        # Compute the coefficients for the target distribution
        self.phik_list = np.zeros(self.ks.shape[0])
        self.pdf_vals = pdf(grids, clusters)
        for i, (k_vec, hk) in enumerate(zip(self.ks, self.hk_list)):
            fk_vals = np.prod(np.cos(np.pi * k_vec / self.L_list * grids), axis=1) / hk
            phik = np.sum(fk_vals * self.pdf_vals) * dx * dy
            self.phik_list[i] = phik

        # Specify the dynamic system
        self.umax = 0.30  # desired velocity 0.3 m/s
        self.safety_distance = 0.30  # Safety distance for collision avoidance

        self.gamma = 10  # CBF parameter
        self.h_pow = 3  # CBF parameter

        self.ck_list_updates = np.zeros(self.ks.shape[0])

        self.controller_timer = self.create_timer(self.rate, self.control)

        self.metric_logs = []
        self.grids = grids

    def pose_callback(self, msg):
        # Expose own pose from ros2 topic to self.position
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.position = np.array([x,y,z])
    
    def all_poses_callback(self, msg):
        # Expose other poses from ros2 topic to self.other_poses
        other_poses = {}
        for pose in msg.poses:
                x = pose.pose.position.x
                y = pose.pose.position.y
                z = pose.pose.position.z
                position = [x + self.L_list[0] / 2, y + self.L_list[1] / 2, z]
                other_poses[pose.name] = position
        self.other_poses = other_poses

    def control(self):

        xt = self.other_poses.copy()

        if self.has_taken_off == False:
            self.takeoff_client.call_async(Trigger.Request())
            self.has_taken_off = True

        elif len(xt.keys()) < len(self.CF_DICT.keys()):
            print("camera failure")

            for cfname in self.CF_DICT.keys():

                cmd_vel_msg = Twist()
                cmd_vel_msg.linear.x = 0.0
                cmd_vel_msg.linear.y = 0.0
                cmd_vel_msg.linear.z = 0.0
                self.ros_pubs[cfname].publish(cmd_vel_msg)
        
        elif self.true_timestep > 250:

            if self.it >= 1000 and self.land == True:
                self.landing_dict["cf1"].call_async(Trigger.Request())
                self.land = False
        
            ut_dict = {}
            #print("Positions: " + str(xt))

            for agent, position in xt.items():

                self.trajectories[agent][0].append(position[0])
                self.trajectories[agent][1].append(position[1])
                self.trajectories[agent][2].append(position[2])

                if position[2] > 0.5:

                    #----------------------For RViz: drone trace publisher-----------------------------------------
                    scale_variable_x = 1/(self.L_list[0]/99)
                    scale_variable_y = 1/(self.L_list[1]/99)
                    pointx = round(position[0]*scale_variable_x) # 0-99 position on x axis
                    pointy = round(position[1]*scale_variable_y) # 0-99 position on y axis
                    try:
                        scale = 0.02
                        sensor_range_msg = Marker()
                        sensor_range_msg.header.frame_id = "world"
                        sensor_range_msg.header.stamp = self.get_clock().now().to_msg()
                        sensor_range_msg.id = 100*pointx+pointy+1 # replaces MarkerArray marker with same id
                        sensor_range_msg.type = 1
                        sensor_range_msg.scale.x = scale
                        sensor_range_msg.scale.y = scale
                        sensor_range_msg.scale.z = scale
                        sensor_range_msg.pose.position.x = self.grids_x[0][pointx] - (self.L_list[1] / 2)
                        sensor_range_msg.pose.position.y = self.grids_y[pointy][0] - (self.L_list[0] / 2)
                        sensor_range_msg.pose.position.z = self.pdf_values[pointy, pointx]/10 # point height from pdf

                        sensor_range_msg.color.a = 0.8
                        
                        sensor_range_msg.color.r = 0.0
                        sensor_range_msg.color.b = 0.0

                        if agent == "cf1":
                            sensor_range_msg.color.g = 1.0
                        elif agent == "cf2":
                            sensor_range_msg.color.g = 0.9
                        elif agent == "cf3":
                            sensor_range_msg.color.g = 0.8
                            sensor_range_msg.color.b = 0.3
                        elif agent == "cf4":
                            sensor_range_msg.color.g = 0.7
                        elif agent == "cf5":
                            sensor_range_msg.color.g = 0.6
                        elif agent == "cf6":
                            sensor_range_msg.color.g = 0.5

                        self.pos_pub_dict[agent].publish(sensor_range_msg)

                    except:
                        self.get_logger().info(f"Could not publish sensor range information for drone cf{agent}.")
                    #----------------------------------------------------------------------------------------

                    # step 1: evaluate all the fourier basis functions at the current state
                    # Equation 9
                    # xt_shift = [x + 1 for x in position]
                    fk_xt_all = np.prod(np.cos(np.pi * self.ks / self.L_list * position[:-1]), axis=1) / self.hk_list
                    
                    # step 2: update the coefficients
                    self.ck_list_updates += (fk_xt_all / self.num_agents) * self.rate
                
                    # step 3: compute the derivative of all basis functions at the current state
                    
                    # Equation 9
                    k1 = np.pi * self.ks[:,0] / self.L_list[0]
                    k2 = np.pi * self.ks[:,1] / self.L_list[1]
            
                    # Equation 20
                    dfk_xt_all = np.array([
                        -k1 * np.sin(k1 * position[0]) * np.cos(k2 * position[1]),
                        -k2 * np.cos(k1 * position[0]) * np.sin(k2 * position[1]),
                    ]) / self.hk_list

                    ckt = self.ck_list_updates / (self.it * self.rate + self.rate)
                    
                    # Equation 16
                    Skt = ckt - self.phik_list
                    # Equation 26
                    bt = np.sum(self.lamk_list * Skt * dfk_xt_all, axis=1)
                    ut = -self.umax * (bt / np.linalg.norm(bt))
            
                    # input CBF:
                    ut, u_nom, all_h = compute_control_input(xt, agent, ut, self.safety_distance, self.gamma, self.h_pow, self.umax, self.CF_DICT, self.L_list)
                    self.h_funcs.append(all_h)

                    #print(np.linalg.norm(ut), self.umax, agent)
                    self.get_logger().info(f"{ut}, {agent}.")
                    ut_dict[agent] = ut

                    self.control_inputs[agent][0].append(ut[0])
                    self.control_inputs[agent][1].append(ut[1])

                else:
                    if self.CF_DICT[agent] != "inactive":
                        self.landing_dict[agent].call_async(Trigger.Request())
                        self.CF_DICT[agent] = "inactive"

            for cfname, active in self.CF_DICT.items():
                cmd_vel_msg = Twist()

                if self.stop or active == "inactive":

                    print(f"STOP ROBOT: {cfname}")
                    cmd_vel_msg.linear.x = 0.0
                    cmd_vel_msg.linear.y = 0.0
                    cmd_vel_msg.linear.z = 0.0
                    
                else:
                    cmd_vel_msg.linear.x = ut_dict[cfname][0]
                    cmd_vel_msg.linear.y = ut_dict[cfname][1]
                    cmd_vel_msg.linear.z = 0.0

                    erg_metric = np.sum(self.lamk_list * np.square(ckt - self.phik_list))
                    self.metric_logs.append(erg_metric)

                self.ros_pubs[cfname].publish(cmd_vel_msg)

            self.it += 1
        
        self.true_timestep += 1
        print(self.true_timestep, self.it)

    def signal_handler(self, sig, frame):
            print('You pressed Ctrl+C. Turning off the controller.')

            # Stop all robots at the end
            self.land_client.call_async(Trigger.Request())
            self.stop = True

            exit()  # Force Exit

def plot_trajectories(grids, ks, agent_ck_lists, tsteps, dt, hk_list, L_list, grids_x, grids_y, pdf_values, num_agents, agent_trajectories, metric_logs, control_inputs):

    plt.rcParams['xtick.labelsize']=16
    plt.rcParams['ytick.labelsize']=16
    plt.rcParams['axes.titlesize']=16
    plt.rcParams['axes.labelsize']=16
    plt.rcParams['legend.fontsize']=16
    plt.rcParams['lines.linewidth']=3

    """# Make sure all arrays have the same length
    min_length = min([len(metric_logs)] + 
                    [len(control_inputs[agent][0]) for agent in control_inputs] +
                    [len(agent_trajectories[agent][0]) for agent in agent_trajectories])
    
    # Calculate how many complete sets of agent metrics we have
    num_complete_timesteps = len(metric_logs) // num_agents
    
    # Create time arrays of consistent length
    time_in_seconds = np.arange(min_length) * dt
    metric_time = np.arange(num_complete_timesteps) * dt"""

    phi_recon = np.zeros(grids.shape[0])
    for i, (k_vec, ck, hk) in enumerate(zip(ks, agent_ck_lists/(tsteps*dt), hk_list)):
        fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1) / hk
        phi_recon += ck * fk_vals

    # Plot the ergodic metric for all agents
    plt.figure(figsize=(10, 6))
    for i in range(1):
        plt.plot(metric_logs[0::num_agents], label=f'Total ergodic metric')
    #plt.xticks([x * dt for x in metric_logs[0::num_agents]])
    plt.xlabel('t [s]')
    plt.ylabel('Ergodic Metric')
    plt.title('Ergodic Metric Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualize the trajectories
    fig, axes = plt.subplots(1, 2, figsize=(15,7), dpi=70, tight_layout=True)

    ax = axes[0]
    ax.set_aspect('equal')
    ax.set_xlim(0, L_list[0])
    ax.set_ylim(0, L_list[1])
    ax.set_title('Original PDF')
    ax.contourf(grids_x, grids_y, pdf_values, cmap='Reds')
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    i = 0
    for agent in agent_trajectories.keys():
        # Check if trajectory data exists before plotting
        if agent_trajectories[agent][0] and agent_trajectories[agent][1]:  # Check if lists are not empty
            ax.plot(agent_trajectories[agent][0], agent_trajectories[agent][1], linestyle='-', marker='', color=colors[i], alpha=0.2, label=f'Drone {i+1}')
            ax.plot(agent_trajectories[agent][0][0], agent_trajectories[agent][1][0], linestyle='', marker='o', markersize=15, color=colors[i], alpha=1.0, label=f'Drone {i+1} start')
        else:
            print(f"Warning: No trajectory data for drone {agent}")
        i += 1
    ax.legend(loc='best')

    ax = axes[1]
    ax.set_aspect('equal')
    ax.set_xlim(0, L_list[0])
    ax.set_ylim(0, L_list[1])
    ax.set_title('Empirical Distribution')
    ax.contourf(grids_x, grids_y, phi_recon.reshape(grids_x.shape), cmap='Blues')
    i = 0
    for agent in agent_trajectories.keys():
        # Check if trajectory data exists before plotting
        if agent_trajectories[agent][0] and agent_trajectories[agent][1]:  # Check if lists are not empty
            ax.plot(agent_trajectories[agent][0], agent_trajectories[agent][1], linestyle='-', marker='', color=colors[i], alpha=0.2, label=f'Drone {i+1}')
            ax.plot(agent_trajectories[agent][0][0], agent_trajectories[agent][1][0], linestyle='', marker='o', markersize=15, color=colors[i], alpha=1.0, label=f'Drone {i+1} start')
        else:
            print(f"Warning: No trajectory data for drone {agent}")
        i += 1
    ax.legend(loc='best')

    plt.show()

    return phi_recon

def plot_rviz(x, y, pdf, broken_sensors_pub, time, L_list):
        # For RViz: broken sensor publisher
        #broken_sensors_pub = Node.create_publisher(MarkerArray, 'nebolab/targeted_points', qos_profile=1)
        markers_it = 0
        try: 
            marker_array_msg = MarkerArray()
            broken_sensors_list = []
            for pointx in range(len(x[0])):
                for pointy in range(len(y)):
                    z = pdf[pointx, pointy] 
                    if z < 0.05: # removes unnecessary lag
                        markers_it += 1
                        continue 
                    else:
                        a_val = z/7
                        z_val = z/10

                    scale = 0.02
                    broken_sensor_msg = Marker()
                    broken_sensor_msg.header.frame_id = "world"
                    broken_sensor_msg.header.stamp = time
                    broken_sensor_msg.id = markers_it
                    broken_sensor_msg.type = 1
                    broken_sensor_msg.scale.x = scale
                    broken_sensor_msg.scale.y = scale
                    broken_sensor_msg.scale.z = scale
                    broken_sensor_msg.pose.position.x = y[pointy][0] - (L_list[0] / 2)
                    broken_sensor_msg.pose.position.y = x[0][pointx] - (L_list[1] / 2)
                    broken_sensor_msg.pose.position.z = z_val
                    broken_sensor_msg.color.a = a_val
                    broken_sensor_msg.color.r = 1.0
                    broken_sensor_msg.color.g = 0.0
                    broken_sensor_msg.color.b = 0.0

                    broken_sensors_list.append(broken_sensor_msg)
                    markers_it += 1

            marker_array_msg.markers = broken_sensors_list
            broken_sensors_pub.publish(marker_array_msg)
        except Exception as ex:
            rclpy.logging.get_logger("CF_CONTROLLER_EXAMPLE").info(f"Target point locating failure: {ex}")

def main(args=None):
    rclpy.init(args=args)
    cfcontroller = CFController()

    try:
        rclpy.spin(cfcontroller)
    except (SystemExit, KeyboardInterrupt) as e:
        rclpy.logging.get_logger("CF_CONTROLLER_EXAMPLE").info('Done')

    phi_recon = plot_trajectories(cfcontroller.grids, cfcontroller.ks, cfcontroller.ck_list_updates, cfcontroller.it, cfcontroller.rate,
                      cfcontroller.hk_list, cfcontroller.L_list, cfcontroller.grids_x, cfcontroller.grids_y, cfcontroller.pdf_values,
                      cfcontroller.num_agents, cfcontroller.trajectories, cfcontroller.metric_logs, cfcontroller.control_inputs)

    # Saved data in JSON

    data_to_save = {
        "trajectories": cfcontroller.trajectories,
        "control_inputs": cfcontroller.control_inputs,
        "ergodic_metrics": cfcontroller.metric_logs,
        "pdf_values": cfcontroller.pdf_values.tolist(),
        "phi_recon": phi_recon.tolist(),
        "grids_x": cfcontroller.grids_x.tolist(),
        "grids_y": cfcontroller.grids_y.tolist(),
        "ks": cfcontroller.ks.tolist(),
        "h_funcs": cfcontroller.h_funcs,
        "parameters": {
            "rate": cfcontroller.rate,
            "umax": cfcontroller.umax,
            "safety_distance": cfcontroller.safety_distance,
            "gamma": cfcontroller.gamma,
            "h_pow": cfcontroller.h_pow,
            "L_list": cfcontroller.L_list.tolist(),
            "num_agents": cfcontroller.num_agents
        }
    }

    with open('/home/localadmin/ergodic_test_record.json', 'w') as out:
        json.dump(data_to_save, out, indent=4)

    cfcontroller.land_client.call_async(Trigger.Request())
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)

    cfcontroller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
