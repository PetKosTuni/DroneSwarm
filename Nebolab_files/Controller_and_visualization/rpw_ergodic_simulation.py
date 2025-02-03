import numpy as np
import cvxopt
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

pdf_area_scaling = 3

# Define the target distribution parameters
mean1 = np.array([0.35 * pdf_area_scaling, 0.38 * pdf_area_scaling])
cov1 = np.array([
     [0.01 * (pdf_area_scaling**2), 0.004 * (pdf_area_scaling**2)],
     [0.004 * (pdf_area_scaling**2), 0.01 * (pdf_area_scaling**2)]
])
w1 = 0.4
    
mean2 = np.array([0.68 * pdf_area_scaling, 0.25 * pdf_area_scaling])
cov2 = np.array([
    [0.005 * (pdf_area_scaling**2), -0.003 * (pdf_area_scaling**2)],
    [-0.003 * (pdf_area_scaling**2), 0.005 * (pdf_area_scaling**2)]
])
w2 = 0.2
    
mean3 = np.array([0.56 * pdf_area_scaling, 0.64 * pdf_area_scaling])
cov3 = np.array([
    [0.008 * (pdf_area_scaling**2), 0.0],
    [0.0, 0.004 * (pdf_area_scaling**2)]
])
w3 = 0.35
    
# Define the probability density function
def pdf(x):
    return w1 * mvn.pdf(x, mean1, cov1) + \
           w2 * mvn.pdf(x, mean2, cov2) + \
           w3 * mvn.pdf(x, mean3, cov3)

# Define a x-by-x 2D search space
L_list = np.array([1.0 * pdf_area_scaling, 1.0 * pdf_area_scaling])

# Discretize the search space
grids_x, grids_y = np.meshgrid(np.linspace(0, L_list[0], 100), np.linspace(0, L_list[1], 100))
grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
dx = L_list[0] / 99
dy = L_list[1] / 99

# Calculate the PDF values for the grid points
pdf_values = np.array([pdf(point) for point in grids])
pdf_values = pdf_values.reshape(grids_x.shape)

# Configure the index vectors
num_k_per_dim = 20
ks_dim1, ks_dim2 = np.meshgrid(np.arange(num_k_per_dim), np.arange(num_k_per_dim))
ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T

# Pre-processing lambda_k and h_k
lamk_list = np.power(1.0 + np.linalg.norm(ks, axis=1), -3/2.0)
hk_list = np.zeros(ks.shape[0])
for i, k_vec in enumerate(ks):
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)
    hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
    hk_list[i] = hk

# Compute the coefficients for the target distribution
phik_list = np.zeros(ks.shape[0])
pdf_vals = pdf(grids)
for i, (k_vec, hk) in enumerate(zip(ks, hk_list)):
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1) / hk
    phik = np.sum(fk_vals * pdf_vals) * dx * dy
    phik_list[i] = phik

# Specify the dynamic system
dt = 0.01
tsteps = 10000
umax = 0.3  # desired velocity 0.3 m/s
safety_distance = 0.15  # Safety distance for collision avoidance
detection_range = 0.3

def dyn(xt, ut):
    return ut

def step(xt, ut, dt):
    return xt + dt * dyn(xt, ut)

def calculate_QP_solver(H, b, all_h, error, constraint_index, gamma, h_pow):
    dist = np.linalg.norm(error)
    h_func = dist**2 - safety_distance**2
    all_h[constraint_index] = h_func

    H[constraint_index] = -2 * error
    b[constraint_index] = gamma * np.power(h_func, h_pow)
    return H, b, all_h

# Control Barrier Function
def compute_control_input(agent_states, agent_index, desired_velocity, gamma, h_pow):
    num_agents = len(agent_states)
    current_state = agent_states[agent_index]
    
    # Nominal control (constant desired velocity)
    u_nom = desired_velocity
    
    # Set up optimization problem
    Q_mat = 2 * cvxopt.matrix(np.eye(2), tc='d')
    c_mat = -2 * cvxopt.matrix(u_nom, tc='d')
    
    wall_up = np.array([current_state[0], 1 * pdf_area_scaling])
    wall_left = np.array([0, current_state[1]])
    wall_right = np.array([1 * pdf_area_scaling, current_state[1]])
    wall_down = np.array([current_state[0], 0])
    walls = [wall_up, wall_left, wall_right, wall_down]
    
    wall_dist = np.array([float('inf'),float('inf')])
    for wall in walls:
        dist = current_state - wall
        if np.linalg.norm(wall_dist) > np.linalg.norm(dist):
            wall_dist = dist
    
    constraint_index = 0
    number_of_threats = 1
    threat_indexes = []
    
    for i in range(num_agents):
        if i != agent_index:
        	
            if (np.linalg.norm(agent_states[i] - current_state)) <= detection_range:
                number_of_threats += 1
                threat_indexes.append(i)
    
    # Inequality constraints (CBF)
    H = np.zeros([number_of_threats, 2])
    b = np.zeros([number_of_threats, 1])
    all_h = np.zeros(number_of_threats)

    for i in range(num_agents):
        if i != agent_index and i in threat_indexes:
            H, b, all_h = calculate_QP_solver(H, b, all_h, current_state - agent_states[i], constraint_index, gamma, h_pow)
            constraint_index += 1
        
    H, b, all_h = calculate_QP_solver(H, b, all_h, wall_dist, constraint_index, gamma, h_pow)
    
    # Convert to cvxopt matrices
    H_mat = cvxopt.matrix(H, tc='d')
    b_mat = cvxopt.matrix(b, tc='d')

    # Solve optimization problem
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(Q_mat, c_mat, H_mat, b_mat, verbose=False)
    
    # Extract solution
    current_input = np.array([sol['x'][0], sol['x'][1]])
        
    return current_input, u_nom, all_h

# Function to run all agents simultaneously
def run_agents(initial_states, num_agents):

    gamma = 5  # CBF parameter
    h_pow = 1    # CBF parameter

    x_trajs = np.zeros((num_agents, tsteps, 2))
    ck_list_updates = np.zeros(ks.shape[0])
    metric_logs = []
    phi_recon_list = []
    control_inputs = np.zeros((num_agents, tsteps, 2))
    xt = initial_states.copy()
    
    for t in range(tsteps):
        for agent in range(num_agents):
        
            # step 1: evaluate all the fourier basis functions at the current state
            # Equation 9
            fk_xt_all = np.prod(np.cos(np.pi * ks / L_list * xt[agent]), axis=1) / hk_list
            
             # step 2: update the coefficients
            ck_list_updates += (fk_xt_all / num_agents) * dt
	    
    	    # step 3: compute the derivative of all basis functions at the current state
            
    	    # Equation 9
            k1 = np.pi * ks[:,0] / L_list[0]
            k2 = np.pi * ks[:,1] / L_list[1]
    
    	    # Equation 20
            x1 = xt[agent,0]
            x2 = xt[agent,1]
            dfk_xt_all = np.array([
		    -k1 * np.sin(k1 * x1) * np.cos(k2 * x2),
		    -k2 * np.cos(k1 * x1) * np.sin(k2 * x2),
    	    ]) / hk_list
	
    	    # step 4: compute control signal
    	    # Equation 10
            ckt = ck_list_updates / (t * dt + dt)
            
    	    # Equation 16
            Skt = ckt - phik_list
    	    # Equation 26
            bt = np.sum(lamk_list * Skt * dfk_xt_all, axis=1)
            ut = -umax * (bt / np.linalg.norm(bt))
    
            # input CBF:
            ut, u_nom, all_h = compute_control_input(xt, agent, ut, gamma, h_pow)
            
            # step 5: execute the control, move on to the next iteration
            xt_new = step(xt[agent], ut, dt)

            xt[agent] = xt_new.copy()
	    
            x_trajs[agent, t, :] = xt[agent].copy()
            control_inputs[agent, t, :] = ut.copy()
        
            # Equation 3 from paper 2agent_stateagent_state
            erg_metric = np.sum(lamk_list * np.square(ckt - phik_list))
            metric_logs.append(erg_metric)
        
        phi_recon_list.append(ck_list_updates.copy())
        
    phi_recon_list = np.asarray(phi_recon_list)
        
    return x_trajs, ck_list_updates, metric_logs, control_inputs, phi_recon_list

# Run agents
num_agents = 3
drone_size = 0.01

initial_states = np.random.uniform(low=0.2 * pdf_area_scaling, high=0.6 * pdf_area_scaling, size=(num_agents, 2))

agent_trajectories, agent_ck_lists, agent_metric_logs, agent_control_inputs, phi_recon_list = run_agents(initial_states, num_agents)

def reconstruct_emp_dist(grids, ks, agent_ck_lists, tsteps, dt, hk_list, L_list):

    # Reconstruct the empirical distribution
    phi_recon = np.zeros(grids.shape[0])
    for i, (k_vec, ck, hk) in enumerate(zip(ks, agent_ck_lists/(tsteps*dt), hk_list)):
        fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1) / hk
        phi_recon += ck * fk_vals
        
    return phi_recon

phi_recon = reconstruct_emp_dist(grids, ks, agent_ck_lists, tsteps, dt, hk_list, L_list)

# Visualize the trajectories
fig, axes = plt.subplots(1, 2, figsize=(15,7), dpi=70, tight_layout=True)

ax = axes[0]
ax.set_aspect('equal')
ax.set_xlim(0, L_list[0])
ax.set_ylim(0, L_list[1])
ax.set_title('Original PDF')
ax.contourf(grids_x, grids_y, pdf_values, cmap='Reds')
colors = ['C0', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C1']
for i in range(num_agents):
    ax.plot(agent_trajectories[i, ::10, 0], agent_trajectories[i, ::10, 1], linestyle='-', marker='', color=colors[i], alpha=0.2, label=f'Agent {i+1}')
    inner1 = plt.Circle((agent_trajectories[i, 0, 0], agent_trajectories[i, 0, 1]), drone_size, color=colors[i], label=f'Agent {i+1} Start')
    inner2 = plt.Circle((agent_trajectories[i, 0, 0], agent_trajectories[i, 0, 1]), safety_distance, edgecolor=colors[i], facecolor='none')
    outer = plt.Circle((agent_trajectories[i, 0, 0], agent_trajectories[i, 0, 1]), detection_range, edgecolor=colors[i], facecolor='none', linestyle='--')
    ax.add_patch(inner1)
    ax.add_patch(inner2)
    ax.add_patch(outer)
ax.legend(loc='best')

ax = axes[1]
ax.set_aspect('equal')
ax.set_xlim(0, L_list[0])
ax.set_ylim(0, L_list[1])
ax.set_title('Empirical Distribution')
ax.contourf(grids_x, grids_y, phi_recon.reshape(grids_x.shape), cmap='Blues')
for i in range(num_agents):
    ax.plot(agent_trajectories[i, ::10, 0], agent_trajectories[i, ::10, 1], linestyle='-', marker='', color=colors[i], alpha=0.2, label=f'Agent {i+1}')
    inner1 = plt.Circle((agent_trajectories[i, 0, 0], agent_trajectories[i, 0, 1]), drone_size, color=colors[i], label=f'Agent {i+1} Start')
    inner2 = plt.Circle((agent_trajectories[i, 0, 0], agent_trajectories[i, 0, 1]), safety_distance, edgecolor=colors[i], facecolor='none')
    outer = plt.Circle((agent_trajectories[i, 0, 0], agent_trajectories[i, 0, 1]), detection_range, edgecolor=colors[i], facecolor='none', linestyle='--')
    ax.add_patch(inner1)
    ax.add_patch(inner2)
    ax.add_patch(outer)
ax.legend(loc='best')

plt.show()

# Plot the ergodic metric for all agents
plt.figure(figsize=(10, 6))
for i in range(1):
    plt.plot(agent_metric_logs[0::num_agents], label=f'Total ergodic metric')
plt.xlabel('Time Step')
plt.ylabel('Ergodic Metric')
plt.title('Ergodic Metric Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot the control inputs for all agents

fig, axes = plt.subplots(num_agents, 1, figsize=(10, 4*num_agents), sharex=True)

if num_agents == 1:
    axes.plot(agent_control_inputs[i, :, 0], label='u_x')
    axes.plot(agent_control_inputs[i, :, 1], label='u_y')
    axes.set_ylabel(f'Agent {i+1} Control')
    axes.legend()
    axes.grid(True)
    axes.set_xlabel('Time Step')
else:
    for i in range(num_agents):
        axes[i].plot(agent_control_inputs[i, :, 0], label='u_x')
        axes[i].plot(agent_control_inputs[i, :, 1], label='u_y')
        axes[i].set_ylabel(f'Agent {i+1} Control')
        axes[i].legend()
        axes[i].grid(True)
        axes[-1].set_xlabel('Time Step')
plt.suptitle('Control Inputs Over Time')
plt.tight_layout()
plt.show()

# Create animation function
def create_multi_agent_animation(agent_trajectories, grids_x, grids_y, pdf_vals, num_agents, emp_dist, grids, ks, tsteps, dt, hk_list, L_list):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=70, tight_layout=True)
    ax.set_aspect('equal')
    ax.set_xlim(0.0, 1.0 * pdf_area_scaling)
    ax.set_ylim(0.0, 1.0 * pdf_area_scaling)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    if emp_dist == False:
        ax.contourf(grids_x, grids_y, pdf_vals.reshape(grids_x.shape), cmap='Reds', levels=10)

    colors = ['C0', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C1']
    robot_trajs = [ax.plot([], [], linestyle='-', marker='', color=colors[i], alpha=0.5)[0] for i in range(num_agents)]
    robot_inner1 = [ax.add_patch(plt.Circle((agent_trajectories[i, 0, 0], agent_trajectories[i, 0, 1]), drone_size, color=colors[i])) for i in range(num_agents)]
    robot_inner2 = [ax.add_patch(plt.Circle((agent_trajectories[i, 0, 0], agent_trajectories[i, 0, 1]), safety_distance, edgecolor=colors[i], facecolor='none')) for i in range(num_agents)]
    robot_outer = [ax.add_patch(plt.Circle((agent_trajectories[i, 0, 0], agent_trajectories[i, 0, 1]), detection_range, edgecolor=colors[i], facecolor='none', linestyle='--')) for i in range(num_agents)]
    
    def update_plot(frame):
        t = int(frame * 10)
        print(t)
        if emp_dist == True:
            phi_recon = reconstruct_emp_dist(grids, ks, pdf_vals[t], tsteps, dt, hk_list, L_list)
            ax.contourf(grids_x, grids_y, phi_recon.reshape(grids_x.shape), cmap='Blues', levels=10)
        ax.patches.clear()
        patches = []
        for i in range(num_agents):
            patches.append(ax.add_patch(plt.Circle((agent_trajectories[i, t, 0], agent_trajectories[i, t, 1]), drone_size, color=colors[i])))
            patches.append(ax.add_patch(plt.Circle((agent_trajectories[i, t, 0], agent_trajectories[i, t, 1]), safety_distance, edgecolor=colors[i], facecolor='none')))
            patches.append(ax.add_patch(plt.Circle((agent_trajectories[i, t, 0], agent_trajectories[i, t, 1]), detection_range, edgecolor=colors[i], facecolor='none', linestyle='--')))
            robot_trajs[i].set_data(
                agent_trajectories[i, :t, 0][::10],
                agent_trajectories[i, :t, 1][::10]
            )
        return patches + robot_trajs

    ani = animation.FuncAnimation(fig, update_plot, frames=int(agent_trajectories.shape[1]/10)-1, blit=True, interval=30, repeat=False)
    plt.close(fig)  # Prevent display of the figure
    return ani

# Create and save the animation, uncomment when animations are needed
#anim = create_multi_agent_animation(agent_trajectories, grids_x, grids_y, pdf_values, num_agents, False, grids, ks, tsteps, dt, hk_list, L_list)
#emp_dist_anim = create_multi_agent_animation(agent_trajectories, grids_x, grids_y, phi_recon_list, num_agents, True, grids, ks, tsteps, dt, hk_list, L_list)

def save_animation(anim, filename, directory):
    directory = r"{}".format(directory)
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    anim.save(filepath, writer='ffmpeg', fps=30)

    print(f"Animation saved to {filepath}")

# Example usage:
# save_animation(anim, 'multi_agent_ergodic_control.mp4', directory='INPUT DIRECTORY')
# save_animation(emp_dist_anim, 'empirical_distribution.mp4', directory='INPUT DIRECTORY')

# Alternatively, you can use forward slashes, which work on both Windows and Unix-like systems:
#save_animation(anim, 'multi_agent_ergodic_control.mp4', directory='C:/Users/User/Desktop/Folders/Misc/Koulu/Robo666')
