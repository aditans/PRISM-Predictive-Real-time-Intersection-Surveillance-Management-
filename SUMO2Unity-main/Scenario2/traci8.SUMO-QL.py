# Step 1: Add modules to provide access to specific libraries and functions
import os  
import sys  
import random
import numpy as np
import matplotlib.pyplot as plt 

# Define Traffic Light ID
TLS_ID = "J1"

# --- ⚠️ UPDATE THESE CREDENTIALS ⚠️ ---
THINGSBOARD_HOST = ''  
DEVICE_ACCESS_TOKEN = 'PASTE_YOUR_NEW_ACCESS_TOKEN_HERE' 


# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci  

# ------------------------------------------------------------------
# Step 4: Define Sumo configuration (CORRECTED FOR RELIABLE LAUNCH)
# ------------------------------------------------------------------
# We remove the ZeroMQ/NetMQ flags to ensure traci.start() works without error.
Sumo_config = [
    'sumo-gui',
    '-c', 'Sumo2Unity.sumocfg',         
    '--step-length', '0.10',
    '--delay', '0',
    '--lateral-resolution', '0',
    
    # --- NETMQ / ZEROMQ OPTIONS REMOVED TO PREVENT SHUTDOWN ERROR ---
]

# ------------------------------------------------------------------
# Step 5: Open connection between SUMO and TraCI (START SUMO AND CONNECT)
# ------------------------------------------------------------------
try:
    # This command reliably starts a new SUMO process and connects the Python client to its server.
    traci.start(Sumo_config)
    traci.gui.setSchema("View #0", "real world")
    print("✅ SUMO started successfully and RL client connected. (NetMQ communication is DISABLED)")
except Exception as e:
    sys.exit(f"❌ Failed to start and connect to SUMO: {e}")

# -------------------------
# Step 6: Define Variables
# -------------------------

# Variables for RL State (queue lengths from detectors and current phase)
q_EB_0 = 0
q_EB_1 = 0
q_EB_2 = 0
q_SB_0 = 0
q_SB_1 = 0
q_SB_2 = 0
### NEW VARIABLES ###
q_WB_0 = 0
q_WB_1 = 0
q_WB_2 = 0
q_NB_0 = 0
q_NB_1 = 0
q_NB_2 = 0
### END NEW VARIABLES ###
current_phase = 0

# ---- Reinforcement Learning Hyperparameters ----
TOTAL_STEPS = 10000    # The total number of simulation steps for continuous (online) training.

ALPHA = 0.1            
GAMMA = 0.9            
EPSILON = 0.1          

ACTIONS = [0, 1]       # The discrete action space (0 = keep phase, 1 = switch phase)

# Q-table dictionary: key = state tuple, value = numpy array of Q-values for each action
Q_table = {}

# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

# -------------------------
# Step 7: Define Functions
# -------------------------


def get_max_Q_value_of_state(s): #1. Objective Function 1
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])

def get_reward(state):  #2. Constraint 2 
    """
    Negative of total queue length to encourage shorter queues.
    Now sums 12 queue lengths.
    """
    total_queue = sum(state[:-1])  # Exclude the current_phase element (12 queue lengths summed)
    reward = -float(total_queue)
    return reward

def get_state():  #3.& 4. Constraint 3 & 4
    global q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2
    global q_WB_0, q_WB_1, q_WB_2, q_NB_0, q_NB_1, q_NB_2, current_phase 
    
    # Detector IDs for Node1-2-EB
    detector_Node1_2_EB_0 = "Node1_2_EB_0"
    detector_Node1_2_EB_1 = "Node1_2_EB_1"
    detector_Node1_2_EB_2 = "Node1_2_EB_2"
    
    # Detector IDs for Node2-7-SB
    detector_Node2_7_SB_0 = "Node2_7_SB_0"
    detector_Node2_7_SB_1 = "Node2_7_SB_1"
    detector_Node2_7_SB_2 = "Node2_7_SB_2"

    # NEW Detector IDs for Node2-3-WB
    detector_Node2_3_WB_0 = "Node2_3_WB_0"
    detector_Node2_3_WB_1 = "Node2_3_WB_1"
    detector_Node2_3_WB_2 = "Node2_3_WB_2"

    # NEW Detector IDs for Node2-5-NB
    detector_Node2_5_NB_0 = "Node2_5_NB_0"
    detector_Node2_5_NB_1 = "Node2_5_NB_1"
    detector_Node2_5_NB_2 = "Node2_5_NB_2"
    
    # Traffic light ID
    traffic_light_id = TLS_ID
    
    # Get queue lengths from existing detectors
    q_EB_0 = get_queue_length(detector_Node1_2_EB_0)
    q_EB_1 = get_queue_length(detector_Node1_2_EB_1)
    q_EB_2 = get_queue_length(detector_Node1_2_EB_2)
    
    q_SB_0 = get_queue_length(detector_Node2_7_SB_0)
    q_SB_1 = get_queue_length(detector_Node2_7_SB_1)
    q_SB_2 = get_queue_length(detector_Node2_7_SB_2)

    # Get queue lengths from NEW detectors
    q_WB_0 = get_queue_length(detector_Node2_3_WB_0)
    q_WB_1 = get_queue_length(detector_Node2_3_WB_1)
    q_WB_2 = get_queue_length(detector_Node2_3_WB_2)
    
    q_NB_0 = get_queue_length(detector_Node2_5_NB_0)
    q_NB_1 = get_queue_length(detector_Node2_5_NB_1)
    q_NB_2 = get_queue_length(detector_Node2_5_NB_2)
    
    # Get current phase index
    current_phase = get_current_phase(traffic_light_id)
    
    # Return the expanded state tuple (12 queue lengths + 1 phase)
    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, 
            q_WB_0, q_WB_1, q_WB_2, q_NB_0, q_NB_1, q_NB_2,
            current_phase)

def apply_action(action, tls_id=TLS_ID): #5. Constraint 5
    """
    Executes the chosen action on the traffic light, combining:
      - Min Green Time check
      - Switching to the next phase if allowed
    Constraint #5: Ensure at least MIN_GREEN_STEPS pass before switching again.
    """
    global last_switch_step
    
    if action == 0:
        # Do nothing (keep current phase)
        return
    
    elif action == 1:
        # Check if minimum green time has passed before switching
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step = current_simulation_step


def update_Q_table(old_state, action, reward, new_state): #6. Constraint 6
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))
    
    
    # 1) Predict current Q-values from old_state (current state)
    old_q = Q_table[old_state][action]
    # 2) Predict Q-values for new_state to get max future Q (new state)
    best_future_q = get_max_Q_value_of_state(new_state)
    # 3) Incorporate ALPHA to partially update the Q-value and update Q table
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)


def get_action_from_policy(state): #7. Constraint 7
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        return int(np.argmax(Q_table[state]))


def get_queue_length(detector_id): #8.Constraint 8
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id): #8.Constraint 8
    return traci.trafficlight.getPhase(tls_id)

# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------

# Lists to record data for plotting
step_history = []
reward_history = []
queue_history = []

cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step
    
    state = get_state()
    action = get_action_from_policy(state)
    apply_action(action)
    
    traci.simulationStep()  # Advance simulation by one step
    
    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward
    
    update_Q_table(state, action, reward, new_state)
    
    # Print Q-values for the old_state right after update
    updated_q_vals = Q_table[state]

    # Record data every 1 steps
    if step % 1 == 0:
        print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}, Q-values(current_state): {updated_q_vals}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))  # sum of all 12 queue lengths
        print("Current Q-table:")
        # Print only a sample of the Q-table for brevity
        for i, (st, qvals) in enumerate(Q_table.items()):
            if i < 5:
                print(f"  {st} -> {qvals}")
            elif i == 5:
                print(f"  ... (showing 5 of {len(Q_table)} states)")
                break
     
# -------------------------
# Step 9: Close connection between SUMO and Traci
# -------------------------
traci.close()

# Print final Q-table info
print("\nOnline Training completed. Final Q-table size:", len(Q_table))
# Print final Q-table info (Sampled)
print("\nFinal Q-table sample:")
for i, (st, actions) in enumerate(Q_table.items()):
    if i < 5:
        print("State:", st, "-> Q-values:", actions)
    elif i == 5:
        print(f"  ... (showing 5 of {len(Q_table)} states)")
        break


# -------------------------
# Visualization of Results
# -------------------------

# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
# ... (rest of visualization code)
plt.title("RL Training: Cumulative Reward over Steps")
plt.legend()
plt.grid(True) 
plt.show()

# Plot Total Queue Length over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length (12 Detectors)")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()