import traci
import zmq
import sys
import time
import json
import random # Used for unique ZeroMQ dealer identity

# --- Configuration ---
RL_TRACI_PORT = 54078  # The port your RL script uses to control SUMO
ZMQ_PUB_PORT = 5556   # Unity Subscriber connects here (SUMO -> Unity)
ZMQ_ROUTER_PORT = 5557 # Unity Dealer connects here (Unity -> SUMO)

# --- ZMQ Context and Sockets ---
context = zmq.Context()
# Publisher socket: Sends data TO Unity (Vehicle/TL state)
pub_socket = context.socket(zmq.PUB)
pub_socket.bind(f"tcp://*:{ZMQ_PUB_PORT}")

# Router socket: Receives data FROM Unity (Ego Vehicle state)
# We will not actively connect this to TraCI, but ensure it receives data.
router_socket = context.socket(zmq.ROUTER)
router_socket.bind(f"tcp://*:{ZMQ_ROUTER_PORT}")

# --- TraCI Connection ---
MAX_ATTEMPTS = 30
connected = False
print("\n--- Starting NetMQ Bridge ---")
print(f"Waiting for RL controller to start SUMO on port {RL_TRACI_PORT}...")

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        # We connect to the SUMO instance started by your RL script
        traci.init(RL_TRACI_PORT)
        connected = True
        print(f"✅ TraCI connection successful after {attempt} attempts!")
        break
    except traci.exceptions.TraCIException:
        time.sleep(1) 

if not connected:
    print("❌ Failed to connect to RL-controlled SUMO via TraCI. Exiting.")
    sys.exit(1)

# --- Main Bridge Loop ---

print(f"ZMQ PUB socket listening on {ZMQ_PUB_PORT}")
print(f"ZMQ ROUTER socket listening on {ZMQ_ROUTER_PORT}")

# --- Set Up Vehicle Subscription for TraCI (Required to get full vehicle list) ---
# We want data on all vehicles, but since your RL script already calls simulationStep(),
# this bridge mainly focuses on getting the vehicle/TL state and sending it.
# We subscribe to the full list of vehicles in the network.
try:
    traci.simulation.subscribeContext(traci.simulation.getID(), 
                                      0, # Time to check (all time)
                                      [traci.constants.VAR_LANE_ID, traci.constants.VAR_POSITION, traci.constants.VAR_ANGLE, traci.constants.VAR_SPEED, traci.constants.VAR_TYPE])
except Exception as e:
    print(f"Warning: Failed to subscribe to simulation context: {e}")

def create_sumo_data_json():
    """Gathers data from TraCI and formats it for Unity (PUB socket)."""
    
    # 1. Gather Vehicle Data (Simplified, as full TraCI vehicle data can be complex)
    vehicle_data = []
    vehicle_ids = traci.vehicle.getIDList()
    
    for vid in vehicle_ids:
        try:
            pos = traci.vehicle.getPosition(vid) # (x, y)
            angle = traci.vehicle.getAngle(vid)
            speed = traci.vehicle.getSpeed(vid)
            v_type = traci.vehicle.getTypeID(vid)
            
            # Unity expects (x, z, y) based on your SimulationController.cs
            # We must map SUMO (x, y, angle) to Unity (x, z, -y, rotation) 
            vehicle_data.append({
                "vehicle_id": vid,
                "position": [pos[0], pos[1], 0.0], # Assuming SUMO y is Unity z, and height is 0
                "angle": angle,
                "type": v_type,
                "long_speed": speed,
                "vert_speed": 0.0,
                "lat_speed": 0.0
            })
        except Exception:
            # Vehicle may have been removed in the current step
            continue

    # 2. Gather Traffic Light Data
    tl_state = traci.trafficlight.getPhaseProgram(TLS_ID).phases[traci.trafficlight.getPhase(TLS_ID)].state
    
    tl_json = {
        "type": "trafficlights",
        "lights": [{
            "junction_id": TLS_ID,
            "state": tl_state
        }]
    }

    vehicle_json = {
        "type": "vehicles",
        "vehicles": vehicle_data
    }
    
    # Send TL update first
    pub_socket.send_string(json.dumps(tl_json))
    
    # Send Vehicle update second
    pub_socket.send_string(json.dumps(vehicle_json))


def receive_ego_data():
    """Receives ego vehicle data from Unity (ROUTER socket)."""
    # This loop requires non-blocking sockets or polling for safety, 
    # but for simplicity, we use TryReceive.
    try:
        # Check if Unity sent data (identity frame, empty frame, data frame)
        msg_parts = router_socket.recv_multipart(zmq.NOBLOCK)
        
        # Check if message is properly formatted (should have 3 parts: identity, empty, data)
        if len(msg_parts) == 3:
            ego_json_str = msg_parts[2].decode('utf-8')
            
            # The structure is complex, but we expect an array of vehicles (just the ego car)
            # You would need to parse this JSON and then use traci.vehicle.moveToVType() 
            # or traci.vehicle.moveToXY() to update the vehicle's position in SUMO.
            
            # For now, just confirming receipt:
            # print(f"Received Ego Update from Unity: {ego_json_str[:50]}...")
            
            # Acknowledge the receipt (ROUTER/DEALER pattern requires a response for clean flow)
            # Send identity back, followed by an empty frame, followed by a response
            router_socket.send_multipart([msg_parts[0], b'', b'ACK'])
            
    except zmq.Again:
        # No message received, safe to ignore
        pass
    except Exception as e:
        print(f"Error receiving ego data: {e}")

try:
    # Main loop runs forever, synchronized by the RL script's traci.simulationStep()
    while True:
        # 1. Gather all required data from TraCI and send to Unity (PUB)
        create_sumo_data_json()
        
        # 2. Check for and receive ego vehicle data from Unity (ROUTER)
        receive_ego_data()
        
        # NOTE: The simulation time is advanced by the RL control script's traci.simulationStep().
        
except KeyboardInterrupt:
    print("\nBridge shutting down.")
except Exception as e:
    print(f"Critical error in bridge loop: {e}")
finally:
    traci.close()
    pub_socket.close()
    router_socket.close()
    context.term()