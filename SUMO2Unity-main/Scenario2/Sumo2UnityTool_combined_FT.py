#  Author  : Ahmad Mohammadi, PhD – York University
#  License : MIT
# ────────────────────────────────────────────────────────────────
import os, sys, json, math, time, queue, threading, statistics, datetime
import tkinter as tk, webbrowser
from tkinter import ttk, messagebox
from PIL import Image, ImageTk   # pip install pillow
import zmq, logging              # pip install pyzmq
import requests                  # pip install requests

# ════════════════════════════════════════════════════════════════
#  DEFAULTS (shared by GUI & simulation)
# ════════════════════════════════════════════════════════════════
DEFAULTS = {
    "IntegrationStartTime": 540,
    "ExperimentStartTime" : 600,
    "ExperimentEndTime"   : 720,
    "steplength"          : 0.1,
    "lateral_resolution"  : 0.3,
    "zoom"                : 150.0,   # (bigger value → closer)
    "subscribe_radius"    : 250.0    # ★ NEW (TraCI context radius)
}

VERSION      = "Sumo2Unity v2.0.0"
LINKEDIN_URL = "https://www.linkedin.com/in/ahmadmohammadi1441/"

# ------------------ ThingsBoard HTTP Sender (background) ------------------
TB_HTTP_URL_TEMPLATE = "https://thingsboard.cloud/api/v1/{token}/telemetry"
TB_TOKEN = os.environ.get("TB_DEVICE_TOKEN", "YOUR_DEVICE_TOKEN")  # set env var or replace
TB_URL = TB_HTTP_URL_TEMPLATE.format(token=TB_TOKEN)
TB_TIMEOUT = 2.0
TB_BATCH_SIZE = 10        # how many payloads to bundle, set 1 for single
TB_RETRY_MAX = 3
TB_RETRY_BASE = 0.5       # seconds

tb_queue = queue.Queue(maxsize=2000)
tb_stop = threading.Event()

def tb_sender_thread():
    session = requests.Session()
    while not tb_stop.is_set():
        try:
            batch = []
            item = tb_queue.get(timeout=1.0)
            batch.append(item)
            while len(batch) < TB_BATCH_SIZE:
                try:
                    batch.append(tb_queue.get_nowait())
                except queue.Empty:
                    break
            payload = batch if len(batch) > 1 else batch[0]
            for attempt in range(1, TB_RETRY_MAX + 1):
                try:
                    r = session.post(TB_URL, json=payload, timeout=TB_TIMEOUT)
                    if r.status_code in (200, 201, 202):
                        break
                    else:
                        time.sleep(TB_RETRY_BASE * (2**(attempt-1)))
                except Exception:
                    time.sleep(TB_RETRY_BASE * (2**(attempt-1)))
            for _ in batch:
                try:
                    tb_queue.task_done()
                except Exception:
                    pass
        except queue.Empty:
            continue
    session.close()

threading.Thread(target=tb_sender_thread, daemon=True).start()

# ═════════ helper to reach packaged resources ═══════════════════
def resource_path(fname: str) -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, fname)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)

# ═════════════════ GUI  SET-UP ══════════════════════════════════
root = tk.Tk(); root.title("Sumo2Unity Tool"); root.resizable(True, True)

def load_resized(path: str, target_w: int) -> ImageTk.PhotoImage:
    try:
        img = Image.open(resource_path(path))
        r   = img.height / img.width
        return ImageTk.PhotoImage(img.resize((target_w, int(target_w*r))))
    except Exception:
        return ImageTk.PhotoImage(Image.new("RGB", (target_w, int(target_w*0.4)), (200,200,200)))

IMG_W = 600
banner_imgs = [load_resized("2.Integration.JPG", IMG_W),
               load_resized("2.Integration_B.JPG", IMG_W)]
banner_lbl  = tk.Label(root, image=banner_imgs[0])
banner_lbl.grid(row=0, column=0, columnspan=4, pady=(6, 12))
def swap(idx=[0]):
    idx[0] = (idx[0] + 1) % len(banner_imgs)
    banner_lbl.configure(image=banner_imgs[idx[0]]); root.after(2000, swap)
root.after(2000, swap)

root.columnconfigure(1, weight=1)
entries, row = {}, 1
for k, v in DEFAULTS.items():
    label_text = ("zoom (bigger value → closer)" if k == "zoom"
                  else "subscribe radius (m)"    if k == "subscribe_radius"
                  else k)
    ttk.Label(root, text=label_text).grid(row=row, column=0,
                                          sticky="e", padx=6, pady=3)
    e = ttk.Entry(root); e.insert(0, str(v))
    e.grid(row=row, column=1, sticky="we", padx=6, pady=3)
    entries[k] = e; row += 1

# ── NEW OPTIONS ────────────────────────────────────────────────
use_gui_var       = tk.BooleanVar(value=True)
rtf_var           = tk.BooleanVar(value=True)
free_cam_var      = tk.BooleanVar(value=False)        # ★ NEW (Free-cam)
fixed_timing_var  = tk.BooleanVar(value=False)       # ★ NEW (use fixed timing)
# Optionally allow user to tune fixed durations in code or add UI later
FIXED_PHASE_DURATIONS = [30.0, 5.0, 30.0, 5.0]       # seconds per phase (change as needed)

ttk.Checkbutton(root, text="Run SUMO with GUI",  variable=use_gui_var)\
   .grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=3); row += 1
ttk.Checkbutton(root, text="Calculate RTF",      variable=rtf_var)\
   .grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=3); row += 1
ttk.Checkbutton(root, text="Free camera (no follow ego vehicle)",
                variable=free_cam_var) \
   .grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=3); row += 1  # ★ NEW
ttk.Checkbutton(root, text="Use fixed timing (no RL)", variable=fixed_timing_var) \
   .grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=3); row += 1
# ────────────────────────────────────────────────────────────────

def center(win):
    win.update_idletasks()
    w, h = win.winfo_width(), win.winfo_height()
    x = (win.winfo_screenwidth()-w)//2
    y = (win.winfo_screenheight()-h)//2
    win.geometry(f"{w}x{h}+{x}+{y}")

def pop_img(title, img_path, w):
    pop = tk.Toplevel(root); pop.title(title); pop.resizable(False, False)
    im  = load_resized(img_path, w); tk.Label(pop, image=im).pack()
    pop.im = im; ttk.Button(pop, text="Close", command=pop.destroy).pack(pady=6)
    center(pop)

def show_help():    pop_img("Help", "Help.JPG", 874)
def show_contact():
    pop = tk.Toplevel(root); pop.title("Contact / License")
    t   = tk.Text(pop, wrap="word", width=80, height=24)
    t.insert("1.0", f"{VERSION}\n\nContact: Ahmad Mohammadi\nLinkedIn: {LINKEDIN_URL}\n\nMIT License — see repository")
    t.config(state="disabled"); t.pack(expand=True, fill="both"); center(pop)
def show_pubs():
    pop = tk.Toplevel(root); pop.title("Publications")
    txt = tk.Text(pop, wrap="word", width=100, height=18)
    pubs = """\
1. Mohammadi, A., Park, P. Y., Nourinejad, M., Cherakkatil, M. S. B., & Park, H. S. (2024, June).
   SUMO2Unity: An Open-Source Traffic Co-Simulation Tool to Improve Road Safety.
   In 2024 IEEE Intelligent Vehicles Symposium (IV) (pp. 2523-2528). IEEE.

2. Mohammadi, A., Park, P. Y., Nourinejad, M., & Cherakkatil, M. S. B. (2025, May).
   Development of a Virtual Reality Traffic Simulation to Analyze Road User Behavior.
   In 2025 7th International Congress on Human-Computer Interaction, Optimization
   and Robotic Applications (ICHORA) (pp. 1-5). IEEE.
   
3. Mohammadi, A., Cherakkatil, M. S. B., Park, P. Y., Nourinejad, M., & Asgary, A. (2025). 
   A novel virtual reality traffic simulation for enhanced traffic safety assessment [Preprint].
   Preprints. https://doi.org/10.20944/preprints202508.0112.v1
"""
    txt.insert("1.0", pubs); txt.config(state="disabled")
    txt.pack(expand=True, fill="both", padx=8, pady=8); center(pop)

# ═════════════════ SIMULATION (run_sim) ═════════════════════════
def run_sim(cfg: dict):
    import traci
    from traci.constants import VAR_POSITION3D, VAR_ANGLE, VAR_TYPE
    import numpy as np # REQUIRED for some RL math
    import random

    # ---------- apply GUI parameters ----------
    IntegrationStartTime = cfg["IntegrationStartTime"]
    ExperimentStartTime  = cfg["ExperimentStartTime"]
    ExperimentEndTime    = cfg["ExperimentEndTime"]
    steplength           = cfg["steplength"]
    lateral_resolution   = cfg["lateral_resolution"]
    zoom_level           = cfg["zoom"]
    subscribe_radius     = cfg["subscribe_radius"]
    use_gui              = cfg["use_gui"]
    calc_rtf             = cfg["calc_rtf"]
    free_cam             = cfg["free_cam"]
    use_fixed_timing     = cfg.get("use_fixed_timing", False)

    # ---------- logging ----------
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # --- 1. Q-LEARNING PARAMETERS AND VARIABLES (only used if not fixed timing) ---
    TLS_ID = "J1" # Target Traffic Light ID
    TOTAL_STEPS = 10000
    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON = 0.1
    ACTIONS = [0, 1] # 0 = Keep, 1 = Switch
    Q_table = {}
    MIN_GREEN_STEPS = 100
    last_switch_step = -MIN_GREEN_STEPS
    current_simulation_step = 0

    # --- Q-table persistence (still safe to keep; RL simply won't update when fixed) ---
    import pickle
    QTABLE_PATH = os.path.join(os.path.dirname(__file__), "Q_table.pkl")
    AUTOSAVE_INTERVAL = 1000

    if not use_fixed_timing:
        if os.path.exists(QTABLE_PATH):
            try:
                with open(QTABLE_PATH, "rb") as f:
                    Q_table = pickle.load(f)
                logger.info(f"Loaded existing Q-table with {len(Q_table)} states.")
            except Exception as e:
                logger.warning(f"Failed to load Q-table: {e}")
        else:
            logger.info("No previous Q-table found. Starting fresh.")
    else:
        logger.info("Starting simulation in fixed-timing mode (no RL).")

    # Detector IDs
    DETECTORS_EB = ["Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2"]
    DETECTORS_SB = ["Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"]
    DETECTORS_WB = ["Node2_3_WB_0", "Node2_3_WB_1", "Node2_3_WB_2"]
    DETECTORS_NB = ["Node2_5_NB_0", "Node2_5_NB_1", "Node2_5_NB_2"]
    ALL_DETECTORS = DETECTORS_EB + DETECTORS_SB + DETECTORS_WB + DETECTORS_NB

    # RL helper functions (used only if RL)
    def get_queue_length(detector_id):
        return traci.lanearea.getLastStepVehicleNumber(detector_id)
    def get_current_phase(tls_id):
        return traci.trafficlight.getPhase(tls_id)
    def get_max_Q_value_of_state(s):
        if s not in Q_table:
            Q_table[s] = np.zeros(len(ACTIONS))
        return np.max(Q_table[s])
    def get_reward(state):
        total_queue = sum(state[:-1])
        return -float(total_queue)
    def get_state():
        state_list = [get_queue_length(d) for d in ALL_DETECTORS]
        current_phase = get_current_phase(TLS_ID)
        state_list.append(current_phase)
        return tuple(state_list)
    def apply_action(action, tls_id=TLS_ID):
        nonlocal last_switch_step, current_simulation_step
        if action == 0:
            return
        elif action == 1:
            if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
                program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                num_phases = len(program.phases)
                next_phase = (get_current_phase(tls_id) + 1) % num_phases
                traci.trafficlight.setPhase(tls_id, next_phase)
                last_switch_step = current_simulation_step
    def update_Q_table(old_state, action, reward, new_state):
        # periodic autosave
        if current_simulation_step and current_simulation_step % AUTOSAVE_INTERVAL == 0:
            try:
                with open(QTABLE_PATH, "wb") as f:
                    pickle.dump(Q_table, f)
                logger.info(f"Autosaved Q-table at step {current_simulation_step} ({len(Q_table)} states).")
            except Exception as e:
                logger.warning(f"Failed to autosave Q-table: {e}")
        if old_state not in Q_table:
            Q_table[old_state] = np.zeros(len(ACTIONS))
        old_q = Q_table[old_state][action]
        best_future_q = get_max_Q_value_of_state(new_state)
        Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)
    def get_action_from_policy(state):
        if random.random() < EPSILON:
            return random.choice(ACTIONS)
        else:
            if state not in Q_table:
                Q_table[state] = np.zeros(len(ACTIONS))
            return int(np.argmax(Q_table[state]))

    # ---------- SUMO paths ----------
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Set SUMO_HOME env variable.")
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

    base_dir = os.path.dirname(sys.executable) if getattr(sys,"frozen",False) else os.path.dirname(__file__)
    sumo_cfg = os.path.join(base_dir, "Sumo2Unity.sumocfg")
    sumo_bin = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [sumo_bin,"-c",sumo_cfg,"--step-length",str(steplength),
                "--lateral-resolution",str(lateral_resolution)]
    if use_gui:
        sumo_cmd += ["--delay","0"]

    traci.start(sumo_cmd)

    # GUI camera helper
    ego = "f_0.0"
    if use_gui and not free_cam:
        view_id = "View #0"
        traci.gui.trackVehicle(view_id, ego)
        traci.gui.setSchema(view_id, "real world")
    def cam_follow(view_id, veh_id):
        if free_cam: return
        try:
            traci.gui.trackVehicle(view_id, veh_id)
            traci.gui.setZoom(view_id, zoom_level)
        except traci.TraCIException:
            pass

    # TraCI context subscription
    traci.vehicle.subscribeContext(
        ego,
        traci.constants.CMD_GET_VEHICLE_VARIABLE,
        subscribe_radius,
        [VAR_POSITION3D, VAR_ANGLE, VAR_TYPE]
    )

    # ZMQ sockets
    ctx  = zmq.Context()
    pub  = ctx.socket(zmq.PUB);    pub.bind("tcp://*:5556")
    rout = ctx.socket(zmq.ROUTER); rout.bind("tcp://*:5557")

    # background Unity RX
    u_q = queue.Queue()
    def rx_unity():
        while True:
            try:
                _ident, msg = rout.recv_multipart()
                u_q.put(json.loads(msg.decode()))
            except Exception:
                logger.exception("Unity RX")
    threading.Thread(target=rx_unity, daemon=True).start()

    # helpers
    WINDOW = 10
    last_pos, last_pos_z, rw_hist = {}, {}, {}
    prof = {k: [] for k in ("Unity","Step","Collect","Send","DataProc","Total")}
    def sleep_precise(d):
        t0 = time.perf_counter()
        while (rem := d - (time.perf_counter()-t0)) > 0:
            if rem > 0.002: time.sleep(0.001)

    # results dir / RTF file
    if calc_rtf:
        res_dir = os.path.join(os.path.abspath(os.path.join(base_dir, os.pardir)), "Results")
        os.makedirs(res_dir, exist_ok=True)
        rtf_f = open(os.path.join(res_dir,"rtf_report.txt"),"w",encoding="utf-8")
        rtf_f.write("Time(s);RTF\n")
    else:
        rtf_f = None

    # telemetry send interval
    SEND_INTERVAL = 1.0  # seconds between telemetry updates
    last_tb_send = -1.0

    # fixed timing state
    fixed_phase_index = 0
    fixed_phase_timer = 0.0
    fixed_phase_durations = FIXED_PHASE_DURATIONS[:]  # seconds list
    # If SUMO has program with matching number of phases, we'll just cycle using indexes.
    logger.info(f"Fixed timing durations: {fixed_phase_durations}")

    # warm-up
    while traci.simulation.getTime() < IntegrationStartTime:
        traci.simulationStep(); cam_follow("View #0", ego) if use_gui else None

    # main loop
    STEP = steplength
    next_step = time.perf_counter() + STEP
    TL_INT = 1.0; last_tl_t = 0.0

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < ExperimentEndTime:
            loop_t0 = time.perf_counter()
            sim_t = traci.simulation.getTime()
            # advance RL step counter only when RL mode
            if not use_fixed_timing:
                current_simulation_step += 1

            # ❶ Unity → SUMO positions
            t0 = time.perf_counter()
            while not u_q.empty():
                for v in u_q.get().get("vehicles", []):
                    if v["vehicle_id"] == ego:
                        traci.vehicle.moveToXY(ego,"",0,
                            float(v["position"][0]), float(v["position"][1]),
                            float(v["angle"]), keepRoute=2)
            prof["Unity"].append(time.perf_counter()-t0)

            # ❷ SUMO step
            t0 = time.perf_counter(); traci.simulationStep()
            prof["Step"].append(time.perf_counter()-t0)
            if use_gui: cam_follow("View #0", ego)

            # If running RL mode, perform RL observation/action/update
            action = None
            reward = 0.0
            if not use_fixed_timing:
                # RL loop
                old_state = get_state()
                action = get_action_from_policy(old_state)
                apply_action(action)
            else:
                # Fixed timing loop: increment timer and switch when needed
                fixed_phase_timer += STEP
                try:
                    current_phase = traci.trafficlight.getPhase(TLS_ID)
                except Exception:
                    current_phase = fixed_phase_index
                # If SUMO phase index mismatches our internal index, sync it
                # but to avoid clashes, we'll compute next when timer exceeds duration
                if fixed_phase_timer >= fixed_phase_durations[fixed_phase_index % len(fixed_phase_durations)]:
                    # advance
                    fixed_phase_timer = 0.0
                    fixed_phase_index = (fixed_phase_index + 1) % len(fixed_phase_durations)
                    try:
                        traci.trafficlight.setPhase(TLS_ID, fixed_phase_index)
                    except Exception:
                        # ignore if setting phase fails (e.g. ID mismatch)
                        pass
                action = "fixed"

            # ❸ collect ego + context vehicles
            t0 = time.perf_counter()
            vlist = traci.vehicle.getIDList()
            vdata = []
            if ego in vlist:
                x,y,z   = traci.vehicle.getPosition3D(ego)
                ang     = traci.vehicle.getAngle(ego)
                vtype   = traci.vehicle.getTypeID(ego)
                vdata.append({"vehicle_id":ego,
                              "position":(round(x,2),round(y,2),round(z,2)),
                              "angle":round(ang,2),"type":vtype,
                              "timestamp":round(time.time(),2)})
                ctx_res = traci.vehicle.getContextSubscriptionResults(ego)
                if ctx_res:
                    for vid in ctx_res.keys():
                        if vid==ego: continue
                        x,y,z = traci.vehicle.getPosition3D(vid)
                        ang   = traci.vehicle.getAngle(vid)
                        vtype = traci.vehicle.getTypeID(vid)
                        vlong = traci.vehicle.getSpeed(vid)
                        vlat  = traci.vehicle.getLateralSpeed(vid)
                        if vid in last_pos_z:
                            pz,pt = last_pos_z[vid]; dt=sim_t-pt
                            vvert = (z-pz)/dt if dt>0 else 0.0
                        else: vvert=0.0
                        last_pos_z[vid]=(z,sim_t)
                        vdata.append({"vehicle_id":vid,
                                      "position":(round(x,3),round(y,3),round(z,3)),
                                      "angle":round(ang,3),"type":vtype,
                                      "long_speed":round(vlong,2),
                                      "vert_speed":round(vvert,3),
                                      "lat_speed":round(vlat,2)})
            vjson = json.dumps({"type":"vehicles","vehicles":vdata}, separators=(',',':'))
            prof["Collect"].append(time.perf_counter()-t0)

            # ❻ traffic lights once per second
            if sim_t-last_tl_t >= TL_INT:
                tls = [{"junction_id":tl,
                        "state":traci.trafficlight.getRedYellowGreenState(tl)}
                       for tl in traci.trafficlight.getIDList()]
                pub.send_string(json.dumps({"type":"trafficlights","lights":tls}, separators=(',',':')))
                last_tl_t = sim_t

            # ❼ publish vehicles
            t0 = time.perf_counter(); pub.send_string(vjson)
            prof["Send"].append(time.perf_counter()-t0)

            # Prepare telemetry periodically (rate-limited)
            try:
                phase_color = traci.trafficlight.getRedYellowGreenState(TLS_ID)
            except Exception:
                phase_color = ""
            try:
                phase_index = traci.trafficlight.getPhase(TLS_ID)
            except Exception:
                phase_index = (fixed_phase_index if use_fixed_timing else 0)

            detectors_payload = {
                "EB": [get_queue_length(d) for d in DETECTORS_EB],
                "WB": [get_queue_length(d) for d in DETECTORS_WB],
                "NB": [get_queue_length(d) for d in DETECTORS_NB],
                "SB": [get_queue_length(d) for d in DETECTORS_SB],
            }
            totals_payload = {
                "EB": sum(get_queue_length(d) for d in DETECTORS_EB),
                "WB": sum(get_queue_length(d) for d in DETECTORS_WB),
                "NB": sum(get_queue_length(d) for d in DETECTORS_NB),
                "SB": sum(get_queue_length(d) for d in DETECTORS_SB),
            }
            total_queue = sum(get_queue_length(d) for d in ALL_DETECTORS)
            num_vehicles = len(traci.vehicle.getIDList())
            avg_wait_time = (total_queue / num_vehicles) if num_vehicles > 0 else 0.0

            now_wall = time.perf_counter()
            if (sim_t - last_tb_send) >= SEND_INTERVAL:
                # Example lat/lon (replace with actual junction coords if you have them)
                lat, lon = 13.347312937036158, 74.7841435058439

                payload = {
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "junction_id": TLS_ID,
                    "latitude": lat,
                    "longitude": lon,
                    "sim_time": sim_t,
                    "phase_index": phase_index,
                    "phase_color": phase_color,
                    # flattened totals for charts
                    "total_EB": totals_payload["EB"],
                    "total_WB": totals_payload["WB"],
                    "total_NB": totals_payload["NB"],
                    "total_SB": totals_payload["SB"],
                    "total_queue": total_queue,
                    "avg_wait_time": avg_wait_time,
                    "reward": reward if not use_fixed_timing else 0.0,
                    "action": ("switch" if action == 1 else ("keep" if action == 0 else "fixed")),
                    "phase_duration_steps": (current_simulation_step - last_switch_step) if not use_fixed_timing else None
                }

                try:
                    tb_queue.put_nowait(payload)
                except queue.Full:
                    logger.warning("ThingsBoard queue full; dropping telemetry")
                last_tb_send = sim_t

            # RL: update Q table if RL mode
            if not use_fixed_timing:
                new_state = get_state()
                reward = get_reward(new_state)
                update_Q_table(old_state, action, reward, new_state)

            # incremental RTF
            if calc_rtf and rtf_started and sim_t >= ExperimentStartTime:
                now = time.perf_counter()
                if sim_t == ExperimentStartTime:
                    rtf_f.write("0.00;0.00\n")
                else:
                    sim_d  = sim_t - last_sim
                    real_d = now   - last_wall
                    rtf_f.write(f"{sim_t-ExperimentStartTime:.2f};{sim_d/real_d:.2f}\n")
                last_sim, last_wall = sim_t, now

            # pacing
            sleep_precise(max(0.0, next_step - time.perf_counter()))
            next_step += STEP
            prof["Total"].append(time.perf_counter()-loop_t0)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # save Q-table if RL
        if not use_fixed_timing:
            try:
                with open(QTABLE_PATH, "wb") as f:
                    pickle.dump(Q_table, f)
                logger.info(f"Final Q-table saved with {len(Q_table)} states to {QTABLE_PATH}")
            except Exception as e:
                logger.warning(f"Failed to save final Q-table: {e}")

        # overall RTF
        if calc_rtf and rtf_started:
            total_w   = time.perf_counter() - start_wall_t
            total_sim = traci.simulation.getTime() - start_sim_t
            if total_w>0: logger.info("RTF overall %.2f", total_sim/total_w)
        if start_rec_sent:
            pub.send_string(json.dumps({"type":"command","command":"STOP_RECORDING"}))
        if rtf_f: rtf_f.close()
        traci.close(); pub.close(); rout.close(); ctx.term()
        logger.info("Finished, connections closed.")

# ═════════════════════ GUI → START BTN ═════════════════════════
def start_clicked():
    try:
        cfg = {k: (int(v.get()) if "Time" in k else float(v.get())) for k,v in entries.items()}
        cfg["use_gui"]  = bool(use_gui_var.get())
        cfg["calc_rtf"] = bool(rtf_var.get())
        cfg["free_cam"] = bool(free_cam_var.get())
        cfg["use_fixed_timing"] = bool(fixed_timing_var.get())
    except ValueError:
        messagebox.showerror("Invalid input","Please enter numeric values.")
        return
    root.destroy(); run_sim(cfg)

# buttons
ttk.Button(root,text="Help",command=show_help)        .grid(row=row,column=0,pady=12,padx=6,sticky="w")
ttk.Button(root,text="Contact / License",command=show_contact)\
                                                     .grid(row=row,column=1,pady=12,padx=6,sticky="w")
ttk.Button(root,text="Publications",command=show_pubs)\
                                                     .grid(row=row,column=2,pady=12,padx=6,sticky="w")
ttk.Button(root,text="Start simulation",command=start_clicked)\
                                                     .grid(row=row,column=3,pady=12,padx=6,sticky="e")

root.update_idletasks()
root.geometry("+{}+{}".format((root.winfo_screenwidth()-root.winfo_width())//2,
                              (root.winfo_screenheight()-root.winfo_height())//2))
root.mainloop()
