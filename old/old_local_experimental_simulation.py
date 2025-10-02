# USE: nohup python local_experimental_simulation.py <map_nr> <num_agents> <intent_ver> <cooperative(0|1)> <game_version> <training_id> <checkpoint_number> [enable_video(true|false)] > local_simulation.log 2>&1 &

import time
import threading
import csv
import os
import sys
from pathlib import Path

import ray
import numpy as np
import cv2

from engine.app.ai_only_app import AIOnlySessionApp
from engine.extensions.renderer2d.renderer_2d_offline import Renderer2DOffline
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from spoiled_broth.game import SpoiledBroth as Game
from spoiled_broth.rl.rllib_controller import RLlibController
from spoiled_broth.rl.rllib_controller_lstm import RLlibControllerLSTM
from spoiled_broth.config import *

# Minimal, cleaned-up local simulation runner.
# Keeps: CLI args, controller loading, game factory cleanup, simple action/state CSVs,
# decision watcher that queries controllers and submits intents, offline renderer and
# optional video recording. Redundant/duplicated code and overly defensive blocks removed.

# --- CLI / params ---
if len(sys.argv) < 8:
    print("Usage: python local_experimental_simulation.py <map_nr> <num_agents> <intent_ver> <cooperative(0|1)> <game_version> <training_id> <checkpoint_number> [enable_video(true|false)]")
    sys.exit(1)

CLUSTER='cuenca'  # Options: 'brigit', 'local', 'cuenca'

training_map_nr = sys.argv[1]
num_agents = int(sys.argv[2])
intent_version = sys.argv[3]
cooperative = int(sys.argv[4])
game_version = str(sys.argv[5]).lower()
training_id = sys.argv[6]
checkpoint_number = int(sys.argv[7])
ENABLE_VIDEO_RECORDING = True if len(sys.argv) <= 8 or str(sys.argv[8]).lower() != "false" else False

# runtime constants (small, obvious defaults)
CF_ENGINE_TICK_RATE = 24
CF_AI_TICK_RATE = 1
FIXED_DURATION_SECONDS = 180
VIDEO_FPS = 24
# Desired agent movement speed in pixels/second for local experiments
AGENT_SPEED_PX_PER_SEC = 32

timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")

# paths
if CLUSTER == 'brigit':
    local = '/mnt/lustre/home/samuloza'
elif CLUSTER == 'cuenca':
    local = ''
elif CLUSTER == 'local':
    local = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
else:
    raise ValueError("Invalid cluster specified. Choose from 'brigit', 'cuenca', or 'local'.")
base_path = Path(f"{local}/data/samuel_lozano/cooked/{game_version}/{intent_version}/map_{training_map_nr}")
base_path = base_path / (f"cooperative/Training_{training_id}" if cooperative else f"competitive/Training_{training_id}")
path_root = Path().resolve() / "spoiled_broth"
checkpoint_dir = base_path / f"checkpoint_{checkpoint_number}"
saving_path = base_path / "simulations"
os.makedirs(saving_path, exist_ok=True)

# determine grid size from map (best-effort)
map_txt_path = path_root / "maps" / f"{training_map_nr}.txt"
grid_size = (8, 8)
try:
    with open(map_txt_path) as f:
        lines = [l.rstrip("\n") for l in f.readlines()]
    if lines:
        grid_size = (len(lines[0]), len(lines))
except Exception:
    pass

# --- Ray init ---
ray.shutdown()
ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)

# --- Controllers ---
use_lstm = False
cfg_path = base_path / "config.txt"
if cfg_path.exists():
    try:
        for line in open(cfg_path):
            if line.strip().startswith("USE_LSTM"):
                use_lstm = line.split(":", 1)[1].strip().lower() == "true"
    except Exception:
        pass

ControllerCls = RLlibControllerLSTM if use_lstm else RLlibController
controllers = {}
for i in range(1, num_agents + 1):
    aid = f"ai_rl_{i}"
    try:
        controllers[aid] = ControllerCls(aid, checkpoint_dir, f"policy_{aid}", competition=(game_version.upper() == "COMPETITION"))
    except Exception as e:
        print(f"Warning: couldn't init controller {aid}: {e}")


# --- Simple CSV loggers ---
state_csv = saving_path / f"simulation_log_checkpoint_{checkpoint_number}_{timestamp}.csv"
action_csv = saving_path / f"actions_checkpoint_{checkpoint_number}_{timestamp}.csv"

class SimpleStateLogger:
    def __init__(self, path):
        self.path = path
        with open(self.path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["agent", "frame", "second", "x", "y", "tile_x", "tile_y", "item", "score"])
    def log(self, frame, game):
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            for aid, obj in game.gameObjects.items():
                if aid.startswith('ai_rl_'):
                    writer.writerow([aid, frame, frame/CF_ENGINE_TICK_RATE, getattr(obj, 'x', 0), getattr(obj, 'y', 0), float(getattr(obj, 'x', 0)) // 16 + 1, float(getattr(obj, 'y', 0)) // 16 + 1, getattr(obj, 'item', ''), getattr(obj, 'score', 0)])

class CompatibleActionTracker:
    """Action tracker that's compatible with existing RLlib controller calls."""
    
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        # Track active actions: agent_id -> action_record
        self.active_actions = {}
        # Minimum duration before action can be considered complete (in frames)
        self.MIN_ACTION_DURATION_FRAMES = 6  # ~0.25 seconds at 24 FPS
        # Track when actions can be completed
        self.action_completion_times = {}
        
        with open(self.path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'agent_id', 'decision_frame', 'completion_frame', 
                'duration_frames', 'action_number', 'action_type', 'target_tile_type', 
                'target_tile_x', 'target_tile_y',  'decision_x', 'decision_y', 'decision_tile_x', 
                'decision_tile_y', 'completion_x', 'completion_y', 'completion_tile_x', 
                'completion_tile_y', 'item_before', 'item_after'
            ])
    
    def _get_tile_info(self, game, target_id):
        """Return (tile_type, x_tile, y_tile) for a given target_id."""
        tile_type, x_tile, y_tile = '', None, None
        try:
            grid = getattr(game, 'grid', None)
            if grid is not None and hasattr(grid, 'tiles'):
                for x in range(grid.width):
                    for y in range(grid.height):
                        t = grid.tiles[x][y]
                        if t and getattr(t, 'id', None) == target_id:
                            tile_type = t.__class__.__name__.lower()
                            x_tile, y_tile = x + 1, y + 1
                            return tile_type, x_tile, y_tile
        except Exception:
            pass
        return tile_type, x_tile, y_tile
    
    def _extract_action_info(self, action, game):
        """Extract action type and target information."""
        action_type = ''
        target_id = ''
        tile_type = ''
        x_tile = None
        y_tile = None
        
        try:
            if action is None:
                return 'do_nothing', '', '', None, None
            
            if isinstance(action, dict):
                action_type = str(action.get('type', action.get('action', 'unknown')))
                target_id = str(action.get('target', ''))
            elif hasattr(action, 'action_type') or hasattr(action, 'type'):
                action_type = str(getattr(action, 'action_type', getattr(action, 'type', 'unknown')))
                target_id = str(getattr(action, 'target', ''))
            else:
                action_type = str(action)
            
            if target_id:
                tile_type, x_tile, y_tile = self._get_tile_info(game, target_id)
                
        except Exception as e:
            action_type = 'unknown'
            print(f"DEBUG: Error extracting action info: {e}")
            
        return action_type, target_id, tile_type, x_tile, y_tile
    
    def start_action(self, agent_id, action_type, tile_or_target, timestamp, action_number=None, **kwargs):
        """Called by RLlib controllers when an action starts - compatible with old API."""
        with self.lock:
            # Get the game and engine from the tracker's references
            game = getattr(self, 'game', None)
            engine = getattr(self, 'engine', None)
            
            if game is None:
                print(f"[ACTION_TRACKER] No game reference available for {agent_id}")
                return  # Can't track without game reference
                
            if engine is None:
                print(f"[ACTION_TRACKER] No engine reference available for {agent_id}")
                return  # Can't track without engine reference
            
            # Get current frame and agent state
            current_frame = getattr(engine, 'tick_count', 0)
            agent_obj = game.gameObjects.get(agent_id)
            
            if agent_obj is None:
                print(f"[ACTION_TRACKER] No agent object found for {agent_id}")
                return
            
            # Create action dict for processing
            if action_type == "click" and tile_or_target:
                action = {"type": "click", "target": getattr(tile_or_target, 'id', str(tile_or_target))}
            elif action_type == "do_nothing":
                action = None
            else:
                action = {"type": action_type}
            
            # Extract action information
            action_type_str, target_id, tile_type, x_tile, y_tile = self._extract_action_info(action, game)
            
            # Get agent current state
            agent_x = getattr(agent_obj, 'x', None)
            agent_y = getattr(agent_obj, 'y', None)
            item_before = getattr(agent_obj, 'item', None)
            
            # Create action record
            action_record = {
                'action': action,
                'action_type': action_type_str,
                'target_id': target_id,
                'tile_type': tile_type,
                'target_x_tile': x_tile,
                'target_y_tile': y_tile,
                'decision_frame': current_frame,
                'decision_x': agent_x,
                'decision_y': agent_y,
                'item_before': item_before,
                'action_number': action_number if action_number is not None else getattr(self, f'_action_counter_{agent_id}', 0) + 1
            }
            
            # Increment action counter for this agent
            if action_number is None:
                setattr(self, f'_action_counter_{agent_id}', action_record['action_number'])
            
            # Store active action - use agent_id as key to avoid conflicts
            self.active_actions[agent_id] = action_record
            
            # Set minimum completion time based on action type
            min_duration = self.MIN_ACTION_DURATION_FRAMES
            if action_type_str == "do_nothing":
                min_duration = 6  # Shorter for do_nothing actions
            elif action_type_str == "click":
                min_duration = 24  # Longer for click actions to allow for actual gameplay
            
            # Record when this action can be completed earliest
            self.action_completion_times[agent_id] = current_frame + min_duration
                
    def can_complete_action(self, agent_id, current_frame):
        """Check if an action has been running long enough to be completed."""
        if agent_id not in self.action_completion_times:
            return True  # No restriction if not tracked
        return current_frame >= self.action_completion_times[agent_id]
    
    def end_action(self, agent_id, timestamp=None, **kwargs):
        """Called by RLlib controllers when an action ends - compatible with old API."""
        with self.lock:
            if agent_id not in self.active_actions:
                return False  # No action to end
            
            action_record = self.active_actions[agent_id]
            
            # Get the game and engine from the tracker's references
            game = getattr(self, 'game', None)
            engine = getattr(self, 'engine', None)
            
            if game is None or engine is None:
                # Can't complete tracking without game/engine, but remove from active actions
                self.active_actions.pop(agent_id, None)
                self.action_completion_times.pop(agent_id, None)
                return False
            
            # Get completion state
            current_frame = getattr(engine, 'tick_count', 0)
            
            # Check if enough time has passed for this action to complete
            if not self.can_complete_action(agent_id, current_frame):
                # Silently return False - action is not ready to complete yet
                return False
            
            agent_obj = game.gameObjects.get(agent_id)
            
            if agent_obj is None:
                self.active_actions.pop(agent_id, None)
                self.action_completion_times.pop(agent_id, None)
                return False
            
            # Get agent completion state
            agent_x = getattr(agent_obj, 'x', None)
            agent_y = getattr(agent_obj, 'y', None)
            item_after = getattr(agent_obj, 'item', None)
            
            # Calculate duration
            duration = current_frame - action_record['decision_frame']
            
            # Ensure non-negative duration and minimum duration
            if duration < 0:
                print(f"Warning: Negative duration for {agent_id}, frame went backwards!")
                duration = 0
                current_frame = action_record['decision_frame']
            elif duration == 0:
                # Force minimum duration of 1 frame
                duration = 1
                current_frame = action_record['decision_frame'] + 1
            
            # Write to CSV
            row = [
                agent_id,
                action_record['decision_frame'],
                current_frame,
                duration,
                action_record['action_number'],
                action_record['action_type'],
                action_record['tile_type'],
                action_record['target_x_tile'],
                action_record['target_y_tile'],
                action_record['decision_x'],
                action_record['decision_y'],
                int(action_record['decision_x']//16+1 if action_record['decision_x'] is not None else 0),
                int(action_record['decision_y']//16+1 if action_record['decision_y'] is not None else 0),
                agent_x,
                agent_y,
                int(agent_x//16+1 if agent_x is not None else 0),
                int(agent_y//16+1 if agent_y is not None else 0),
                action_record['item_before'],
                item_after
            ]
            
            try:
                with open(self.path, 'a', newline='') as f:
                    csv.writer(f).writerow(row)
            except Exception as e:
                print(f"[ACTION_TRACKER] Error writing action log: {e}")
            
            # Remove from active actions and completion times
            self.active_actions.pop(agent_id, None)
            self.action_completion_times.pop(agent_id, None)
            return True  # Action successfully completed
            
    def _force_complete_action(self, agent_id, game, engine):
        """Force complete an action (for cleanup) - called within existing lock."""
        if agent_id not in self.active_actions:
            return
            
        action_record = self.active_actions[agent_id]
        
        # Get completion state
        current_frame = getattr(engine, 'tick_count', 0) if engine else action_record['decision_frame']
        agent_obj = game.gameObjects.get(agent_id) if game else None
        
        # Get agent completion state
        agent_x = getattr(agent_obj, 'x', None) if agent_obj else None
        agent_y = getattr(agent_obj, 'y', None) if agent_obj else None
        item_after = getattr(agent_obj, 'item', None) if agent_obj else None
        
        # Calculate duration
        duration = current_frame - action_record['decision_frame']
        if duration < 0:
            duration = 0
            current_frame = action_record['decision_frame']
        
        # Write to CSV (forced completion)
        row = [
            agent_id,
            action_record['decision_frame'],
            current_frame,
            duration,
            action_record['action_number'],
            action_record['action_type'],
            action_record['tile_type'],
            action_record['target_x_tile'],
            action_record['target_y_tile'],
            action_record['decision_x'],
            action_record['decision_y'],
            int(action_record['decision_x']//16+1 if action_record['decision_x'] is not None else 0),
            int(action_record['decision_y']//16+1 if action_record['decision_y'] is not None else 0),
            agent_x,
            agent_y,
            int(agent_x//16+1 if agent_x is not None else 0),
            int(agent_y//16+1 if agent_y is not None else 0),
            action_record['item_before'],
            item_after
        ]
        
        try:
            with open(self.path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            print(f"[ACTION_TRACKER] Error writing forced completion log: {e}")
        
        # Remove from active actions
        self.active_actions.pop(agent_id, None)
        self.action_completion_times.pop(agent_id, None)
    
    def cleanup(self, game, engine):
        """Complete any remaining active actions at simulation end."""
        print(f"[ACTION_TRACKER] Starting cleanup for {len(self.active_actions)} active actions")
        sys.stdout.flush()
        
        cleanup_start_time = time.time()
        max_cleanup_time = 3.0  # Maximum 3 seconds for cleanup
        
        try:
            # Use a timeout-aware lock acquisition
            lock_acquired = self.lock.acquire(timeout=2.0)
            if not lock_acquired:
                print(f"[ACTION_TRACKER] Could not acquire lock for cleanup, force clearing actions")
                self.active_actions.clear()
                if hasattr(self, 'action_completion_times'):
                    self.action_completion_times.clear()
                sys.stdout.flush()
                return
                
            try:
                active_agents = list(self.active_actions.keys())
                print(f"[ACTION_TRACKER] Active agents to clean up: {active_agents}")
                sys.stdout.flush()
                
                for agent_id in active_agents:
                    # Check timeout
                    if time.time() - cleanup_start_time > max_cleanup_time:
                        print(f"[ACTION_TRACKER] Cleanup timeout reached, force clearing remaining actions")
                        self.active_actions.clear()
                        if hasattr(self, 'action_completion_times'):
                            self.action_completion_times.clear()
                        sys.stdout.flush()
                        break
                        
                    print(f"[ACTION_TRACKER] Cleaning up action for {agent_id}")
                    sys.stdout.flush()
                    try:
                        self._force_complete_action(agent_id, game, engine)
                        print(f"[ACTION_TRACKER] Completed cleanup for {agent_id}")
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"[ACTION_TRACKER] Error cleaning up {agent_id}: {e}")
                        # Force remove the action
                        self.active_actions.pop(agent_id, None)
                        if hasattr(self, 'action_completion_times'):
                            self.action_completion_times.pop(agent_id, None)
                        sys.stdout.flush()
                        
                print(f"[ACTION_TRACKER] Cleanup completed. Remaining active actions: {len(self.active_actions)}")
                sys.stdout.flush()
            finally:
                self.lock.release()
                
        except Exception as e:
            print(f"[ACTION_TRACKER] Critical error during cleanup: {e}")
            # Clear all actions as a last resort
            self.active_actions.clear()
            if hasattr(self, 'action_completion_times'):
                self.action_completion_times.clear()
            sys.stdout.flush()

state_logger = SimpleStateLogger(state_csv)
action_tracker = CompatibleActionTracker(action_csv)

# --- Video recorder ---
class VideoRecorder:
    def __init__(self, out_path, fps=VIDEO_FPS):
        self.out_path = str(out_path)
        self.fps = fps
        self.writer = None
        self.size = None
    def start(self, frame):
        src = frame
        if src.dtype != np.uint8:
            src = (np.clip(src,0.0,1.0)*255).astype(np.uint8)
        if src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        h,w = src.shape[:2]
        fourcc_candidates = ['mp4v','avc1','H264','XVID','MJPG']
        # try candidates until one opens
        for code in fourcc_candidates:
            fourcc = cv2.VideoWriter_fourcc(*code)
            self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w,h))
            try:
                ok = self.writer.isOpened()
            except Exception:
                ok = False
            if ok:
                # store size as (w,h) for resizing later
                self.size = (w, h)
                break
            else:
                # release and try next
                try:
                    self.writer.release()
                except Exception:
                    pass
                self.writer = None
        if self.writer is None:
            print(f"Warning: could not open VideoWriter for {self.out_path}; video will be disabled")
    def write(self, frame):
        if self.writer is None:
            # attempt to (re)start writer
            self.start(frame)
        if self.writer is None:
            return
        try:
            src = frame
            if src.dtype != np.uint8:
                src = (np.clip(src,0.0,1.0)*255).astype(np.uint8)
            if src.ndim == 2:
                src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            # resize if needed
            if self.size is not None:
                w,h = self.size
                if (src.shape[1], src.shape[0]) != (w,h):
                    src = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
            if not self.writer.write(src):
                # OpenCV's VideoWriter.write doesn't always return a value; detect via isOpened
                if not getattr(self.writer, 'isOpened', lambda: True)():
                    print(f"Warning: VideoWriter not open for {self.out_path}")
        except Exception as e:
            print(f"Warning: failed to write video frame: {e}")
    def stop(self):
        if self.writer:
            self.writer.release()

# --- Game factory (light cleanup) ---
def game_factory():
    game = Game(map_nr=training_map_nr, grid_size=grid_size, intent_version=intent_version)
    # clear items on non-dispenser tiles and reset agent hand/state
    try:
        grid = getattr(game, 'grid', None)
        if grid is not None:
            from spoiled_broth.world.tiles import Dispenser
            for x in range(grid.width):
                for y in range(grid.height):
                    t = grid.tiles[x][y]
                    if t is None:
                        continue
                    if hasattr(t, 'item') and not isinstance(t, Dispenser):
                        t.item = None
                    if hasattr(t, 'cut_stage'):
                        t.cut_stage = 0
        for aid,obj in list(game.gameObjects.items()):
            if aid.startswith('ai_rl_'):
                if hasattr(obj,'item'):
                    obj.item = None
                if hasattr(obj,'action_complete'):
                    obj.action_complete = True
                if hasattr(obj,'current_action'):
                    obj.current_action = None
    except Exception:
        pass
    return game

# Instantiate AI-only session with controllers directly
engine_app = AIOnlySessionApp(game_factory=game_factory, ui_modules=[Renderer2DModule()], agent_map=controllers, path_root=path_root, tick_rate=CF_ENGINE_TICK_RATE, ai_tick_rate=CF_AI_TICK_RATE, is_max_speed=False)
session = engine_app.get_session()
game = session.engine.game

# Attach the compatible action tracker to the game object for RLlib controllers
try:
    game.action_tracker = action_tracker
    # Give the action tracker references to both game and engine
    action_tracker.game = game
    action_tracker.engine = session.engine
    print(f"Action tracker attached to game object, game id: {id(game)}")
    print(f"Action tracker game reference: {id(getattr(action_tracker, 'game', None))}")
    print(f"Action tracker engine reference: {id(getattr(action_tracker, 'engine', None))}")
except Exception as e:
    print(f"Failed to attach action tracker: {e}")

# Verify agents were added and attach them to controllers
print(f"Added {len(game.gameObjects)} game objects to the session")
for aid, ctrl in controllers.items():
    agent_obj = game.gameObjects.get(aid)
    if agent_obj is not None:
        ctrl.agent = agent_obj
        print(f"Successfully attached controller to agent {aid}")
    else:
        print(f"Warning: Agent {aid} not found in game objects")

agent_map = controllers

# Ensure agents have the configured speed
for aid, obj in list(game.gameObjects.items()):
    if aid.startswith('ai_rl_') and obj is not None:
        try:
            obj.speed = AGENT_SPEED_PX_PER_SEC
        except Exception:
            pass

# --- Rendering + main loop ---
total_ticks = FIXED_DURATION_SECONDS * CF_ENGINE_TICK_RATE
total_frames = total_ticks

def serialize_ui_state():
    payload = {}
    for m in session.ui_modules:
        payload.update(m.serialize_for_agent(game, session.engine, None))
    payload['tick'] = session.engine.tick_count
    return payload

renderer = Renderer2DOffline(path_root=path_root, tile_size=16)
recorder = None
prev_state = None

# Start the engine session - it will run in its own thread
try:
    session.start()
    print("Session started successfully")
    sys.stdout.flush()
except RuntimeError as e:
    if "threads can only be started once" in str(e):
        print("Session already started, continuing...")
        sys.stdout.flush()
    else:
        raise e

# Wait for the engine to start up
print("Waiting for engine to start up...")
sys.stdout.flush()
time.sleep(0.5)

progress_step = max(1, int(total_frames * 0.05))
next_progress = progress_step

print(f"Starting simulation with {total_frames} frames ({FIXED_DURATION_SECONDS} seconds)")
sys.stdout.flush()

# Calculate the frame interval based on the desired frame rate
frame_interval = 1.0 / CF_ENGINE_TICK_RATE
start_time = time.time()

for frame_idx in range(total_frames):
    # Wait for the right time for this frame
    target_time = start_time + frame_idx * frame_interval
    current_time = time.time()
    if current_time < target_time:
        time.sleep(target_time - current_time)
    
    curr_state = serialize_ui_state()
    prev_state = prev_state or curr_state
    frame = renderer.render_to_array(prev_state, curr_state, 0.0)
    state_logger.log(frame_idx, game)

    if ENABLE_VIDEO_RECORDING:
        # prepare HUD: compute coop score and per-agent scores of ai_rl_1 and ai_rl_2
        coop_score = 0
        scores = {}
        try:
            for aid, obj in game.gameObjects.items():
                if aid.startswith('ai_rl_') and obj is not None:
                    s = int(getattr(obj, 'score', 0) or 0)
                    coop_score += s
                    scores[aid] = s
        except Exception:
            pass

        # Compose HUD text
        hud_lines = [f"Coop: {coop_score}"]
        # Show explicit ai_rl_1 and ai_rl_2 (if exist), else show available agents
        for label in ('ai_rl_1','ai_rl_2'):
            hud_lines.append(f"{label}: {scores.get(label, 0)}")

        # draw into a padded canvas (bottom area)
        if recorder is None:
            recorder = VideoRecorder(saving_path / f"offline_recording_{timestamp}.mp4")

        try:
            src = frame
            if src.dtype != np.uint8:
                src = (np.clip(src,0.0,1.0)*255).astype(np.uint8)
            if src.ndim == 2:
                src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            h, w = src.shape[:2]
            # font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 1
            line_h = cv2.getTextSize('Tg', font, scale, thickness)[0][1] + 6
            pad_h = max(48, 12 + len(hud_lines) * line_h)
            canvas = np.zeros((h + pad_h, w, 3), dtype=np.uint8)
            canvas[0:h, 0:w] = src
            # draw HUD lines
            start_y = h + 12 + line_h
            # draw a darker background strip for readability
            cv2.rectangle(canvas, (0, h), (w, h + pad_h), (20,20,20), -1)
            for i, txt in enumerate(hud_lines):
                y = start_y + i * line_h
                # shadow
                cv2.putText(canvas, txt, (8, y), font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
                cv2.putText(canvas, txt, (8, y), font, scale, (255,255,255), thickness, cv2.LINE_AA)
            recorder.write(canvas)
        except Exception:
            try:
                recorder.write(frame)
            except Exception:
                pass
    prev_state = curr_state
    # Print progress every 5%
    if frame_idx + 1 >= next_progress:
        percent = int(100 * (frame_idx + 1) / total_frames)
        print(f"Simulation progress: {percent}% ({frame_idx + 1}/{total_frames} frames)")
        sys.stdout.flush()  # Force output to appear immediately
        next_progress += progress_step

# cleanup
print("Stopping video recorder...")
sys.stdout.flush()
if recorder:
    recorder.stop()

print("Cleaning up actions...")
sys.stdout.flush()
# Complete any remaining actions
try:
    print(f"Active actions before cleanup: {list(action_tracker.active_actions.keys())}")
    sys.stdout.flush()
    action_tracker.cleanup(game, session.engine)
    print("Action cleanup completed.")
    sys.stdout.flush()
except Exception as e:
    print(f"Error during action cleanup: {e}")
    sys.stdout.flush()

print("Stopping engine...")
sys.stdout.flush()
try:
    # Simple approach: try engine stop with a short timeout using threading
    import threading
    
    engine_stop_completed = threading.Event()
    engine_stop_result = {"error": None}
    
    def stop_engine():
        try:
            engine_app.stop()
            engine_stop_completed.set()
        except Exception as e:
            engine_stop_result["error"] = e
            engine_stop_completed.set()
    
    # Start engine stop in a separate thread
    stop_thread = threading.Thread(target=stop_engine)
    stop_thread.daemon = True  # Allow main process to exit even if this hangs
    stop_thread.start()
    
    # Wait up to 5 seconds for engine to stop
    if engine_stop_completed.wait(timeout=5.0):
        if engine_stop_result["error"]:
            print(f"Error stopping engine: {engine_stop_result['error']}")
        else:
            print("Engine stopped.")
        sys.stdout.flush()
    else:
        print("Engine stop timed out after 5 seconds, continuing...")
        sys.stdout.flush()
        
except Exception as e:
    print(f"Error during engine stop process: {e}")
    sys.stdout.flush()

print("Simulation finished")
print(f"State CSV: {state_csv}")
print(f"Action CSV: {action_csv}")
sys.stdout.flush()  # Force final output to appear immediately
