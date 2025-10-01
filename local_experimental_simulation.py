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
AGENT_SPEED_PX_PER_SEC = 100

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

class ActionTracker:
    """Track performed actions per-agent by watching agent.current_action and action_complete.

    Each frame we inspect agents; when current_action changes from previous observed value
    we record a start entry. When the agent's action becomes complete (or None) we
    record an end entry. Times are frames (tick counts are deduced from session.engine).
    """
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        # active: agent_id -> record
        self.active = {}
        # track last submission ticks we've consumed per agent to avoid re-starting
        self.seen_submission_tick = {}
        with open(self.path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                    'agent_id', 'action_type', 'target_tile_type', 'target_tile_x', 'target_tile_y',
                    'start_frame', 'end_frame', 'duration_frames',
                    'start_x', 'start_y', 'start_tile_x', 'start_tile_y',
                    'end_x', 'end_y', 'end_tile_x', 'end_tile_y',
                    'target_x_tile', 'target_y_tile',
                    'item_on_hand_start', 'item_on_hand_end'
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


    def _extract_action_meta(self, action_sig, game):
        """Return (action_type, target_id, target_tile_type, target_x_tile, target_y_tile)."""
        action_type = ''
        target_id = ''
        tile_type = ''
        x_tile = None
        y_tile = None
        try:
            if action_sig is None:
                return (None, '', None, '', None, None)
            if isinstance(action_sig, (list, tuple)) and len(action_sig) >= 2:
                action_type = str(action_sig[0])
                target_id = str(action_sig[1]) if action_sig[1] is not None else ''
            else:
                action_type = str(action_sig)
            tile_type, x_tile, y_tile = self._get_tile_info(game, target_id)
        except Exception:
            pass
        return (action_type, target_id, tile_type, x_tile, y_tile)

    # In start_action, store item_on_hand_start:
    def start_action(self, agent_id, action_type, target, timestamp=None, action_number=None):
        with self.lock:
            if agent_id in self.active:
                return
            target_id = ''
            target_type = ''
            try:
                if target is None:
                    target_id = ''
                elif hasattr(target, 'id'):
                    target_id = str(getattr(target, 'id', ''))
                else:
                    target_id = str(target)
            except Exception:
                target_id = str(target)
            start_frame = None
            try:
                start_frame = int(getattr(session.engine, 'tick_count', None))
            except Exception:
                start_frame = None
            start_x = None
            start_y = None
            item_on_hand_start = None
            try:
                g = globals().get('game')
                if g is not None:
                    agent_obj = g.gameObjects.get(agent_id)
                    if agent_obj is not None:
                        start_x = getattr(agent_obj, 'x', None)
                        start_y = getattr(agent_obj, 'y', None)
                        item_on_hand_start = getattr(agent_obj, 'item', None)
            except Exception:
                pass
            sig = (str(action_type), target_id)
            self.active[agent_id] = {
                'action_sig': sig,
                'start_frame': start_frame,
                'start_x': start_x,
                'start_y': start_y,
                'start_ts': timestamp,
                'action_number': action_number,
                'item_on_hand_start': item_on_hand_start,
            }

    def end_action(self, agent_id, timestamp=None, game=None):
        with self.lock:
            if agent_id not in self.active:
                return
            rec = self.active.get(agent_id)
            try:
                end_frame = int(getattr(session.engine, 'tick_count', None))
            except Exception:
                end_frame = None
            start_frame = rec.get('start_frame')
            if start_frame is None and end_frame is None:
                start_frame = 0
                end_frame = 0
            elif start_frame is None:
                start_frame = end_frame
            elif end_frame is None:
                end_frame = start_frame
            duration = end_frame - start_frame
            
            # Add safety check for negative duration
            if duration < 0:
                print(f"Warning: Negative duration detected for {agent_id}, adjusting...")
                duration = 0
                end_frame = start_frame
                
            if start_frame == end_frame and duration == 0:
                # Only record if frames are different or there's duration
                self.active.pop(agent_id, None)
                self.seen_submission_tick.pop(agent_id, None)
                return
            action_type, target_id, tile_type, x_tile, y_tile = self._extract_action_meta(rec.get('action_sig'), game or globals().get('game'))
            start_x = rec.get('start_x')
            start_y = rec.get('start_y')
            end_x = None
            end_y = None
            item_on_hand_start = None
            item_on_hand_end = None
            try:
                g = game or globals().get('game')
                if g is not None:
                    agent_obj = g.gameObjects.get(agent_id)
                    if agent_obj is not None:
                        end_x = getattr(agent_obj, 'x', None)
                        end_y = getattr(agent_obj, 'y', None)
                        item_on_hand_end = getattr(agent_obj, 'item', None)
                    # item at start
                    item_on_hand_start = rec.get('item_on_hand_start')
            except Exception:
                pass
            row = [
                agent_id, action_type, tile_type, x_tile, y_tile,
                int(start_frame), int(end_frame), int(duration),
                start_x, start_y, int(start_x//16+1 if start_x is not None else 0), int(start_y//16+1 if start_y is not None else 0),
                end_x, end_y, int(end_x//16+1 if end_x is not None else 0), int(end_y//16+1 if end_y is not None else 0),
                x_tile, y_tile,
                item_on_hand_start, item_on_hand_end
            ]
            try:
                with open(self.path, 'a', newline='') as f:
                    csv.writer(f).writerow(row)
            except Exception:
                pass
            self.active.pop(agent_id, None)
            self.seen_submission_tick.pop(agent_id, None)

    def _force_end(self, agent_id, timestamp=None, reason=None):
        """Force-end an action (used by Agent timeout handler)."""
        # reason currently unused but kept for compatibility
        self.end_action(agent_id, timestamp)

    def _action_signature(self, action):
        try:
            # prefer a concise signature
            if action is None:
                return None
            if isinstance(action, dict):
                target = action.get('target')
                return (action.get('type') or action.get('action') or str(action), str(target))
            # object-like
            target = getattr(action, 'target', None)
            atype = getattr(action, 'action_type', None) or getattr(action, 'type', None) or repr(action)
            return (str(atype), str(target))
        except Exception:
            return (repr(action), None)

    def update(self, frame_idx, game, engine, last_submissions=None, last_submission_ticks=None):
        """Call once per rendered frame to detect starts/ends. Only log action end when a new action starts."""
        with self.lock:
            for aid, obj in list(game.gameObjects.items()):
                if not aid.startswith('ai_rl_'):
                    continue
                cur = getattr(obj, 'current_action', None)
                sig = self._action_signature(cur)
                if sig is None and last_submissions is not None and last_submission_ticks is not None:
                    last = last_submissions.get(aid)
                    last_tick = last_submission_ticks.get(aid)
                    if last is not None and last_tick is not None and self.seen_submission_tick.get(aid) != last_tick:
                        sig = self._action_signature(last)

                active = self.active.get(aid)

                # If there is an active record and a new action is starting, log the previous action end
                if active is not None and sig is not None and active.get('action_sig') != sig:
                    item_on_hand_end = getattr(obj, 'item', None)
                    action_sig = active.get('action_sig')
                    action_type, target_id, tile_type, x_tile, y_tile = self._extract_action_meta(action_sig, game)
                    end_x = getattr(obj, 'x', None)
                    end_y = getattr(obj, 'y', None)
                    # Fix: End the previous action at the frame before the new one starts
                    end_frame = max(frame_idx - 1, active.get('start_frame', 0))
                    start_frame = int(active.get('start_frame', frame_idx))
                    duration = end_frame - start_frame
                    # Ensure duration is never negative
                    if duration < 0:
                        duration = 0
                        end_frame = start_frame

                    if start_frame != end_frame or duration > 0:  # Only log if there's actual duration
                        row = [
                            aid, str(action_type), tile_type, x_tile, y_tile,
                            int(start_frame), int(end_frame), int(duration),
                            active.get('start_x'), active.get('start_y'),
                            int(active.get('start_x')//16+1 if active.get('start_x') is not None else 0),
                            int(active.get('start_y')//16+1 if active.get('start_y') is not None else 0),
                            end_x, end_y,
                            int(end_x//16+1 if end_x is not None else 0),
                            int(end_y//16+1 if end_y is not None else 0),
                            x_tile, y_tile,
                            active.get('item_on_hand_start'), item_on_hand_end
                        ]
                        try:
                            with open(self.path, 'a', newline='') as f:
                                csv.writer(f).writerow(row)
                        except Exception:
                            pass
                    self.active.pop(aid, None)
                    self.seen_submission_tick.pop(aid, None)

                # If there's no active record and we see a non-empty action, start it
                if active is None and sig is not None:
                    start_x = getattr(obj, 'x', None)
                    start_y = getattr(obj, 'y', None)
                    item_on_hand_start = getattr(obj, 'item', None)
                    self.active[aid] = {
                        'action_sig': sig,
                        'start_frame': frame_idx,
                        'start_x': start_x,
                        'start_y': start_y,
                        'item_on_hand_start': item_on_hand_start,
                    }
                    if last_submission_ticks is not None:
                        last_tick = last_submission_ticks.get(aid)
                        if last_tick is not None:
                            self.seen_submission_tick[aid] = last_tick

state_logger = SimpleStateLogger(state_csv)
action_tracker = ActionTracker(action_csv)

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

# Instantiate AI-only session (empty agent_map; we'll add agents manually)
engine_app = AIOnlySessionApp(game_factory=game_factory, ui_modules=[Renderer2DModule()], agent_map={}, path_root=path_root, tick_rate=CF_ENGINE_TICK_RATE, ai_tick_rate=CF_AI_TICK_RATE, is_max_speed=False)
session = engine_app.get_session()
game = session.engine.game

# Attach trackers/loggers to the live game object for use by controllers/agents
try:
    game.action_tracker = action_tracker
except Exception:
    pass

# add agents and attach controllers
for aid, ctrl in controllers.items():
    try:
        session.add_agent(aid)
        ctrl.agent = game.gameObjects.get(aid)
    except Exception as e:
        print(f"Warning: couldn't add agent {aid}: {e}")

agent_map = controllers

# Ensure agents have the configured speed
for aid, obj in list(game.gameObjects.items()):
    if aid.startswith('ai_rl_') and obj is not None:
        try:
            obj.speed = AGENT_SPEED_PX_PER_SEC
        except Exception:
            pass

# --- Decision watcher (small, readable) ---
last_submission_tick = {}
last_submission_action = {}
_decision_running = True

def decision_watcher():
    poll = 0.01
    methods = ('request_action','choose_action','act')
    while _decision_running:
        try:
            for aid, ctrl in list(agent_map.items()):
                agent_obj = game.gameObjects.get(aid)
                if agent_obj is None:
                    continue
                finished = getattr(agent_obj,'action_complete',True)
                current = getattr(agent_obj,'current_action',None)
                eng_tick = getattr(session.engine,'tick_count',None)
                last_tick = last_submission_tick.get(aid)
                if not (finished is True or current is None):
                    continue
                if last_tick is not None and eng_tick is not None and eng_tick <= last_tick:
                    continue
                # skip if engine has buffered intent
                ib = getattr(session.engine,'intent_buffer',None)
                if ib is not None and aid in ib:
                    continue

                action = None
                for m in methods:
                    fn = getattr(ctrl,m,None)
                    if not callable(fn):
                        continue
                    try:
                        action = fn(game, aid)
                    except TypeError:
                        try:
                            action = fn(aid, game)
                        except Exception:
                            try:
                                action = fn()
                            except Exception:
                                action = None
                    if action is not None:
                        try:
                            session.engine.submit_intent(aid, action)
                            last_submission_tick[aid] = getattr(session.engine,'tick_count',None)
                            last_submission_action[aid] = action
                        except Exception as e:
                            print(f"Failed to submit intent for {aid}: {e}")
                        break
        except Exception as e:
            print(f"Decision watcher error: {e}")
        time.sleep(poll)

if agent_map:
    _watcher = threading.Thread(target=decision_watcher, daemon=True)
    _watcher.start()

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

progress_step = max(1, int(total_frames * 0.05))
next_progress = progress_step
for frame_idx in range(total_frames):
    curr_state = serialize_ui_state()
    prev_state = prev_state or curr_state
    frame = renderer.render_to_array(prev_state, curr_state, 0.0)
    state_logger.log(frame_idx, game)
    # update action tracker (detect action starts/ends). Pass the last_submission_action
    # so actions that were submitted but not yet reflected in current_action are captured.
    action_tracker.update(frame_idx, game, session.engine, last_submission_action)

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
        next_progress += progress_step

# cleanup
if recorder:
    recorder.stop()
_decision_running = False
if ' _watcher' in globals():
    try:
        _watcher.join(timeout=1.0)
    except Exception:
        pass

engine_app.stop()
print("Simulation finished")
print(f"State CSV: {state_csv}")
print(f"Action CSV: {action_csv}")
