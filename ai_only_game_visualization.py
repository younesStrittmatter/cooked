from engine.app.ai_only_app import AIOnlySessionApp
from spoiled_broth.game import SpoiledBroth as Game
from spoiled_broth.rl.rllib_controller import RLlibController
from spoiled_broth.rl.rllib_controller_lstm import RLlibControllerLSTM
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from pathlib import Path
from spoiled_broth.config import *
from functools import partial
from spoiled_broth.rl.game_env import GameEnv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
import ray
from ray.tune.registry import register_env
from spoiled_broth.game import SpoiledBroth
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import os
import logging
import time
import threading
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
from io import BytesIO
from PIL import Image
from flask import jsonify
import signal
import sys
import socket

training_map_nr = 'simple_kitchen'
playing_map_nr = training_map_nr

num_agents = int(sys.argv[1])
intent_version = sys.argv[2]
cooperative = int(sys.argv[3])
training_id = sys.argv[4]
checkpoint_number = int(sys.argv[5])

def find_free_port(start_port=5000, max_tries=100):
    port = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                s.close()
                return port
            except OSError:
                port += 1
    raise RuntimeError("Could not find a free port in range.")

# Find first free port starting from 5000
free_port = find_free_port(start_port=5000)
print(f"Using free port: {free_port}")

# --- Determine grid size from map file ---
map_txt_path = Path(__file__).parent / 'spoiled_broth' / 'maps' / f'{training_map_nr}.txt'
with open(map_txt_path, 'r') as f:
    map_lines = [line.rstrip('\n') for line in f.readlines()]
rows = len(map_lines)
cols = len(map_lines[0]) if rows > 0 else 0
grid_size = (cols, rows)  # (width, height)

# Configuration for game duration and actions
cf_AI_TICK_RATE = 1  # Actions per second (low rate for complete actions)
ACTIONS_PER_AGENT = 100  # Reduced from 4000 to reduce load

# Video speed configuration
VIDEO_FPS = 60  # Increased from 15 for faster playback
VIDEO_SPEED_MULTIPLIER = 2.0  # Makes video play 2x faster
FRAME_SKIP = 1  # Capture every Nth frame (1=no skip, 2=every other frame, etc.)
ENABLE_VIDEO_RECORDING = True  # Set to False to skip video recording entirely

# Calculate AI tick rate to achieve desired number of actions in desired duration
DESIRED_DURATION_SECONDS = ACTIONS_PER_AGENT / cf_AI_TICK_RATE

print(f"Game Configuration:")
print(f"  Desired Duration: {DESIRED_DURATION_SECONDS} seconds")
print(f"  Actions per Agent: {ACTIONS_PER_AGENT}")
print(f"  AI Tick Rate: {cf_AI_TICK_RATE:.2f} actions/second")
print(f"  Expected Actions per Agent: {cf_AI_TICK_RATE * DESIRED_DURATION_SECONDS:.0f}")
print(f"  Video FPS: {VIDEO_FPS}")
print(f"  Video Speed Multiplier: {VIDEO_SPEED_MULTIPLIER}x")
print(f"  Frame Skip: {FRAME_SKIP} (capture every {FRAME_SKIP}th frame)")
print(f"  Effective Video FPS: {VIDEO_FPS * VIDEO_SPEED_MULTIPLIER:.0f}")
print(f"  Actual Capture FPS: {VIDEO_FPS / FRAME_SKIP:.1f}")
print(f"  Video Recording: {'Enabled' if ENABLE_VIDEO_RECORDING else 'Disabled'}")
print()

#local = '/mnt/lustre/home/samuloza'
local = ''
#local = 'C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'

if num_agents == 1:
    raw_dir = f'{local}/data/samuel_lozano/cooked/pretraining'
else:
    raw_dir = f'{local}/data/samuel_lozano/cooked'

if intent_version is not None:
    intent_dir = f'{raw_dir}/{intent_version}'
else:
    intent_dir = f'{raw_dir}'

if cooperative: 
    base_path = Path(f"{intent_dir}/map_{training_map_nr}/cooperative/{training_id}")
else:
    base_path = Path(f"{intent_dir}/map_{training_map_nr}/competitive/{training_id}")

config_path = base_path / "config.txt"
if config_path.exists():
    print(f"\n--- Contents of config.txt for training_id '{training_id}' ---")
    with open(config_path, "r") as f:
        for i in range(7):
            line = f.readline()
            if not line:
                break
            print(line.rstrip())
else:
    print(f"config.txt not found at: {config_path}")

print(f"\n--- Using checkpoint number: {checkpoint_number} ---\n")

def env_creator(config):
    return GameEnv(**config)

# Initialize Ray manually BEFORE RLlib does it automatically
ray.shutdown()
ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)

register_env("spoiled_broth", lambda config: ParallelPettingZooEnv(env_creator(config)))

log = logging.getLogger('werkzeug')
log.disabled = True

path_root = Path().resolve() / "spoiled_broth"
checkpoint_dir = Path(f"{base_path}/checkpoint_{checkpoint_number}/")
if not checkpoint_dir.exists():
    raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

class VideoRecorder:
    def __init__(self, output_path="game_recording.mp4", fps=24):
        self.output_path = output_path
        self.fps = fps
        self.recording = False
        self.video_writer = None
        self.frame_count = 0
        
    def start_recording(self):
        self.recording = True
        self.frame_count = 0
        print(f"Started recording video to {self.output_path}")
        
    def capture_frame(self, frame, scores=None):
        if self.recording:
            # Draw the scores on a separate black bar above the frame
            if scores is not None:
                text = " | ".join([f"{agent}: {score}" for agent, score in scores.items()])
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                color = (0, 255, 0)  # bright green
                thickness = 2

                # Compute text size to center it if you want
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_height = text_size[1]
                bar_height = text_height + 20  # add some padding

                # Create a new black bar image
                bar = np.zeros((bar_height, frame.shape[1], 3), dtype=np.uint8)

                # Put text onto the bar
                text_x = 10
                text_y = bar_height - 10
                cv2.putText(bar, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

                # Stack the bar on top of the frame
                frame = np.vstack((bar, frame))

            # Initialize writer if needed
            if self.video_writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

            self.video_writer.write(frame)
            self.frame_count += 1
            
    def stop_recording(self):
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print(f"Video saved successfully to {self.output_path} ({self.frame_count} frames)")
            
    def save_video(self):
        # This method is no longer needed as frames are written directly
        pass

def capture_canvas_frames(url="http://localhost:5000", duration_seconds=60, fps=15, frame_skip=1):
    """Capture frames from the game canvas using Selenium for a specified duration"""
    
    # Calculate number of frames based on duration and fps
    num_frames = int(duration_seconds * fps)
    frames_to_capture = num_frames // frame_skip
    
    # Setup Chrome options for headless browsing with better compatibility
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=800,600")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    chrome_options.add_argument("--disable-features=VizDisplayCompositor")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = None
    try:
        print(f"Starting Chrome WebDriver...")
        # Try to use service for better compatibility
        try:
            service = Service(log_output=os.devnull)  # Suppress service logs
            driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception as service_error:
            print(f"Service-based WebDriver failed, trying direct initialization: {service_error}")
            driver = webdriver.Chrome(options=chrome_options)
        
        print(f"Chrome WebDriver started successfully")
        
        print(f"Navigating to {url}...")
        driver.get(url)
        print(f"Page loaded successfully")
        
        # Wait for the canvas to be available
        print(f"Waiting for canvas element...")
        wait = WebDriverWait(driver, 20)  # Increased timeout
        canvas = wait.until(EC.presence_of_element_located((By.ID, "scene")))
        print(f"Canvas element found")
        
        # Wait a bit for the game to load
        print(f"Waiting for game to load...")
        time.sleep(5)  # Increased wait time
        
        recorder = VideoRecorder(output_path=f"{base_path}/ai_only_recording_checkpoint_{checkpoint_number}.mp4", fps=fps)
        recorder.start_recording()
        
        print(f"Capturing {frames_to_capture} frames over {duration_seconds} seconds (every {frame_skip}th frame)...")
        
        frame_count = 0
        for i in range(num_frames):
            try:
                # Only capture every Nth frame
                if i % frame_skip == 0:
                    # Capture the canvas as base64
                    canvas_base64 = driver.execute_script("""
                        var canvas = document.getElementById('scene');
                        if (!canvas) {
                            throw new Error('Canvas element not found');
                        }
                        return canvas.toDataURL('image/png').substring(22);
                    """)
                    
                    # Convert base64 to numpy array
                    img_data = base64.b64decode(canvas_base64)
                    img = Image.open(BytesIO(img_data))
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    
                    session = engine_app.get_session()
                    scores = session.engine.game.agent_scores

                    recorder.capture_frame(frame, scores=scores)
                    frame_count += 1
                
                # Wait for next frame
                time.sleep(1.0 / fps)
                
                if (i + 1) % (10 * frame_skip) == 0:
                    elapsed_time = (i + 1) / fps
                    print(f"Captured {frame_count}/{frames_to_capture} frames ({elapsed_time:.1f}s/{duration_seconds}s)")
                    
            except Exception as e:
                print(f"Error capturing frame {i}: {e}")
                print(f"Frame capture error details: {type(e).__name__}")
                # Continue with next frame instead of stopping
                continue
        
        recorder.stop_recording()
    
        
    except Exception as e:
        print(f"Error capturing frames: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        
        # Try to get more information about the driver state
        if driver:
            try:
                print(f"Current URL: {driver.current_url}")
                print(f"Page title: {driver.title}")
                print(f"Page source length: {len(driver.page_source)}")
            except Exception as debug_e:
                print(f"Could not get debug info: {debug_e}")
    finally:
        if driver:
            try:
                driver.quit()
                print("Chrome WebDriver closed successfully")
            except Exception as e:
                print(f"Error closing WebDriver: {e}")

use_lstm = False
if config_path.exists():
    with open(config_path, "r") as f:
        for line in f:
            if line.strip().startswith("USE_LSTM"):
                key, value = line.strip().split(":", 1)
                value = value.strip().lower()
                if value == "true":
                    use_lstm = True
                break
print(f"\nUsing LSTM controller: {'Yes' if use_lstm else 'No'}\n")

# Choose controller class
controller_cls = RLlibControllerLSTM if use_lstm else RLlibController

# Create the agent_map dynamically
if num_agents == 1:
    agent_map = {
        "ai_rl_1": controller_cls("ai_rl_1", checkpoint_dir, "policy_ai_rl_1")
    }
else:
    agent_map = {
        "ai_rl_1": controller_cls("ai_rl_1", checkpoint_dir, "policy_ai_rl_1"),
        "ai_rl_2": controller_cls("ai_rl_2", checkpoint_dir, "policy_ai_rl_2")
    }

# Create the AI-only engine app
engine_app = AIOnlySessionApp(
    game_factory=partial(Game, map_nr=playing_map_nr, grid_size=grid_size, intent_version=intent_version),
    ui_modules=[Renderer2DModule()],
    agent_map=agent_map,
    path_root=path_root,
    tick_rate=24,
    ai_tick_rate=cf_AI_TICK_RATE,
    is_max_speed=False
)

app = engine_app.app

if __name__ == "__main__":
    # Start the Flask app in a separate thread
    def run_app():
        app.run(port=free_port, debug=False, use_reloader=False)
    
    app_thread = threading.Thread(target=run_app)
    app_thread.daemon = True
    app_thread.start()
    
    # Wait for the app to start
    time.sleep(3)
    
    if ENABLE_VIDEO_RECORDING:
        try:
            # Capture video frames
            capture_canvas_frames(url=f"http://localhost:{free_port}", duration_seconds=DESIRED_DURATION_SECONDS, fps=VIDEO_FPS, frame_skip=FRAME_SKIP)
        except Exception as e:
            print(f"Video recording failed: {e}")
            print("Falling back to running game without video recording...")
            time.sleep(DESIRED_DURATION_SECONDS)
    else:
        print("Video recording disabled. Running game for specified duration...")
        time.sleep(DESIRED_DURATION_SECONDS)
    
    # Stop the session
    engine_app.stop()
    
    print("Game execution completed!")
