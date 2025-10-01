from engine.app.session_app import SessionApp
from spoiled_broth.game import SpoiledBroth as Game
from spoiled_broth.rl.rllib_controller import RLlibController
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
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
from io import BytesIO
from PIL import Image
from flask import jsonify
import signal
import sys

map_nr = 1
training_id = 'PPO_spoiled_broth_2025-05-20_06-21-00b6di_pyk'
checkpoint_number = 100

# Configuration for game duration and actions
cf_AI_TICK_RATE = 2  # Actions per second
ACTIONS_PER_AGENT = 50  # Reduced from 4000 to reduce load

# Calculate AI tick rate to achieve desired number of actions in desired duration
DESIRED_DURATION_SECONDS = ACTIONS_PER_AGENT / cf_AI_TICK_RATE

print(f"Game Configuration:")
print(f"  Desired Duration: {DESIRED_DURATION_SECONDS} seconds")
print(f"  Actions per Agent: {ACTIONS_PER_AGENT}")
print(f"  Calculated AI Tick Rate: {cf_AI_TICK_RATE:.2f} actions/second")
print(f"  Expected Actions per Agent: {cf_AI_TICK_RATE * DESIRED_DURATION_SECONDS:.0f}")
print()

local = '/mnt/lustre/home/samuloza'
#local = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'

base_path = Path(f"{local}/data/samuel_lozano/cooked/saved_models/map_{map_nr}/{training_id}")
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

# Inicializa Ray manualmente ANTES de que RLlib lo haga automáticamente
ray.shutdown()
ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)

register_env("spoiled_broth", lambda config: ParallelPettingZooEnv(env_creator(config)))

log = logging.getLogger('werkzeug')
log.disabled = True

path_root = Path().resolve() / "spoiled_broth"
checkpoint_dir = Path(f"{local}/data/samuel_lozano/cooked/saved_models/map_{map_nr}/{training_id}/Checkpoint_{checkpoint_number}/")
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
        
    def capture_frame(self, frame):
        if self.recording:
            if self.video_writer is None:
                # Initialize video writer with frame dimensions
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

def capture_canvas_frames(url="http://localhost:5000", duration_seconds=60, fps=15):
    """Capture frames from the game canvas using Selenium for a specified duration"""
    
    global global_recorder  # Declare global at the beginning of the function
    
    # Calculate number of frames based on duration and fps
    num_frames = int(duration_seconds * fps)
    
    # Setup Chrome options for headless browsing
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=800,600")
    chrome_options.add_argument("--disable-gpu")  # Disable GPU to reduce memory usage
    chrome_options.add_argument("--disable-extensions")  # Disable extensions
    
    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # Wait for the canvas to be available
        wait = WebDriverWait(driver, 10)
        canvas = wait.until(EC.presence_of_element_located((By.ID, "scene")))
        
        # Wait a bit for the game to load
        time.sleep(2)
        
        recorder = VideoRecorder(output_path=f"{base_path}/spoiled_broth_recording_checkpoint_{checkpoint_number}.mp4", fps=fps)
        recorder.start_recording()
        
        print(f"Capturing {num_frames} frames over {duration_seconds} seconds...")
        
        for i in range(num_frames):
            try:
                # Capture the canvas as base64
                canvas_base64 = driver.execute_script("""
                    var canvas = document.getElementById('scene');
                    return canvas.toDataURL('image/png').substring(22);
                """)
                
                # Convert base64 to numpy array
                img_data = base64.b64decode(canvas_base64)
                img = Image.open(BytesIO(img_data))
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                recorder.capture_frame(frame)
                
                # Wait for next frame
                time.sleep(1.0 / fps)
                
                if (i + 1) % 10 == 0:
                    elapsed_time = (i + 1) / fps
                    print(f"Captured {i + 1}/{num_frames} frames ({elapsed_time:.1f}s/{duration_seconds}s)")
                    
            except Exception as e:
                print(f"Error capturing frame {i}: {e}")
                # Continue with next frame instead of stopping
                continue
        
        recorder.stop_recording()
    
        
    except Exception as e:
        print(f"Error capturing frames: {e}")
        # Clear global recorder on error
        global_recorder = None
    finally:
        if driver:
            driver.quit()

# Create the engine app
engine_app = SessionApp(
    game_factory=partial(Game, map_nr=map_nr),
    ui_modules=[Renderer2DModule()],
    agent_map={
        "ai_1": RLlibController("ai_1", checkpoint_dir, "policy_ai_rl_1"),
        "ai_2": RLlibController("ai_2", checkpoint_dir, "policy_ai_rl_2")
    },
    path_root=path_root,
    tick_rate=24,
    ai_tick_rate=cf_AI_TICK_RATE,
    n_players=0, 
    is_max_speed=False
)

app = engine_app.app

if __name__ == "__main__":
    # Start the Flask app in a separate thread
    def run_app():
        app.run(port=5000, debug=False, use_reloader=False)
    
    app_thread = threading.Thread(target=run_app)
    app_thread.daemon = True
    app_thread.start()
    
    # Wait for the app to start
    time.sleep(3)
    
    # Capture video frames
    capture_canvas_frames(duration_seconds=DESIRED_DURATION_SECONDS, fps=15)
    
    print("Video recording completed!")