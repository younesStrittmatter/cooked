# USE:   <input_path> <map_nr> <lr> <game_version> [<num_agents>] [<checkpoint_id>] [<pretraining>] [<seed>] > log_training.log 2>&1 &
# Example: nohup python training-DTDE-spoiled_broth.py ./cuenca/input_0_0.txt baseline_division_of_labor_v2 0.0003 classic 1 none no 0 > log_training.log 2>&1 &

import os
import sys
from spoiled_broth.rl.make_train_rllib import make_train_rllib
import ray
import torch

# PyTorch, NumPy, MKL, etc. not creating more threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

##### Cluster config ##################
# Resource allocation optimized for RL training
NUM_GPUS = 0.2  # Full GPU for neural network training
NUM_CPUS = 12   # Increased CPU cores for parallel environments
NUM_ENV_WORKERS = 8  # Parallel environment workers
NUM_LEARNER_WORKERS = 1  # GPU learner workers
CLUSTER = 'cuenca'  # Options: 'brigit', 'local', 'cuenca'

# Read input file
input_path = sys.argv[1]
MAP_NR = str(sys.argv[2]).lower()
LR = float(sys.argv[3])
GAME_VERSION = str(sys.argv[4]).lower() ## If game_version = classic, one type of food (tomato); if game_version = competition, two types of food (tomato and pumpkin)
if len(sys.argv) > 5:
    NUM_AGENTS = int(sys.argv[5])
    if NUM_AGENTS not in [1, 2]:
        raise ValueError("NUM_AGENTS must be 1 or 2")
else:
    NUM_AGENTS = 2  # Default to 2 agents for backward compatibility

# Optional checkpoint paths for loading pretrained policies
CHECKPOINT_ID = str(sys.argv[6]) if len(sys.argv) > 6 else None
PRETRAINING = str(sys.argv[7]) if len(sys.argv) > 7 else None
SEED = int(sys.argv[8]) if len(sys.argv) > 8 else 0

with open(input_path, "r") as f:
    lines = f.readlines()
    alpha_1, beta_1 = [round(float(x), 4) for x in lines[0].strip().split()]
    walking_speed_1, cutting_speed_1 = [round(float(x), 4) for x in lines[1].strip().split()]
    if NUM_AGENTS == 2:
        alpha_2, beta_2 = [round(float(x), 4) for x in lines[2].strip().split()]
        walking_speed_2, cutting_speed_2 = [round(float(x), 4) for x in lines[3].strip().split()]    

if CLUSTER == 'brigit':
    local = '/mnt/lustre/home/samuloza'
elif CLUSTER == 'cuenca':
    local = ''
elif CLUSTER == 'local':
    local = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
else:
    raise ValueError("Invalid cluster specified. Choose from 'brigit', 'cuenca', or 'local'.")

# Hyperparameters - Optimized for parallel training
NUM_ENVS = NUM_ENV_WORKERS  # Use all environment workers
INNER_SECONDS = 180 # In seconds
NUM_EPOCHS = 200
TRAIN_BATCH_SIZE = 4000  # Increased for better GPU utilization (NUM_ENVS * rollout_fragment_length * num_timesteps)
SGD_MINIBATCH_SIZE = 500  # Optimized minibatch size for GPU
NUM_SGD_ITER = 10  # Number of SGD iterations per training batch
SHOW_EVERY_N_EPOCHS = 1
SAVE_EVERY_N_EPOCHS = 500
PAYOFF_MATRIX = [1,1,-2]

# Neural network architecture
MLP_LAYERS = [512, 512, 256]

# Game characteristics
PENALTIES_CFG = {
    "busy": 0.01, # Penalty per second spent busy
    "useless_action": 2.0, # Penalty for useless actions
    "destructive_action": 10.0, # Penalty for destructive actions
    "inaccessible_tile": 5.0, # Penalty for trying to access an inaccessible tile
    "not_available": 2.0, # Penalty for trying to perform an action that is not available
}

REWARDS_CFG = {
    "raw_food": 0.0,
    "plate": 0.0,
    "counter": 0.0,
    "cut": 0.0,
    "salad": 0.0,
    "deliver": 10.0,
}

# Dynamic rewards configuration - exponential decay [rewards_cfg = original_rewards_cfg * exp(-decay_rate * (episode - decay_start_episode))]
DYNAMIC_REWARDS_CFG = {
    "enabled": False,  # Set to False to disable dynamic rewards
    "decay_rate": 0.005,  # Decay rate for exponential function (higher = faster decay)
    "min_reward_multiplier": 0.00,  # Minimum multiplier (e.g., 0.1 = 10% of initial reward)
    "decay_start_episode": 100,  # Episode to start applying decay (0 = from beginning)
    "affected_rewards": ["raw_food", "plate", "counter", "cut", "salad"],  # Which reward types to apply decay to
}

# Dynamic PPO parameters configuration - exponential decay for exploration and policy change control
# This gradually reduces clip_eps (policy change constraint) and ent_coef (exploration) during training
# Starting with high values for exploration, then reducing them for more stable exploitation
DYNAMIC_PPO_PARAMS_CFG = {
    "enabled": False,  # Set to False to disable dynamic PPO parameters
    "decay_rate": 0.0001,  # Decay rate for exponential function (higher = faster decay)
    "min_param_multiplier": 0.1,  # Minimum multiplier (e.g., 0.1 = 10% of initial value)
    "decay_start_episode": 100,  # Episode to start applying decay (0 = from beginning)
    "affected_params": ["ent_coef"],  # Which PPO parameters to apply decay to
}

WAIT_FOR_ACTION_COMPLETION = True  # Flag to ensure actions complete before next step

# Path definitions
if NUM_AGENTS == 1:
    save_dir = f'{local}/data/samuel_lozano/cooked/pretraining/{GAME_VERSION}/map_{MAP_NR}'
    reward_weights = {"ai_rl_1": (alpha_1, beta_1)}
    walking_speeds = {"ai_rl_1": walking_speed_1}
    cutting_speeds = {"ai_rl_1": cutting_speed_1}
else: 
    save_dir = f'{local}/data/samuel_lozano/cooked/{GAME_VERSION}/map_{MAP_NR}'
    reward_weights = {
        "ai_rl_1": (alpha_1, beta_1),
        "ai_rl_2": (alpha_2, beta_2),
    }
    walking_speeds = {
        "ai_rl_1": walking_speed_1,
        "ai_rl_2": walking_speed_2,
    }
    cutting_speeds = {
        "ai_rl_1": cutting_speed_1,
        "ai_rl_2": cutting_speed_2,
    }

os.makedirs(save_dir, exist_ok=True)

pretrained_policies = {}

if CHECKPOINT_ID is not None and CHECKPOINT_ID.lower() != "none":
    if PRETRAINING.upper() == "YES":
        CHECKPOINT_PATH = f'{local}/data/samuel_lozano/cooked/pretraining/{GAME_VERSION}/map_{MAP_NR}/Training_{CHECKPOINT_ID}'
    else:
        CHECKPOINT_PATH = f'{save_dir}/Training_{CHECKPOINT_ID}'

    if NUM_AGENTS == 1:
        pretrained_policies["ai_rl_1"] = {"path": CHECKPOINT_PATH}

    elif NUM_AGENTS == 2:
        pretrained_policies["ai_rl_1"] = {"path": CHECKPOINT_PATH}
        pretrained_policies["ai_rl_2"] = {"path": CHECKPOINT_PATH}

# Determine grid size from map file (text format)
map_txt_path = os.path.join(os.path.dirname(__file__), 'spoiled_broth', 'maps', f'{MAP_NR}.txt')
if not os.path.exists(map_txt_path):
    raise FileNotFoundError(f"Map file {map_txt_path} not found.")
with open(map_txt_path, 'r') as f:
    map_lines = [line.rstrip('\n') for line in f.readlines()]
rows = len(map_lines)
cols = len(map_lines[0]) if rows > 0 else 0
if rows != cols:
    raise ValueError(f"Map must be square, but got {rows} rows and {cols} columns.")
GRID_SIZE = (rows, cols)

# RLlib specific configuration - Optimized for GPU training
config = {
    "NUM_ENVS": NUM_ENVS,
    "INNER_SECONDS": INNER_SECONDS,
    "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
    "SGD_MINIBATCH_SIZE": SGD_MINIBATCH_SIZE,
    "NUM_SGD_ITER": NUM_SGD_ITER,
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_AGENTS": NUM_AGENTS,
    "SHOW_EVERY_N_EPOCHS": SHOW_EVERY_N_EPOCHS,
    "SAVE_EVERY_N_EPOCHS": SAVE_EVERY_N_EPOCHS,
    "LR": LR,
    "MAP_NR": MAP_NR,
    "REWARD_WEIGHTS": reward_weights,
    "GAME_VERSION": GAME_VERSION,
    "PAYOFF_MATRIX": PAYOFF_MATRIX,
    "WALKING_SPEEDS": walking_speeds,
    "CUTTING_SPEEDS": cutting_speeds,
    "INITIAL_SEED": SEED,
    "PENALTIES_CFG": PENALTIES_CFG,
    "REWARDS_CFG": REWARDS_CFG,
    "DYNAMIC_REWARDS_CFG": DYNAMIC_REWARDS_CFG,
    "DYNAMIC_PPO_PARAMS_CFG": DYNAMIC_PPO_PARAMS_CFG,
    "WAIT_FOR_COMPLETION": WAIT_FOR_ACTION_COMPLETION,
    "SAVE_DIR": save_dir,
    "CHECKPOINT_ID_USED": pretrained_policies,  # Add pretrained policies configuration
    "PRETRAINED": PRETRAINING if PRETRAINING else "No",
    # RLlib specific parameters - Optimized for GPU
    "NUM_ENV_WORKERS": NUM_ENV_WORKERS,  # Parallel environment workers (CPU)
    "NUM_LEARNER_WORKERS": NUM_LEARNER_WORKERS,  # GPU learner workers
    "NUM_UPDATES": NUM_SGD_ITER,  # Number of SGD iterations per batch
    "GAMMA": 0.975,      # Discount factor for future rewards (close to 1 = long-term, lower = short-term)
    "GAE_LAMBDA": 0.95, # Lambda for Generalized Advantage Estimation (controls bias-variance tradeoff in advantage calculation)
    "ENT_COEF": 0.07,   # Entropy coefficient (controls exploration: higher = more random actions)
    "CLIP_EPS": 0.2,    # PPO clip parameter (limits how much the policy can change at each update; stabilizes training)
    "VF_COEF": 0.5,     # Value function loss coefficient (relative weight of value loss vs. policy loss)
    "GRID_SIZE": GRID_SIZE,
    "FCNET_HIDDENS": MLP_LAYERS,  # Hidden layer sizes for MLP
    "FCNET_ACTIVATION": "tanh",  # Activation function for MLP ("tanh", "relu", etc.)
    # Resource allocation
    "NUM_CPUS": NUM_CPUS,
    "NUM_GPUS": NUM_GPUS,
    # Performance optimizations
    "ROLLOUT_FRAGMENT_LENGTH": 200,  # Steps per rollout fragment
    "BATCH_MODE": "complete_episodes",  # Collect complete episodes for better learning
    "COMPRESS_OBSERVATIONS": False,  # Disable compression for speed
    "NUM_CPUS_PER_WORKER": 1,  # CPU cores per environment worker
    "NUM_GPUS_PER_WORKER": 0,  # Environment workers run on CPU only
    "NUM_CPUS_FOR_DRIVER": 1,  # Driver CPU usage
}

if ray.is_initialized():
    ray.shutdown()

# Initialize Ray with optimized resource allocation
ray.init(
    num_cpus=NUM_CPUS,
    num_gpus=1,  # Ensure GPU is available
    object_store_memory=2000000000,  # 2GB object store for efficient data transfer
    _plasma_directory="/tmp",  # Use fast storage for plasma store
)

# Optimize PyTorch for GPU training
torch.set_num_threads(2)  # Limit CPU threads per process to avoid oversubscription

# GPU optimization settings
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # GPU memory optimizations
    torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of GPU memory
    torch.cuda.empty_cache()  # Clear cache
    
    # Performance optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
    
    print("GPU optimizations enabled")
else:
    print("CUDA not available, falling back to CPU training")

# Environment optimization
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Enable async CUDA operations
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth

# Run training with performance monitoring
print(f"Starting training with configuration:")
print(f"  Environment workers: {NUM_ENV_WORKERS} (CPU)")
print(f"  Learner workers: {NUM_LEARNER_WORKERS} (GPU)")
print(f"  Train batch size: {TRAIN_BATCH_SIZE}")
print(f"  SGD minibatch size: {SGD_MINIBATCH_SIZE}")
print(f"  Total CPU cores: {NUM_CPUS}")
print(f"  GPU allocation: {NUM_GPUS}")

# Monitor GPU memory before training
if torch.cuda.is_available():
    print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

try:
    trainer, current_date, final_episode_count = make_train_rllib(config)

    # Save the final model
    path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
    os.makedirs(path, exist_ok=True)

    # Update config with final episode count
    if final_episode_count is not None:
        config["NUM_EPISODES"] = final_episode_count
        
        # Append the final episode count to config.txt
        config_path = os.path.join(path, "config.txt")
        with open(config_path, "a") as f:
            f.write(f"NUM_EPISODES: {final_episode_count}\n")

    # Save the final policy
    final_checkpoint = trainer.save(os.path.join(path, f"checkpoint_final"))
    print(f"Final checkpoint saved at {final_checkpoint}")
    if final_episode_count is not None:
        print(f"Training completed after {final_episode_count} episodes")

except Exception as e:
    print(f"Error during training: {e}")
    raise
finally:
    # Proper cleanup to avoid Ray shutdown warnings
    try:
        if 'trainer' in locals() and trainer is not None:
            # Stop the trainer properly
            trainer.stop()
            print("Trainer stopped successfully")
    except Exception as cleanup_error:
        print(f"Warning: Error during trainer cleanup: {cleanup_error}")
    
    try:
        # Shutdown Ray
        ray.shutdown()
        print("Ray shutdown completed")
    except Exception as ray_error:
        print(f"Warning: Error during Ray shutdown: {ray_error}")