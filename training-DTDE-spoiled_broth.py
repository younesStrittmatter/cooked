# USE: nohup python training-DTDE-spoiled_broth.py <input_path> <map_nr> <lr> <game_version> [<num_agents>] [<checkpoint_id>] [<pretraining>] [<seed>] > log_training.txt &

import os
import sys
from spoiled_broth.rl.make_train_rllib import make_train_rllib
import ray
import torch

# PyTorch, NumPy, MKL, etc. not creating more threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

##### Cluster config ##################
NUM_GPUS = 0.2
NUM_CPUS = 10
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

# Hyperparameters
NUM_ENVS = 1
INNER_SECONDS = 180 # In seconds
NUM_EPOCHS = 25000
TRAIN_BATCH_SIZE = 200 # (Train batch size should be aprox equal to inner_seconds)
NUM_MINIBATCHES = 10
SHOW_EVERY_N_EPOCHS = 1
SAVE_EVERY_N_EPOCHS = 500
PAYOFF_MATRIX = [1,1,-2]

# Neural network architecture
MLP_LAYERS = [512, 512, 256]

# Game characteristics
PENALTIES_CFG = {
    "busy": 0.01, # Penalty per second spent busy
    "useless_action": 5.0, # Penalty for useless actions
    "destructive_action": 10.0, # Penalty for destructive actions
    "inaccessible_tile": 5.0, # Penalty for trying to access an inaccessible tile
}

REWARDS_CFG = {
    "raw_food": 2.0,
    "plate": 2.0,
    "counter": 0.5,
    "cut": 5.0,
    "salad": 7.0,
    "deliver": 10.0,
}

# Dynamic rewards configuration - exponential decay [rewards_cfg = original_rewards_cfg * exp(-decay_rate * (episode - decay_start_episode))]
DYNAMIC_REWARDS_CFG = {
    "enabled": True,  # Set to False to disable dynamic rewards
    "decay_rate": 0.0001,  # Decay rate for exponential function (higher = faster decay)
    "min_reward_multiplier": 0.05,  # Minimum multiplier (e.g., 0.1 = 10% of initial reward)
    "decay_start_episode": 0,  # Episode to start applying decay (0 = from beginning)
    "affected_rewards": ["raw_food", "plate", "counter", "cut", "salad"],  # Which reward types to apply decay to
}

# Dynamic PPO parameters configuration - exponential decay for exploration and policy change control
# This gradually reduces clip_eps (policy change constraint) and ent_coef (exploration) during training
# Starting with high values for exploration, then reducing them for more stable exploitation
DYNAMIC_PPO_PARAMS_CFG = {
    "enabled": True,  # Set to False to disable dynamic PPO parameters
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

# RLlib specific configuration
config = {
    "NUM_ENVS": NUM_ENVS,
    "INNER_SECONDS": INNER_SECONDS,
    "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
    "NUM_MINIBATCHES": NUM_MINIBATCHES,
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
    # RLlib specific parameters
    "NUM_UPDATES": 10,  # Number of updates of the policy
    "GAMMA": 0.975,      # Discount factor for future rewards (close to 1 = long-term, lower = short-term)
    "GAE_LAMBDA": 0.95, # Lambda for Generalized Advantage Estimation (controls bias-variance tradeoff in advantage calculation)
    "ENT_COEF": 0.07,   # Entropy coefficient (controls exploration: higher = more random actions)
    "CLIP_EPS": 0.2,    # PPO clip parameter (limits how much the policy can change at each update; stabilizes training)
    "VF_COEF": 0.5,     # Value function loss coefficient (relative weight of value loss vs. policy loss)
    "GRID_SIZE": GRID_SIZE,
    "FCNET_HIDDENS": MLP_LAYERS,  # Hidden layer sizes for MLP
    "FCNET_ACTIVATION": "tanh",  # Activation function for MLP ("tanh", "relu", etc.)
    "NUM_CPUS": NUM_CPUS,
    "NUM_GPUS": NUM_GPUS,
}

if ray.is_initialized():
    ray.shutdown()
ray.init(num_cpus=NUM_CPUS)

torch.set_num_threads(NUM_CPUS)

# Run training
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