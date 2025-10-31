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
NUM_GPUS = 0.3
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

if NUM_AGENTS == 1:
    reward_weights = {"ai_rl_1": (alpha_1, beta_1)}
    walking_speeds = {"ai_rl_1": walking_speed_1}
    cutting_speeds = {"ai_rl_1": cutting_speed_1}
else:
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
INNER_SECONDS = 360 # In seconds
NUM_EPOCHS = 5000
TRAIN_BATCH_SIZE = 400 
NUM_MINIBATCHES = 20
SHOW_EVERY_N_EPOCHS = 1
SAVE_EVERY_N_EPOCHS = 500
PAYOFF_MATRIX = [1,1,-2]

# Neural network architecture
MLP_LAYERS = [512, 512, 256]

if NUM_AGENTS == 1:
    save_dir = f'{local}/data/samuel_lozano/cooked/pretraining/{GAME_VERSION}/map_{MAP_NR}'
else: 
    save_dir = f'{local}/data/samuel_lozano/cooked/{GAME_VERSION}/map_{MAP_NR}'

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
    "WAIT_FOR_COMPLETION": True,
    "SAVE_DIR": save_dir,
    "CHECKPOINT_ID_USED": pretrained_policies,  # Add pretrained policies configuration
    "PRETRAINED": PRETRAINING if PRETRAINING else "No",
    # RLlib specific parameters
    "NUM_UPDATES": 10,  # Number of updates of the policy
    "GAMMA": 0.975,  # Discount factor
    "GAE_LAMBDA": 0.95,  # GAE-Lambda parameter
    "ENT_COEF": 0.07,  # Entropy coefficient
    "CLIP_EPS": 0.2,  # PPO clip parameter
    "VF_COEF": 0.5,  # Value function coefficient
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
trainer, current_date = make_train_rllib(config)

# Save the final model
path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
os.makedirs(path, exist_ok=True)

# Save the final policy
final_checkpoint = trainer.save(os.path.join(path, f"checkpoint_final"))
print(f"Final checkpoint saved at {final_checkpoint}")