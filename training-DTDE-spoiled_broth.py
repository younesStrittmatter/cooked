import os
import sys
from spoiled_broth.rl.make_train_rllib import make_train_rllib
import ray
import torch

# PyTorch, NumPy, MKL, etc. not creating more threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

##### Cluster config ##################
NUM_GPUS = 1
NUM_CPUS = 24
CLUSTER = 'brigit'  # Options: 'brigit', 'local', 'cuenca'

# Read input file
input_path = sys.argv[1]
MAP_NR = sys.argv[2]
LR = float(sys.argv[3])
COOPERATIVE = int(sys.argv[4])
## If game_version = CLASSIC, one type of food (tomato); if game_version = COMPETITION, two types of food (tomato and potato)
GAME_VERSION = sys.argv[5]
INTENT_VERSION = sys.argv[6]
if len(sys.argv) > 7:
    NUM_AGENTS = int(sys.argv[7])
    if NUM_AGENTS not in [1, 2]:
        raise ValueError("NUM_AGENTS must be 1 or 2")
else:
    NUM_AGENTS = 2  # Default to 2 agents for backward compatibility

with open(input_path, "r") as f:
    lines = f.readlines()
    alpha_1, beta_1 = map(float, lines[0].strip().split())
    if NUM_AGENTS == 2:
        alpha_2, beta_2 = map(float, lines[1].strip().split())

if NUM_AGENTS == 1:
    reward_weights = {
        "ai_rl_1": (alpha_1, beta_1),
    }
else:
    reward_weights = {
        "ai_rl_1": (alpha_1, beta_1),
        "ai_rl_2": (alpha_2, beta_2),
    }

if CLUSTER == 'brigit':
    local = '/mnt/lustre/home/samuloza'
elif CLUSTER == 'cuenca':
    local = ''
elif CLUSTER == 'local':
    local = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
else:
    raise ValueError("Invalid cluster specified. Choose from 'brigit', 'cuenca', or 'local'.")

# Hiperparámetros
NUM_ENVS = 1
NUM_INNER_STEPS = 450
NUM_EPOCHS = 7500
NUM_MINIBATCHES = 20
SHOW_EVERY_N_EPOCHS = 1000
SAVE_EVERY_N_EPOCHS = 500
CONV_FILTERS = [
    [32, [3, 3], 1],  # Output: (32, 3, 3)
    [64, [2, 2], 1],  # Output: (64, 2, 2)
    #[64, [2, 2], 1],  # Output: (64, 2, 2)
]
PAYOFF_MATRIX = [1,1,-2]
MLP_LAYERS = [512, 512, 256]
USE_LSTM = False

if NUM_AGENTS == 1:
    raw_dir = f'{local}/data/samuel_lozano/cooked/pretraining'
else: 
    raw_dir = f'{local}/data/samuel_lozano/cooked'

if GAME_VERSION == "CLASSIC":
    game_dir = f'{raw_dir}/classic'
elif GAME_VERSION == "COMPETITION":
    game_dir = f'{raw_dir}/competition'
else:
    game_dir = f'{raw_dir}'

if INTENT_VERSION is not None:
    intent_dir = f'{game_dir}/{INTENT_VERSION}'
else: 
    intent_dir = f'{game_dir}'

if COOPERATIVE:
    save_dir = f'{intent_dir}/map_{MAP_NR}/cooperative'
else:
    save_dir = f'{intent_dir}/map_{MAP_NR}/competitive'

os.makedirs(save_dir, exist_ok=True)

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
    "NUM_INNER_STEPS": NUM_INNER_STEPS,
    "NUM_MINIBATCHES": NUM_MINIBATCHES,
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_AGENTS": NUM_AGENTS,
    "SHOW_EVERY_N_EPOCHS": SHOW_EVERY_N_EPOCHS,
    "SAVE_EVERY_N_EPOCHS": SAVE_EVERY_N_EPOCHS,
    "LR": LR,
    "MAP_NR": MAP_NR,
    "COOPERATIVE": COOPERATIVE,
    "REWARD_WEIGHTS": reward_weights,
    "INTENT_VERSION": INTENT_VERSION,
    "GAME_VERSION": GAME_VERSION,
    "PAYOFF_MATRIX": PAYOFF_MATRIX,
    "SAVE_DIR": save_dir,
    # RLlib specific parameters
    "NUM_UPDATES": 10,  # Number of updates of the policy
    "GAMMA": 0.975,  # Discount factor
    "GAE_LAMBDA": 0.95,  # GAE-Lambda parameter
    "ENT_COEF": 0.07,  # Entropy coefficient
    "CLIP_EPS": 0.2,  # PPO clip parameter
    "VF_COEF": 0.5,  # Value function coefficient
    "CONV_FILTERS": CONV_FILTERS,
    "GRID_SIZE": GRID_SIZE,
    # MLP/LSTM model config
    "USE_LSTM": USE_LSTM,  # Set to True to use LSTM
    "FCNET_HIDDENS": MLP_LAYERS,  # Hidden layer sizes for MLP
    "FCNET_ACTIVATION": "tanh",  # Activation function for MLP ("tanh", "relu", etc.)
    "MAX_SEQ_LEN": 20,  # Sequence length for LSTM (if used)
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
final_checkpoint = trainer.save(os.path.join(path, f"checkpoint_{NUM_EPOCHS}"))
print(f"Final checkpoint saved at {final_checkpoint}")