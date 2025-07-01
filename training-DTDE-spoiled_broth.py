import os
import sys
from spoiled_broth.rl.make_train_rllib import make_train_rllib

# Leer archivo de entrada
input_path = sys.argv[1]
MAP_NR = sys.argv[2]
LR = float(sys.argv[3])
COOPERATIVE = int(sys.argv[4])

with open(input_path, "r") as f:
    lines = f.readlines()
    alpha_1, beta_1 = map(float, lines[0].strip().split())
    alpha_2, beta_2 = map(float, lines[1].strip().split())

reward_weights = {
    "ai_rl_1": (alpha_1, beta_1),
    "ai_rl_2": (alpha_2, beta_2),
}

local = '/mnt/lustre/home/samuloza'
#local = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'

# Hiperparámetros
NUM_ENVS = 1
NUM_INNER_STEPS = 150
NUM_EPOCHS = 20000
NUM_MINIBATCHES = 15
NUM_AGENTS = 2
SHOW_EVERY_N_EPOCHS = 1000
SAVE_EVERY_N_EPOCHS = 500

if COOPERATIVE:
    save_dir = f'{local}/data/samuel_lozano/cooked/map_{MAP_NR}/cooperative'
else:
    save_dir = f'{local}/data/samuel_lozano/cooked/map_{MAP_NR}/competitive'

os.makedirs(save_dir, exist_ok=True)

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
    "SAVE_DIR": save_dir,
    # RLlib specific parameters
    "NUM_UPDATES": 10,  # Number of updates of the policy
    "GAMMA": 0.9,  # Discount factor
    "GAE_LAMBDA": 0.95,  # GAE-Lambda parameter
    "ENT_COEF": 0.05,  # Entropy coefficient
    "CLIP_EPS": 0.2,  # PPO clip parameter
    "VF_COEF": 0.5  # Value function coefficient
}

# Run training
trainer, current_date = make_train_rllib(config)

# Save the final model
path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
os.makedirs(path, exist_ok=True)

# Save the final policy
final_checkpoint = trainer.save(os.path.join(path, "final_checkpoint"))
print(f"Final checkpoint saved at {final_checkpoint}")