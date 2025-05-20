import os
import shutil
import json
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from spoiled_broth.rl.game_env import GameEnv
from pathlib import Path
import supersuit as ss
import numpy as np
import sys
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

inputs = [line.strip() for line in sys.stdin.readlines()]

rl_model = inputs[0]
map_nr = int(inputs[1])
n_iterations = int(inputs[2])
save_every_n_iterations = int(inputs[3])
w1_1 = float(inputs[4])
w1_2 = float(inputs[5])
w2_1 = float(inputs[6])
w2_2 = float(inputs[7])

# Define reward weights
reward_weights = {
    "ai_rl_1": (w1_1, w1_2),
    "ai_rl_2": (w2_1, w2_2),
}

base_save_dir = f"/data/samuel_lozano/cooked/saved_models/map_{map_nr}/"
Path(base_save_dir).mkdir(parents=True, exist_ok=True)

def env_creator(config):
    # Your existing GameEnv class goes here
    return GameEnv(**config)

# Initialize Ray
os.environ["RAY_TMPDIR"] = "/data/samuel_lozano/tmp_ray/"
ray.shutdown()
ray.init(ignore_reinit_error=True, _temp_dir=os.environ["RAY_TMPDIR"])

# Register the environment with RLLib
register_env("spoiled_broth", lambda config: ParallelPettingZooEnv(env_creator(config)))

# Define separate policies for each agent
def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    return f"policy_{agent_id}"

# Configuration for multi-agent training
config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True
    )
    .environment(
        env="spoiled_broth",
        env_config={
            "reward_weights": reward_weights,
            "map_nr": map_nr
        },
        clip_actions=True,
    )
    .multi_agent(
        policies={
            "policy_ai_rl_1": (
                None,  # Use default PPO policy
                env_creator({"map_nr": 1}).observation_space("ai_rl_1"),
                env_creator({"map_nr": 1}).action_space("ai_rl_1"),
                {}
            ),
            "policy_ai_rl_2": (
                None,  # Use default PPO policy
                env_creator({"map_nr": 1}).observation_space("ai_rl_2"),
                env_creator({"map_nr": 1}).action_space("ai_rl_2"),
                {}
            )
        },
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["policy_ai_rl_1", "policy_ai_rl_2"]
    )
    .resources(num_gpus=1)  # Set to 1 if you have a GPU
    .env_runners(num_env_runners=2)
    .training(
        train_batch_size=4000,
        minibatch_size=500,
        num_epochs=10
    )
)

# Build algorithm with error handling
try:
    algo = config.build_algo()
    full_run_dir = algo.logdir
    NAME_RAY = Path(full_run_dir).name
    print(f"Algorithm built successfully: {NAME_RAY}")
except Exception as e:
    print(f"Error building algorithm: {e}")
    raise

# Save training config
training_dir = Path(f"{base_save_dir}{NAME_RAY}")
training_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(full_run_dir, training_dir, dirs_exist_ok=True)

def safe_config_dict(d):
    def make_serializable(o):
        try:
            json.dumps(o)
            return o
        except TypeError:
            return str(o)

    return {
        k: make_serializable(v)
        for k, v in d.items()
    }

config_path = f"{training_dir}/config.txt"
with open(config_path, "w") as f:
    f.write("==== Training parameters ====\n")
    f.write(f"RL model: {rl_model}\n")
    f.write(f"Map number: {map_nr}\n")
    f.write(f"Number of iterations: {n_iterations}\n")
    f.write(f"Saved every N iterations: {save_every_n_iterations}\n")
    f.write(f"Reward weights ai_rl_1: {w1_1}, {w1_2}\n")
    f.write(f"Reward weights ai_rl_2: {w2_1}, {w2_2}\n")
    f.write("\n==== Complete configuration ====\n")
    f.write(json.dumps(safe_config_dict(config.to_dict()), indent=4))

csv_file_path = f'{training_dir}/reward_data.csv'
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["iteration", "total_reward", "reward_agent_1", "reward_agent_2"])

for i in range(n_iterations): 
    result = algo.train()
    reward_agent_1 = result['env_runners']['module_episode_returns_mean'].get('policy_ai_rl_1', 0)
    reward_agent_2 = result['env_runners']['module_episode_returns_mean'].get('policy_ai_rl_2', 0)
    total_reward = result['env_runners']['episode_return_mean']

    print(f"Iteration {i}:")
    print(f"  Agent 1 reward: {reward_agent_1}")
    print(f"  Agent 2 reward: {reward_agent_2}")
    print(f"  Total reward: {total_reward}")

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i, total_reward, reward_agent_1, reward_agent_2])

    # Save checkpoint
    if i % save_every_n_iterations == 0:
        checkpoint_obtained = algo.save()
        checkpoint_path = Path(checkpoint_obtained.checkpoint.path)
        
        custom_checkpoint_dir = Path(f"{training_dir}/Checkpoint_{i}")
        custom_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for item in checkpoint_path.iterdir():
            dst = custom_checkpoint_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)

        print(f"Checkpoint {i} saved in {custom_checkpoint_dir}")