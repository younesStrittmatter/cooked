import os
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from spoiled_broth.rl.game_env import GameEnv
from collections import defaultdict
from pathlib import Path
import csv
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)

def env_creator(config):
    # Your existing GameEnv class goes here
    return GameEnv(**config)

# Define separate policies for each agent
def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    return f"policy_{agent_id}"

def make_train_rllib(config):
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
    os.makedirs(path, exist_ok=True)
    config["PATH"] = path

    # Save config to file
    with open(os.path.join(path, "config.txt"), "w") as f:
        for key, val in config.items():
            f.write(f"{key}: {val}\n")

    # Register the environment with RLLib
    register_env("spoiled_broth", lambda config: ParallelPettingZooEnv(env_creator(config)))

    # Configuration for multi-agent training
    ppo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .environment(
            env="spoiled_broth",
            env_config={
                "reward_weights": config["REWARD_WEIGHTS"],
                "map_nr": config["MAP_NR"],
                "cooperative": config["COOPERATIVE"],
                "step_per_episode": config["NUM_INNER_STEPS"],
                "path": config["PATH"]
            },
            clip_actions=True,
        )
        .multi_agent(
            policies={
                "policy_ai_rl_1": (
                    None,  # Use default PPO policy
                    env_creator({"map_nr": config["MAP_NR"]}).observation_space("ai_rl_1"),
                    env_creator({"map_nr": config["MAP_NR"]}).action_space("ai_rl_1"),
                    {}
                ),
                "policy_ai_rl_2": (
                    None,  # Use default PPO policy
                    env_creator({"map_nr": config["MAP_NR"]}).observation_space("ai_rl_2"),
                    env_creator({"map_nr": config["MAP_NR"]}).action_space("ai_rl_2"),
                    {}
                )
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["policy_ai_rl_1", "policy_ai_rl_2"]
        )
        .resources(num_gpus=1, num_gpus_per_worker=1)  # Set to 1 if you have a GPU
        .env_runners(num_env_runners=1)
        .training(
                train_batch_size=config["NUM_ENVS"] * config["NUM_INNER_STEPS"],
                lr=config["LR"],
                gamma=config["GAMMA"],
                lambda_=config["GAE_LAMBDA"],
                entropy_coeff=config["ENT_COEF"],
                clip_param=config["CLIP_EPS"],
                vf_loss_coeff=config["VF_COEF"],
                minibatch_size=config["NUM_INNER_STEPS"] // config["NUM_MINIBATCHES"],
                num_epochs=config["NUM_UPDATES"],
                model={
                    "fcnet_hiddens": config["MODEL_SIZE"],
                    "fcnet_activation": "tanh",
                    "use_lstm": config["USE_LSTM"],
                    "use_attention": False,
                }
            )
    )

    # Build algorithm with error handling
    trainer = ppo_config.build_algo()

    for epoch in range(config["NUM_EPOCHS"]): 
        result = trainer.train()

        # Log metrics
        if epoch % config["SHOW_EVERY_N_EPOCHS"] == 0:
            print(f"Epoch {epoch} complete.")

        # Save checkpoint
        if epoch % config["SAVE_EVERY_N_EPOCHS"] == 0:
            checkpoint_path = trainer.save(os.path.join(path, f"checkpoint_{epoch}"))
            print(f"Checkpoint saved at {checkpoint_path}")