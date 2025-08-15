import os
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from spoiled_broth.rl.game_env import GameEnv
from spoiled_broth.rl.game_env_competition import GameEnvCompetition
from collections import defaultdict
from pathlib import Path
import csv
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)

def env_creator(config):
    # Your existing GameEnv class goes here
    return GameEnv(**config)

def env_creator_competition(config):
    # Your existing GameEnv class goes here
    return GameEnvCompetition(**config)

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

    if config["GAME_VERSION"] == "CLASSIC":
        # Register the environment with RLLib
        register_env("spoiled_broth", lambda cfg: ParallelPettingZooEnv(env_creator(cfg)))
    elif config["GAME_VERSION"] == "COMPETITION":
        # Register the environment with RLLib
        register_env("spoiled_broth", lambda cfg: ParallelPettingZooEnv(env_creator_competition(cfg)))
    else:
        print("Incorrect GAME_VERSION!")

    # --- Dynamic policy setup ---
    num_agents = config.get("NUM_AGENTS", 2)
    agent_ids = [f"ai_rl_{i+1}" for i in range(num_agents)]
    policies = {}
    policies_to_train = []
    for agent_id in agent_ids:
        if config["GAME_VERSION"] == "CLASSIC":
            policies[f"policy_{agent_id}"] = (
                None,  # Use default PPO policy
                env_creator({"map_nr": config["MAP_NR"], "grid_size": config.get("GRID_SIZE", (8, 8))}).observation_space(agent_id),
                env_creator({"map_nr": config["MAP_NR"], "grid_size": config.get("GRID_SIZE", (8, 8))}).action_space(agent_id),
                {}
            )
        elif config["GAME_VERSION"] == "COMPETITION":
            policies[f"policy_{agent_id}"] = (
                None,  # Use default PPO policy
                env_creator_competition({"map_nr": config["MAP_NR"], "grid_size": config.get("GRID_SIZE", (8, 8))}).observation_space(agent_id),
                env_creator_competition({"map_nr": config["MAP_NR"], "grid_size": config.get("GRID_SIZE", (8, 8))}).action_space(agent_id),
                {}
            )
        policies_to_train.append(f"policy_{agent_id}")

    def dynamic_policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        return f"policy_{agent_id}"

    # --- Model config for MLP or LSTM ---
    model_config = {
        "fcnet_hiddens": config.get("FCNET_HIDDENS", [256, 256]),
        "fcnet_activation": config.get("FCNET_ACTIVATION", "tanh"),
        "use_lstm": config.get("USE_LSTM", False),
        "max_seq_len": config.get("MAX_SEQ_LEN", 20),
    }
    
    # --- Model config for CNN ---
    #model_config={
    #            "conv_filters": config["CONV_FILTERS"],
    #            "conv_activation": "tanh",
    #            "use_lstm": False,
    #            "use_attention": False,
    #        }

    # Configuration for multi-agent training
    ppo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .rl_module(
            model_config=model_config
        )
        .environment(
            env="spoiled_broth",
            env_config={
                "reward_weights": config["REWARD_WEIGHTS"],
                "map_nr": config["MAP_NR"],
                "cooperative": config["COOPERATIVE"],
                "step_per_episode": config["NUM_INNER_STEPS"],
                "path": config["PATH"],
                "grid_size": config.get("GRID_SIZE", (8, 8)),
                "intent_version": config.get("INTENT_VERSION", None),
                "payoff_matrix": config.get("PAYOFF_MATRIX", [1,1,-2]),
                "initial_seed": config.get("INITIAL_SEED", 0),
            },
            clip_actions=True,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=dynamic_policy_mapping_fn,
            policies_to_train=policies_to_train
        )
        .resources(num_gpus=config["NUM_GPUS"]/2)  # Set to 1 if you have a GPU
        .env_runners(num_env_runners=1, 
                    num_gpus_per_env_runner=config["NUM_GPUS"]/2,
                    num_cpus_per_env_runner=int(config["NUM_CPUS"] / 2))
        .training(
                train_batch_size=config["NUM_ENVS"] * config["NUM_INNER_STEPS"],
                lr=config["LR"],
                gamma=config["GAMMA"],
                lambda_=config["GAE_LAMBDA"],
                entropy_coeff=config["ENT_COEF"],
                clip_param=config["CLIP_EPS"],
                vf_loss_coeff=config["VF_COEF"],
                minibatch_size=config["NUM_INNER_STEPS"] // config["NUM_MINIBATCHES"],
                num_epochs=config["NUM_UPDATES"]
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

    return trainer, current_date