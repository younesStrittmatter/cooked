import os
import pickle
import math
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.tune.registry import register_env
from spoiled_broth.rl.game_env import GameEnv
from spoiled_broth.rl.dynamic_ppo_params import calculate_dynamic_ppo_params
import warnings
from datetime import datetime
from ray.tune.logger import CSVLogger
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

    # --- Load distance_map cache based on map_nr ---
    map_nr = config["MAP_NR"]
    cache_dir = os.path.join(os.path.dirname(__file__), "../maps/distance_cache")
    # Prefer the precomputed .npz cache file in the repo cache
    cache_filename = f"distance_map_{str(map_nr)}.npz"
    cache_path = os.path.join(cache_dir, cache_filename)
    distance_map_path = None
    if os.path.exists(cache_path):
        distance_map_path = cache_path
    else:
        print(f"[WARNING] .npz Distance map cache not found for map_nr '{map_nr}'. Expected path: {cache_path}")

    register_env("spoiled_broth", lambda cfg: ParallelPettingZooEnv(env_creator(cfg)))

    # --- Dynamic policy setup ---
    num_agents = config.get("NUM_AGENTS", 2)
    agent_ids = [f"ai_rl_{i+1}" for i in range(num_agents)]
    policies = {}
    policies_to_train = []
    for agent_id in agent_ids:
        env_cfg = {
            "map_nr": config["MAP_NR"],
            "grid_size": config.get("GRID_SIZE", (8, 8)),
            # pass path (prefer .npz) to distance map; can be None
            "distance_map": distance_map_path,
            "walking_speeds": config.get("WALKING_SPEEDS", None),
            "cutting_speeds": config.get("CUTTING_SPEEDS", None),
            "penalties_cfg": config.get("PENALTIES_CFG", None),
            "rewards_cfg": config.get("REWARDS_CFG", None),
            "dynamic_rewards_cfg": config.get("DYNAMIC_REWARDS_CFG", None),
        }
        policies[f"policy_{agent_id}"] = (
            None,  # Use default PPO policy
            env_creator(env_cfg).observation_space(agent_id),
            env_creator(env_cfg).action_space(agent_id),
            {}
        )
        policies_to_train.append(f"policy_{agent_id}")

    start_epoch = 0
    end_epoch = config["NUM_EPOCHS"]
    start_episode = 0
    checkpoint_episodes = []
    if "CHECKPOINT_ID_USED" in config and not ("PRETRAINED" in config and config["PRETRAINED"] == "Yes"): 
        for agent_id, checkpoint_info in config["CHECKPOINT_ID_USED"].items():
            if checkpoint_info and "path" in checkpoint_info:
                try:
                    checkpoint_path = checkpoint_info["path"]
                    stats_path = os.path.join(checkpoint_info["path"], "training_stats.csv")
                    if os.path.exists(stats_path):
                        with open(stats_path, "r") as f:
                            lines = f.readlines()
                            if lines:
                                last_line = lines[-1].strip()
                                if last_line:
                                    episode_str = last_line.split(",")[0]
                                    try:
                                        checkpoint_episodes.append(int(episode_str))
                                    except Exception:
                                        pass

                except Exception as e:
                    msg = f"\nError loading {agent_id} policy: {e}"
                    checkpoint_log_lines.append(msg)
                    raise

        if checkpoint_episodes:
            start_episode = min(checkpoint_episodes) + 1
            start_epoch = start_episode
            end_epoch = start_epoch + config["NUM_EPOCHS"]

    def dynamic_policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        return f"policy_{agent_id}"

    # --- Model config for MLP ---
    model_config = {
        "fcnet_hiddens": config.get("FCNET_HIDDENS", [256, 256]),
        "fcnet_activation": config.get("FCNET_ACTIVATION", "tanh"),
    }
    
    # Configuration for multi-agent training (time-based episodes)
    # Ensure train_batch_size is a fixed number of transitions, not tied to episode/step count
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
                "game_mode": config["GAME_VERSION"],
                "inner_seconds": config["INNER_SECONDS"],
                "path": config["PATH"],
                "grid_size": config.get("GRID_SIZE", (8, 8)),
                "payoff_matrix": config.get("PAYOFF_MATRIX", [1,1,-2]),
                "initial_seed": config.get("INITIAL_SEED", 0),
                "wait_for_completion": config.get("WAIT_FOR_COMPLETION", True),
                "start_epoch": start_episode,
                "distance_map": distance_map_path,
                "walking_speeds": config.get("WALKING_SPEEDS", None),
                "cutting_speeds": config.get("CUTTING_SPEEDS", None),
                "penalties_cfg": config.get("PENALTIES_CFG", None),
                "rewards_cfg": config.get("REWARDS_CFG", None),
                "dynamic_rewards_cfg": config.get("DYNAMIC_REWARDS_CFG", None),
            },
            clip_actions=True,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=dynamic_policy_mapping_fn,
            policies_to_train=policies_to_train
        )
        .resources(num_gpus=config["NUM_GPUS"]/2)
        .env_runners(num_env_runners=1, 
                    num_gpus_per_env_runner=config["NUM_GPUS"]/2,
                    num_cpus_per_env_runner=int(config["NUM_CPUS"] / 2))
        .training(
                train_batch_size=config.get("TRAIN_BATCH_SIZE", 32),  # Fixed number of transitions
                lr=config["LR"],
                gamma=config["GAMMA"],
                lambda_=config["GAE_LAMBDA"],
                entropy_coeff=config["ENT_COEF"],
                clip_param=config["CLIP_EPS"],
                vf_loss_coeff=config["VF_COEF"],
                minibatch_size=max(1, config.get("TRAIN_BATCH_SIZE", 32) // config["NUM_MINIBATCHES"]),
                num_epochs=config["NUM_UPDATES"]
            )
    )

    # Build algorithm with error handling
    trainer = ppo_config.build_algo()
    print("Algorithm and environment successfully initialized.")

    # Initialize dynamic PPO parameters if configured
    dynamic_ppo_params_cfg = config.get("DYNAMIC_PPO_PARAMS_CFG", None)
    initial_ppo_params = {
        "clip_eps": config["CLIP_EPS"],
        "ent_coef": config["ENT_COEF"]
    }
    
    if dynamic_ppo_params_cfg and dynamic_ppo_params_cfg.get("enabled", False):
        print(f"Dynamic PPO parameters enabled with config: {dynamic_ppo_params_cfg}")
        print(f"Initial PPO params: clip_eps={initial_ppo_params['clip_eps']}, ent_coef={initial_ppo_params['ent_coef']}")

    checkpoint_log_lines = []
    # Load pretrained policies if specified in config
    if "CHECKPOINT_ID_USED" in config: 
        msg = f"Loading pretrained policies from checkpoints:\n {config['CHECKPOINT_ID_USED']}" 
        checkpoint_log_lines.append(msg)
        if "PRETRAINED" in config and config["PRETRAINED"] == "Yes":
            msg = f"\nPRETRAINING: Loading ai_rl_1 policy into all agents."
            checkpoint_log_lines.append(msg)
            # First, find and load the ai_rl_1 policy
            ai_rl_1_policy_module = None

            for agent_id, checkpoint_info in config["CHECKPOINT_ID_USED"].items():
                if agent_id == "ai_rl_1" and checkpoint_info and "path" in checkpoint_info:
                    try:
                        checkpoint_path = checkpoint_info["path"]
                        policy_id = f"policy_{agent_id}"

                        # --- Load the MultiRLModule directly from checkpoint ---
                        rl_module_path = os.path.join(
                            checkpoint_path,
                            "checkpoint_final",
                            "learner_group",
                            "learner",
                            "rl_module"
                        )
                        multi_rl_module = MultiRLModule.from_checkpoint(rl_module_path)

                        if policy_id not in multi_rl_module.keys():
                            raise ValueError(f"Policy {policy_id} not found in the checkpoint.")

                        # --- Extract the ai_rl_1 policy module ---
                        ai_rl_1_policy_module = multi_rl_module[policy_id]
                        msg = f"\nLoaded ai_rl_1 policy module from {checkpoint_path}"
                        checkpoint_log_lines.append(msg)
                        break

                    except Exception as e:
                        msg = f"\nError loading ai_rl_1 policy: {e}"
                        checkpoint_log_lines.append(msg)
                        raise
                    
            if ai_rl_1_policy_module is None:
                raise ValueError(f"\nai_rl_1 policy not found in {config['CHECKPOINT_ID_USED']} config")

            # Now load the ai_rl_1 policy into ALL policies
            for agent_id, checkpoint_info in config["CHECKPOINT_ID_USED"].items():
                if checkpoint_info and "path" in checkpoint_info:
                    try:
                        policy_id = f"policy_{agent_id}"

                        def load_weights(learner):
                            if policy_id in learner.module:
                                current_module = learner.module[policy_id]
                                # Load ai_rl_1 weights into this policy
                                current_module.load_state_dict(ai_rl_1_policy_module.state_dict())

                        trainer.learner_group.foreach_learner(load_weights)

                        msg = f"\nSuccessfully loaded ai_rl_1 policy weights into {agent_id} from {checkpoint_path}"
                        checkpoint_log_lines.append(msg)

                    except Exception as e:
                        msg = f"\nError loading ai_rl_1 weights into {agent_id}: {e}"
                        checkpoint_log_lines.append(msg)
                        raise
        else:
            msg = f"\nNO PRETRAINING: Loading each policy from its own checkpoint."
            checkpoint_log_lines.append(msg)
            # Load each specified policy from its own checkpoint
            for agent_id, checkpoint_info in config["CHECKPOINT_ID_USED"].items():
                if checkpoint_info and "path" in checkpoint_info:
                    try:
                        checkpoint_path = checkpoint_info["path"]
                        policy_id = f"policy_{agent_id}"

                        # --- Load the MultiRLModule directly from checkpoint ---
                        rl_module_path = os.path.join(
                            checkpoint_path,
                            "checkpoint_final",
                            "learner_group",
                            "learner",
                            "rl_module"
                        )
                        multi_rl_module = MultiRLModule.from_checkpoint(rl_module_path)

                        if policy_id not in multi_rl_module.keys():
                            raise ValueError(f"\nPolicy {policy_id} not found in the checkpoint.")

                        def load_weights(learner):
                            if policy_id in learner.module:
                                current_module = learner.module[policy_id]
                                current_module.load_state_dict(multi_rl_module[policy_id].state_dict())

                        trainer.learner_group.foreach_learner(load_weights)

                        msg = f"\nSuccessfully loaded {agent_id} policy weights from {checkpoint_path}"
                        checkpoint_log_lines.append(msg)

                    except Exception as e:
                        msg = f"\nError loading {agent_id} policy: {e}"
                        checkpoint_log_lines.append(msg)
                        raise

            if checkpoint_episodes:
                start_episode = min(checkpoint_episodes) + 1
                start_epoch = start_episode
                end_epoch = start_epoch + config["NUM_EPOCHS"]
    else:
        msg = f"No checkpoint specified, training from scratch."
        checkpoint_log_lines.append(msg)
    
    # Write checkpoint log to file in the training directory
    log_path = os.path.join(path, "checkpoint_load_log.txt")
    with open(log_path, "w") as logf:
        for line in checkpoint_log_lines:
            logf.write(line + "\n")        

    # Save initial checkpoint at episode 0 (or start_episode if resuming)
    initial_episode = start_episode
    initial_checkpoint_path = trainer.save(os.path.join(path, f"checkpoint_{initial_episode}"))
    print(f"Initial checkpoint saved at {initial_checkpoint_path} (episode {initial_episode})")

    for epoch in range(start_epoch, end_epoch):
        # Update dynamic PPO parameters if configured
        if dynamic_ppo_params_cfg and dynamic_ppo_params_cfg.get("enabled", False):
            # Get current episode count from training_stats.csv for parameter updates
            current_episode = epoch  # fallback to epoch if CSV reading fails
            stats_csv_path = os.path.join(path, "training_stats.csv")
            if os.path.exists(stats_csv_path):
                try:
                    with open(stats_csv_path, "r") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if last_line:
                                current_episode = int(last_line.split(",")[0])
                except Exception:
                    pass  # Use fallback value
            
            # Calculate new PPO parameters using the dynamic_ppo_params module
            updated_ppo_params = calculate_dynamic_ppo_params(
                current_episode,
                initial_ppo_params,
                dynamic_ppo_params_cfg
            )
            
            # Update the algorithm's configuration 
            # Note: For RLlib, we'll update the config dict and try to apply changes
            try:
                if "clip_eps" in updated_ppo_params:
                    trainer.config["clip_param"] = updated_ppo_params["clip_eps"]
                if "ent_coef" in updated_ppo_params:
                    trainer.config["entropy_coeff"] = updated_ppo_params["ent_coef"]
                
                # Some RLlib versions might require updating the learner config as well
                if hasattr(trainer, 'learner_group') and trainer.learner_group:
                    def update_learner_config(learner):
                        if "clip_eps" in updated_ppo_params:
                            learner.config["clip_param"] = updated_ppo_params["clip_eps"]
                        if "ent_coef" in updated_ppo_params:
                            learner.config["entropy_coeff"] = updated_ppo_params["ent_coef"]
                    trainer.learner_group.foreach_learner(update_learner_config)
                    
            except Exception as e:
                # If direct config update fails, log it but continue training
                print(f"Warning: Could not update PPO parameters dynamically: {e}")
            
            # Log parameter changes periodically
            if current_episode % 100 == 0 and current_episode > 0:
                affected_params = dynamic_ppo_params_cfg.get("affected_params", [])
                print(f"[Episode {current_episode}] Updated PPO parameters:")
                for param_name in affected_params:
                    if param_name in updated_ppo_params:
                        initial_val = initial_ppo_params[param_name]
                        current_val = updated_ppo_params[param_name]
                        print(f"  {param_name}: {initial_val:.4f} -> {current_val:.4f}")

        result = trainer.train()

        # Log metrics
        if (epoch - start_epoch - 1) % config["SHOW_EVERY_N_EPOCHS"] == 0:
            print(f"Epoch {epoch} complete.")

        # Save checkpoint
        if (epoch - start_epoch - 1) % config["SAVE_EVERY_N_EPOCHS"] == 0:
            # Get current episode count from training_stats.csv for checkpoint naming
            current_episode = epoch  # fallback to epoch if CSV reading fails
            stats_csv_path = os.path.join(path, "training_stats.csv")
            if os.path.exists(stats_csv_path):
                try:
                    with open(stats_csv_path, "r") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if last_line:
                                current_episode = int(last_line.split(",")[0])
                except Exception:
                    pass  # Use fallback value
            
            checkpoint_path = trainer.save(os.path.join(path, f"checkpoint_{current_episode}"))
            print(f"Checkpoint saved at {checkpoint_path} (episode {current_episode})")

    # Get the final episode count from training_stats.csv
    final_episode_count = None
    stats_csv_path = os.path.join(path, "training_stats.csv")
    if os.path.exists(stats_csv_path):
        try:
            with open(stats_csv_path, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        final_episode_count = int(last_line.split(",")[0])
        except Exception as e:
            print(f"Warning: Could not read final episode count from CSV: {e}")

    return trainer, current_date, final_episode_count