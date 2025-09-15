import os
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.algorithms.ppo import PPO
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

    start_epoch = 0
    end_epoch = config["NUM_EPOCHS"]
    checkpoint_epochs = []
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
                                    epoch_str = last_line.split(",")[0]
                                    try:
                                        checkpoint_epochs.append(int(epoch_str))
                                    except Exception:
                                        pass

                except Exception as e:
                    msg = f"\nError loading {agent_id} policy: {e}"
                    checkpoint_log_lines.append(msg)
                    raise

        if checkpoint_epochs:
            start_epoch = min(checkpoint_epochs) + 1
            end_epoch = start_epoch + config["NUM_EPOCHS"]

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
                "wait_for_completion": config.get("WAIT_FOR_COMPLETION", True),
                "start_epoch": start_epoch
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
    print("Algorithm and environment successfully initialized.")

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

            if checkpoint_epochs:
                start_epoch = min(checkpoint_epochs) + 1
                end_epoch = start_epoch + config["NUM_EPOCHS"]
    else:
        msg = f"No checkpoint specified, training from scratch."
        checkpoint_log_lines.append(msg)
    
    # Write checkpoint log to file in the training directory
    log_path = os.path.join(path, "checkpoint_load_log.txt")
    with open(log_path, "w") as logf:
        for line in checkpoint_log_lines:
            logf.write(line + "\n")        

    for epoch in range(start_epoch, end_epoch):
        result = trainer.train()

        # Log metrics
        if (epoch - start_epoch) % config["SHOW_EVERY_N_EPOCHS"] == 0:
            print(f"Epoch {epoch} complete.")

        # Save checkpoint
        if (epoch - start_epoch) % config["SAVE_EVERY_N_EPOCHS"] == 0:
            checkpoint_path = trainer.save(os.path.join(path, f"checkpoint_{epoch}"))
            print(f"Checkpoint saved at {checkpoint_path}")

    return trainer, current_date