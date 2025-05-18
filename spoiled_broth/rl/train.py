from spoiled_broth.rl.game_env import GameEnv
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import numpy as np

save_path = Path(__file__).parent / "saved_models" / "ppo_spoiled_broth"


class SimpleLoggingCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1, model=None):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.model =model

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.locals["rewards"])
            print(f"[Step {self.n_calls}] Mean reward: {mean_reward:.3f}")
            if self.model:
                self.model.save(str(save_path))
                print(f"Model saved at {save_path}")
        return True

def main():
    print("Creating environment...")
    env = GameEnv()

    # Wrap BEFORE converting to vec_env
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    # Then convert to VecEnv
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=4, base_class="stable_baselines3")

    print("Starting training...")
    try:
        print('loading model')
        model = PPO.load(save_path, env=env)
    except Exception as e:
        model = PPO("MlpPolicy", env, ent_coef=.05 ,verbose=1)
    model.learn(
        total_timesteps=100_000_000,
        callback=SimpleLoggingCallback(check_freq=10_000, model=model),
    )

    model.save(str(save_path))
    print(f"Model saved at {save_path}")


if __name__ == "__main__":
    main()
