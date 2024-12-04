import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.utils import set_random_seed
from envs.MatmulEnv import *
from src.utils import *
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--exp_folder", type=str)
    args = parser.parse_args()
    return args

def main(
    exp_folder: str,
):
    config = load_config(os.path.join(exp_folder, "config.yaml"))
    env_config = config["env"]
    algo = config["train"]["algorithm"]  # Algorithm (SAC, PPO, TD3, etc.)
    datasizes = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 8192]
    env_config["datasizes"] = datasizes
    env_config["episode_len"] = len(env_config["datasizes"])
    eval_env = MatmulEnv(config=env_config, device=1, mode="eval", random_sample=False)

    # Algorithm selection
    ALGO_STR_TO_CLASS = {
        "SAC": SAC,
        "PPO": PPO,
        "TD3": TD3,
    }
    algo_class = ALGO_STR_TO_CLASS[algo]

    # Initialize the model based on the chosen algorithm
    model_folder = os.path.join(exp_folder, "model", "best_model.zip")
    model = algo_class.load(model_folder, env=eval_env)

    obs, _ = eval_env.reset()
    for i in range(env_config["episode_len"]):
        # set_random_seed(random.randint(0, 2**32 - 1))
        # obs, _ = eval_env.reset()
        action, _states = model.predict(obs)
        obs, rewards, terminate, _, info = eval_env.step(action)
        print(info)

if __name__ == "__main__":
    args = get_args()
    main(**vars(args))