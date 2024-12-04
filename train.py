import time
import os
import torch
import wandb
import random
from gymnasium.wrappers import FrameStack
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    CallbackList,
)
from callback import CustomEvalCallback
from wandb.integration.sb3 import WandbCallback
from argparse import ArgumentParser
from envs.MatmulEnv import MatmulEnv
from src.utils import load_config, save_config

# Register the custom Matmul environment for training
register(id="matmul", entry_point="envs.MatmulEnv:MatmulEnv")

def get_args():
    """Parse and return command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file.")
    args = parser.parse_args()
    return args

def train(config: dict):
    """
    Main training loop for reinforcement learning using Stable-Baselines3.

    Args:
        config (dict): Training and environment configuration loaded from the config file.
    """
    # General configuration
    general_config = config["general"]
    name = general_config["name"]
    log_wandb = general_config["log_wandb"]
    mode = general_config["mode"]
    verbose = general_config["verbose"]

    # Training-specific configuration
    train_config = config["train"]
    n_envs = train_config["n_envs"]  # Number of parallel environments
    algo = train_config["algorithm"]  # Algorithm (SAC, PPO, TD3, etc.)
    seed = train_config["seed"]
    total_timesteps = train_config["total_timesteps"]
    model_save_freq = train_config["model_save_freq"]
    eval_freq = train_config["eval_freq"]
    n_eval_episodes = train_config["n_eval_episodes"]
    framestack = train_config["framestack"]
    random_sample = train_config["random_sample"]

    # Environment configuration
    env_config = config["env"]
    env_id = env_config["env_id"]
    cudagraph = env_config["cudagraph"]
    sim = env_config["sim"]
    fp = env_config["fp"]  # Floating point precision
    matrix_size = env_config["size"]
    datasizes = env_config["datasizes"]
    train_device = env_config["train_dev"]
    eval_device = env_config["eval_dev"]
    warmup = env_config["warmup"]
    repeat = env_config["repeat"]

    # Set random seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    set_random_seed(seed)
    print("Training with seed:", seed)

    # Set experiment folder and save the config
    if mode =="search":
        data_folder = "./data"
    else:
        data_folder = "./exp"
    date_stamp = time.strftime("%Y%m%d-%H%M")[2:]
    exp_name = f"{algo}--{name}--{date_stamp}"
    exp_folder = os.path.join(data_folder, f"FP{fp}-S{matrix_size}", exp_name)
    print("Saving experiment to:", exp_folder)
    os.makedirs(exp_folder, exist_ok=True)
    save_config(config, exp_folder)

    env = MatmulEnv(
        config=env_config,
        exp_folder=exp_folder,
        device=train_device,
        mode=mode,
        random_sample=random_sample,
    )
    if framestack:
        env = FrameStack(env, framestack)

    # Set the device for evaluation (GPU)
    torch.cuda.set_device(eval_device)

    # Algorithm selection
    ALGO_STR_TO_CLASS = {
        "SAC": SAC,
        "PPO": PPO,
        "TD3": TD3,
    }
    algo_class = ALGO_STR_TO_CLASS[algo]

    # Initialize the model based on the chosen algorithm
    model = algo_class("MlpPolicy", env, verbose=verbose)

    # Setup Weights & Biases (WandB) logging if enabled
    if log_wandb:
        if sim:
            project_name = "IBM-RLxTRITON-sim"
        else:
            project_name = "IBM-RLxTRITON"
        run = wandb.init(
            project=project_name,
            name=exp_name,
            sync_tensorboard=True,  # Sync TensorBoard logs to WandB
            monitor_gym=True,  # Upload videos of agent behavior
            save_code=True,  # Save the code snapshot
        )

    # Set evaluation environment and callback configuration
    eval_env = MatmulEnv(config=env_config, device=eval_device, mode="eval", random_sample=False)
    if framestack:
        eval_env = FrameStack(eval_env, framestack)

    # Set up evaluation and logging callbacks
    callback_list = [
        CustomEvalCallback(
            eval_env=eval_env,
            eval_freq=eval_freq,
            save_freq=model_save_freq,
            n_eval_episodes=n_eval_episodes,
            best_model_save_path=f"{exp_folder}/model",
            log_wandb=log_wandb,
            deterministic=True,
            sim=sim,
        ),
    ]

    if log_wandb:
        # Add WandBCallback for further logging
        callback_list.append(
            WandbCallback(
                gradient_save_freq=10,
                model_save_freq=model_save_freq,
                model_save_path=exp_folder,
                verbose=verbose,
                log="all",
            )
        )

    # Combine all callbacks
    callback = CallbackList(callback_list)

    # Start the training process
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        callback=callback,
        tb_log_name=exp_name,
        progress_bar=False if verbose else True,
    )

    # Save the final model
    model.save(f"{exp_folder}/model")

if __name__ == "__main__":
    # Parse command-line arguments and load config file
    args = get_args()
    config = load_config(args.config)
    # Start training
    train(config)