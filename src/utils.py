import os
import yaml
from typing import Callable


def load_config(config_path: str) -> dict:
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: dict, save_dir: str):
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
