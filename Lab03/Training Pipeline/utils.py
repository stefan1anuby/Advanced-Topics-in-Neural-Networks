import torch
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_device(device_type="cuda"):
    return torch.device("cuda" if torch.cuda.is_available() and device_type == "cuda" else "cpu")
