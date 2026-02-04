import yaml
from src.validate_config import validate_config

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    validate_config(config)

    return config

