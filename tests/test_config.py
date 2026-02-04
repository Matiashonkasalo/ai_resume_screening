import yaml
from src.config import load_config
import pytest

def test_load_valid_config(tmp_path):

    config_data = {
        "experiment": {"name": "test_exp"},
        "model": {"selected": "random_forest"},
        "models": {
            "random_forest": {"class_weight": "balanced"}
        },
        "tuning": {"enabled": True, "n_trials": 5},
        "data": {"processed_dir": "data", "artifacts_dir": "artifacts"},
    }

    config_file = tmp_path / "config.yaml"

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)

    assert config["model"]["selected"] == "random_forest"




def test_invalid_model_name(tmp_path):

    config_data = {
        "experiment": {"name": "test_exp"},
        "model": {"selected": "invalid_model"},
        "models": {},
        "tuning": {"enabled": True, "n_trials": 5},
        "data": {"processed_dir": "data", "artifacts_dir": "artifacts"},
    }

    config_file = tmp_path / "config.yaml"

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValueError):
        load_config(config_file)


def test_selected_model_missing_in_models_section(tmp_path):

    config_data = {
        "experiment": {"name": "test_exp"},
        "model": {"selected": "random_forest"},
        "models": {},  
        "tuning": {"enabled": True, "n_trials": 5},
        "data": {"processed_dir": "data", "artifacts_dir": "artifacts"},
    }

    config_file = tmp_path / "config.yaml"

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValueError):
        load_config(config_file)


def test_valid_config_without_class_weight(tmp_path):

    config_data = {
        "experiment": {"name": "test_exp"},
        "model": {"selected": "random_forest"},
        "models": {
            "random_forest": {}
        },
        "tuning": {"enabled": True, "n_trials": 5},
        "data": {"processed_dir": "data", "artifacts_dir": "artifacts"},
    }

    config_file = tmp_path / "config.yaml"

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)

    assert config["model"]["selected"] == "random_forest"
