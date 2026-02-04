SUPPORTED_MODELS = {"random_forest", "adaboost", "xgboost"}
SUPPORTED_CLASS_WEIGHT = {"balanced", None}


def validate_config(config):

    # Required top-level sections

    required_sections = ["experiment", "model", "models", "tuning", "data"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing config section: {section}")

    # Experiment

    if "name" not in config["experiment"]:
        raise ValueError("Missing experiment.name")

    # Model selection

    if "selected" not in config["model"]:
        raise ValueError("Missing model.selected")

    selected_model = config["model"]["selected"]

    if selected_model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {selected_model}")

    # Model-specific config

    if selected_model not in config["models"]:
        raise ValueError(f"Missing config for selected model: {selected_model}")

    model_cfg = config["models"][selected_model]

    class_weight = model_cfg.get("class_weight")

    if class_weight not in SUPPORTED_CLASS_WEIGHT:
        raise ValueError(f"Unsupported class_weight: {class_weight}")

    # Tuning validation

    if not isinstance(config["tuning"]["enabled"], bool):
        raise ValueError("tuning.enabled must be boolean")

    if not isinstance(config["tuning"]["n_trials"], int):
        raise ValueError("tuning.n_trials must be int")

    # Data validation

    required_data_keys = ["processed_dir", "artifacts_dir"]

    for key in required_data_keys:
        if key not in config["data"]:
            raise ValueError(f"Missing data.{key}")
