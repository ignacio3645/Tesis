from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str):

    path = PROJECT_ROOT / config_path

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config