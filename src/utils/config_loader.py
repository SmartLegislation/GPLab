import yaml
from typing import Dict, Any

def load_config(config_path: str = "../config/config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        raise

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    keys = key_path.split('.')
    value = config
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default





