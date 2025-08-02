import json
import random
from typing import List, Dict, Any

def load_agent_data(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Agent data file should contain a JSON list of agent objects.")
        return data
    except FileNotFoundError:
        print(f"Error: Agent data file not found at {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON agent data file: {e}")
        raise
    except ValueError as e:
        print(f"Error in agent data format: {e}")
        raise

def sample_agents(
    all_agents: List[Dict[str, Any]],
    sample_size: int,
    seed: int,
    method: str = "random"
) -> List[Dict[str, Any]]:
    if sample_size >= len(all_agents):
        print("Warning: Sample size is larger than or equal to the total number of agents. Using all agents.")
        return all_agents

    random.seed(seed)
    if method == "random":
        return random.sample(all_agents, sample_size)
    else:
        # Implement other sampling methods if needed
        print(f"Warning: Sampling method '{method}' not implemented. Using random sampling.")
        return random.sample(all_agents, sample_size)

def get_nested_value(data: Dict, key_path: str, default: Any = None) -> Any:
    keys = key_path.split('.')
    value = data
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default





