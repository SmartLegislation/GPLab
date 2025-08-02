import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigLoader:
    """Configuration loader for YAML files"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                print(f"Configuration file not found: {self.config_path}")
                return {}
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {}
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration"""
        return self.config
    
    def get_llm_configs(self) -> list:
        """Get LLM API configurations"""
        return self.config.get('llm_api_configs', [])
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent-related configuration"""
        return {
            'agent_data_path': self.config.get('agent_data_path', ''),
            'agent_sample_size': self.config.get('agent_sample_size', 100),
            'agent_sample_seed': self.config.get('agent_sample_seed', 42),
            'agent_sampling_method': self.config.get('agent_sampling_method', 'random'),
            'memory_reflection_threshold': self.config.get('memory_reflection_threshold', 5),
            'memory_reflection_count': self.config.get('memory_reflection_count', 2),
            'memory_time_weight_lambda': self.config.get('memory_time_weight_lambda', 0.8),
            'memory_retrieval_k': self.config.get('memory_retrieval_k', 5),
            'agent_decision_principles': self.config.get('agent_decision_principles', [])
        }
    
    def get_subsystem_configs(self) -> Dict[str, Any]:
        """Get subsystem configurations"""
        return {
            'subsystem_directory_group': self.config.get('subsystem_directory_group', ''),
            'active_subsystems': self.config.get('active_subsystems', []),
            'subsystem_configs': self.config.get('subsystem_configs', {})
        }
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get simulation parameters"""
        return {
            'simulation_name_prefix': self.config.get('simulation_name_prefix', 'gplab_sim'),
            'num_epochs': self.config.get('num_epochs', 12),
            'start_date': self.config.get('start_date', '2024-01-01'),
            'epoch_duration_days': self.config.get('epoch_duration_days', 30),
            'save_decision_prompts': self.config.get('save_decision_prompts', True),
            'log_level': self.config.get('log_level', 'INFO')
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding model configuration"""
        return {
            'use_local_embedding_model': self.config.get('use_local_embedding_model', False),
            'local_embedding_model_path': self.config.get('local_embedding_model_path', ''),
            'embedding_dimension': self.config.get('embedding_dimension', 384),
            'embedding_device': self.config.get('embedding_device', 'cpu'),
            'embedding_api_index': self.config.get('embedding_api_index', [0])
        }
    
    def get_qps_config(self) -> Dict[str, Any]:
        """Get QPS control configuration"""
        return {
            'max_concurrent_requests': self.config.get('max_concurrent_requests', 5),
            'agent_driver_llm_index': self.config.get('agent_driver_llm_index', [0]),
            'other_llm_index': self.config.get('other_llm_index', [0])
        }
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration value"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_formatted_config(self) -> str:
        """Get formatted configuration as YAML string"""
        try:
            return yaml.dump(self.config, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            return f"Error formatting configuration: {e}" 