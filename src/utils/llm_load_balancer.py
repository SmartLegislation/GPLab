import random
import threading
from typing import List, Dict, Any, Union, Optional
from .llm_clients import OpenaiModel
from .logger import get_logger

logger = get_logger(__name__)

class LLMLoadBalancer:
    def __init__(self, llm_clients: Dict[int, OpenaiModel], config: Dict[str, Any]):
        """
        Initialize the load balancer with LLM clients and configuration.
        
        Args:
            llm_clients: Dictionary mapping indices to OpenaiModel instances
            config: Configuration dictionary containing index arrays and weights
        """
        self.llm_clients = llm_clients
        self.config = config
        self._lock = threading.Lock()  # Thread-safe selection
        
        # Extract weighted client pools
        self.agent_driver_pool = self._build_weighted_pool("agent_driver_llm_index")
        self.other_llm_pool = self._build_weighted_pool("other_llm_index")
        
        logger.info(f"LLM Load Balancer initialized with {len(self.agent_driver_pool)} agent drivers and {len(self.other_llm_pool)} other LLMs")
    
    def _build_weighted_pool(self, config_key: str) -> List[OpenaiModel]:
        """
        Build a weighted pool of LLM clients based on configuration.
        
        Args:
            config_key: Configuration key for the indices (e.g., "agent_driver_llm_index")
            
        Returns:
            List of OpenaiModel instances with weighted representation
        """
        indices = self.config.get(config_key, [])
        
        # Handle single index (backward compatibility)
        if isinstance(indices, int):
            indices = [indices]
        
        weighted_pool = []
        
        for index in indices:
            if index in self.llm_clients:
                client = self.llm_clients[index]
                # Get weight from LLM API config
                llm_api_configs = self.config.get('llm_api_configs', [])
                weight = 1.0
                if 0 <= index < len(llm_api_configs):
                    weight = llm_api_configs[index].get('weight', 1.0)
                
                # Add client multiple times based on weight (rounded to nearest integer)
                weight_count = max(1, int(round(weight)))
                weighted_pool.extend([client] * weight_count)
                
                logger.debug(f"Added LLM client {index} with weight {weight} ({weight_count} instances) to {config_key}")
            else:
                logger.warning(f"LLM client index {index} not found in available clients for {config_key}")
        
        return weighted_pool
    
    def get_agent_driver_client(self, agent_id: Optional[str] = None) -> Optional[OpenaiModel]:
        """
        Get a randomly selected agent driver LLM client based on weights.
        
        Args:
            agent_id: Optional agent ID for logging purposes
            
        Returns:
            OpenaiModel instance or None if no clients available
        """
        with self._lock:
            if not self.agent_driver_pool:
                logger.error("No agent driver LLM clients available")
                return None
            
            selected_client = random.choice(self.agent_driver_pool)
            
            if agent_id:
                logger.debug(f"Selected agent driver LLM for agent {agent_id}: {selected_client.config.description}")
            
            return selected_client
    
    def get_other_llm_client(self, purpose: Optional[str] = None) -> Optional[OpenaiModel]:
        """
        Get a randomly selected other LLM client based on weights.
        
        Args:
            purpose: Optional purpose description for logging
            
        Returns:
            OpenaiModel instance or None if no clients available
        """
        with self._lock:
            if not self.other_llm_pool:
                logger.error("No other LLM clients available")
                return None
            
            selected_client = random.choice(self.other_llm_pool)
            
            if purpose:
                logger.debug(f"Selected other LLM for {purpose}: {selected_client.config.description}")
            
            return selected_client
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get load balancer statistics.
        
        Returns:
            Dictionary with statistics about the load balancer
        """
        return {
            "total_clients": len(self.llm_clients),
            "agent_driver_pool_size": len(self.agent_driver_pool),
            "other_llm_pool_size": len(self.other_llm_pool),
            "available_indices": list(self.llm_clients.keys())
        } 