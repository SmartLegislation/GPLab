'''
Author: FoeverTree 11965818+cromwell_mao@user.noreply.gitee.com
Date: 2025-05-13 11:34:34
LastEditors: FoeverTree 11965818+cromwell_mao@user.noreply.gitee.com
LastEditTime: 2025-05-13 11:41:42
FilePath: \GPLab\src\subsystems\base.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from src.simulation.time import SimulationTime
# Assume SharedBlackboard will be in src.utils.shared_blackboard
# We cannot add imports for non-existent files/modules,
# so this will be a conceptual change.
# from src.utils.shared_blackboard import SharedBlackboard # User needs to create this

class SocialSystemBase(ABC):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Optional[Any] = None): # blackboard: Optional[SharedBlackboard]
        self.name = name
        self.config = config
        self.blackboard = blackboard # Store blackboard instance
        self.current_time: SimulationTime = None 
        self.system_state: Dict[str, Any] = {} 
        self.required_agent_attributes: List[str] = config.get("required_agent_attributes", [])
        self.environment_attributes: List[str] = config.get("environment_attributes", [])
        # Decision attributes can now be a list of dicts {name: str, description: str}
        # or a list of strings (for backward compatibility or simple cases)
        raw_decision_attributes = config.get("decision_attributes", [])
        self.decision_attributes: List[Dict[str, str]] = []
        if raw_decision_attributes and isinstance(raw_decision_attributes[0], dict):
            self.decision_attributes = raw_decision_attributes
        else: # Assume list of strings
            self.decision_attributes = [{"name": attr, "description": "<your decision value>"} for attr in raw_decision_attributes]

        self.evaluation_results: Dict[str, Any] = {} 

    def _post_to_blackboard(self, key: str, value: Any):
        if self.blackboard:
            # Construct a unique key if desired, e.g., using system name
            # Example: full_key = f"{self.name}.{key}"
            # self.blackboard.post_data(full_key, value, source_system=self.name)
            self.blackboard.post_data(key, value, source_system=self.name) # Using original key for simplicity now
        else:
            from src.utils.logger import get_logger
            logger = get_logger(self.name)
            logger.warning(f"Blackboard not available. Cannot post data for key: {key}")

    def _get_from_blackboard(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        if self.blackboard:
            return self.blackboard.get_data(key, default)
        else:
            from src.utils.logger import get_logger
            logger = get_logger(self.name)
            logger.warning(f"Blackboard not available. Cannot get data for key: {key}. Returning default.")
            return default

    def set_time(self, simulation_time: SimulationTime):
        self.current_time = simulation_time

    @abstractmethod
    def init(self, all_agent_data: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        # Processes decisions from *all* agents for this subsystem in the current epoch
        pass

    @abstractmethod
    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        # Returns environment information relevant *to a specific agent*
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        pass

    def update_time(self):
        # This method is primarily managed by the scheduler's set_time
        # Subsystems can override if they need specific time-based updates internally
        # For now, just log the time update
        from src.utils.logger import get_logger
        logger = get_logger(self.name)
        if self.current_time:
            logger.debug(f"Time updated to epoch {self.current_time.get_current_epoch()}, time {self.current_time.get_current_time_str()}")
        else:
             logger.warning("Attempted to update time, but current_time is not set.")

    def get_state_for_persistence(self) -> Dict[str, Any]:
        # Return system state suitable for saving to DB
        # Can be overridden by subclasses if state needs special handling
        return self.system_state

    def get_evaluation_results(self) -> Dict[str, Any]:
        return self.evaluation_results





