from typing import Dict, Any, List, Optional
import datetime
import json

from .memory import MemorySystem
from .prompts.agent_prompt_template import construct_agent_prompt
from src.utils.llm_clients import OpenaiModel, parse_json_response
from src.utils.embedding_clients import EmbeddingClient
from src.utils.data_loader import get_nested_value
from src.utils.logger import get_logger
from src.simulation.time import SimulationTime
from src.subsystems.base import SocialSystemBase

logger = get_logger(__name__)

class SocialAgent:
    def __init__(self, 
                 agent_data: Dict[str, Any],
                 decision_llm_client: OpenaiModel,
                 emotion_llm_client: OpenaiModel, 
                 memory_system: MemorySystem,
                 max_retries: int = 3):
        self.static_attributes = agent_data
        self.agent_id = str(get_nested_value(agent_data, "id", "unknown_agent"))
        self.decision_llm_client = decision_llm_client
        self.emotion_llm_client = emotion_llm_client
        self.memory_system = memory_system
        self.max_retries = max_retries
        
        self.current_emotion: str = "Neutral"
        self.last_decision: Optional[Dict[str, Any]] = None
        self.last_perception: Optional[Dict[str, Any]] = None
        self.last_prompt: Optional[str] = None

        logger.info(f"Initialized SocialAgent: {self.agent_id}")

    async def _update_emotion(self, current_time: datetime.datetime):
        # Retrieve recent memories/actions to gauge emotion
        # Use a generic query relevant to recent experience
        query = "What have I experienced or done recently?"
        recent_memories = await self.memory_system.retrieve_memories(query, current_time)
        history_for_emotion = "\n".join(recent_memories[-5:]) # Use last 5 retrieved memories
        if self.last_decision and 'reasoning' in self.last_decision:
             history_for_emotion += f"\nLast Action Reasoning: {self.last_decision['reasoning']}"

        if history_for_emotion:
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    self.current_emotion = await self.emotion_llm_client.evaluate_emotion(history_for_emotion)
                    logger.debug(f"Agent {self.agent_id} updated emotion to: {self.current_emotion}")
                    break
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Agent {self.agent_id}: Failed to update emotion (attempt {retry_count}/{self.max_retries}): {e}")
                    if retry_count == self.max_retries:
                        logger.error(f"Agent {self.agent_id}: Max retries reached for emotion update, using default.")
                        self.current_emotion = "Neutral" # Fallback
        else:
            self.current_emotion = "Neutral"

    async def step(self, 
                 simulation_time: SimulationTime,
                 active_subsystems: Dict[str, SocialSystemBase], # name -> instance
                 environment_perception: Dict[str, Any],
                 agent_decision_principles: List[str]) -> Dict[str, Any]:
        
        current_dt = simulation_time.current_time
        logger.info(f"Agent {self.agent_id} - Epoch {simulation_time.get_current_epoch()} Step Start")

        # 1. Update Emotion
        await self._update_emotion(current_dt)

        # 2. Summarize current perception (but do not store in memory system yet)
        self.last_perception = environment_perception
        perception_summary = self._summarize_perception(environment_perception)

        # 3. Retrieve Relevant Memories (using current perception to frame the query for PAST memories)
        # Create a query based on current perception to find relevant past experiences.
        if perception_summary:
            memory_query = f"Given the current situation (I am observing: {perception_summary}), what past experiences, observations, or reflections are most relevant to inform my next decision?"
        else:
            memory_query = "What are the most relevant past experiences or reflections to inform my next decision considering my current state?"
        
        relevant_memories = await self.memory_system.retrieve_memories(memory_query, current_dt)

        # 4. Construct Prompt (using retrieved memories and current perception)
        prompt = construct_agent_prompt(
            static_attributes=self.static_attributes,
            relevant_memories=relevant_memories, # These are from the past
            emotion=self.current_emotion,
            environment_perception=environment_perception, # This is the current, full perception for the prompt
            active_subsystems=active_subsystems,
            agent_decision_principles=agent_decision_principles,
            example_decision=self._get_example_decision(active_subsystems)
        )
        self.last_prompt = prompt
        logger.debug(f"Agent {self.agent_id} Prompt:\n{prompt}")

        # 5. Call LLM for Decision with retries
        retry_count = 0
        raw_response = None
        parsed_decision = None

        while retry_count < self.max_retries:
            try:
                raw_response = await self.decision_llm_client.async_text_query(prompt)
                if raw_response:
                    parsed_decision = await parse_json_response(self.decision_llm_client, raw_response)
                    if parsed_decision:
                        self.last_decision = parsed_decision
                        logger.info(f"Agent {self.agent_id} made decision: {self.last_decision}")
                        break
                    else:
                        logger.warning(f"Agent {self.agent_id}: Failed to parse decision JSON (attempt {retry_count + 1}/{self.max_retries})")
                else:
                    logger.warning(f"Agent {self.agent_id}: No response from decision LLM (attempt {retry_count + 1}/{self.max_retries})")
                
                retry_count += 1
                if retry_count == self.max_retries:
                    logger.error(f"Agent {self.agent_id}: Max retries reached, using default decision")
                    self.last_decision = self._generate_default_decision(active_subsystems)
                    parsed_decision = self.last_decision
            except Exception as e:
                retry_count += 1
                logger.warning(f"Agent {self.agent_id}: Error during decision making (attempt {retry_count}/{self.max_retries}): {e}")
                if retry_count == self.max_retries:
                    logger.error(f"Agent {self.agent_id}: Max retries reached after errors, using default decision")
                    self.last_decision = self._generate_default_decision(active_subsystems)
                    parsed_decision = self.last_decision

        if parsed_decision: # Store the action taken and its reasoning
            reasoning = parsed_decision.get("reasoning", "No explicit reasoning provided.")
            decision_summary_text = self._summarize_decision(parsed_decision)
            await self.memory_system.add_memory(f"At {current_dt.strftime('%Y-%m-%d %H:%M')}, following my observation, I decided to {decision_summary_text}. My reasoning was: {reasoning}", current_dt, type="action")

        return self.last_decision

    def _summarize_perception(self, perception: Dict[str, Any]) -> str:
        parts = []
        for sys_name, info in perception.items():
            if isinstance(info, dict):
                summary = ", ".join([f"{k}: {v}" for k, v in info.items()])
                parts.append(f"{sys_name}: {summary}")
            else:
                 parts.append(f"{sys_name}: {info}")
        return "; ".join(parts)
    
    def _summarize_decision(self, decision: Dict[str, Any]) -> str:
        parts = []
        for sys_name, actions in decision.items():
             if sys_name == 'reasoning': continue
             if isinstance(actions, dict):
                 summary = ", ".join([f"{k}={v}" for k, v in actions.items()])
                 parts.append(f"{sys_name} actions ({summary})")
             else:
                 parts.append(f"{sys_name}={actions}")
        return "; ".join(parts) if parts else "take default actions"

    def _get_example_decision(self, active_subsystems: Dict[str, SocialSystemBase]) -> Dict[str, Any]:
        example = {"reasoning": "Explain why you're doing it"}
        for name, system in active_subsystems.items():
            example[name] = {}
            for attr_config in system.decision_attributes: # attr_config is now a dict
                attr_name = attr_config.get("name")
                attr_description = attr_config.get("description", "<your decision value>")
                if attr_name:
                    example[name][attr_name] = attr_description
        return example

    def _generate_default_decision(self, active_subsystems: Dict[str, SocialSystemBase]) -> Dict[str, Any]:
        logger.warning(f"Agent {self.agent_id}: Generating default decision.")
        decision = {"reasoning": "Failed to generate a valid decision via LLM, using defaults."}
        for name, system in active_subsystems.items():
            default_sys_decision = {}
            for attr_config in system.decision_attributes: # attr_config is a dict {name: str, description: str}
                attr_name = attr_config.get("name")
                if not attr_name: continue

                # Try to infer a very basic default value type based on name or description
                # This is still rudimentary and ideally subsystems have their own default logic
                if "willingness" in attr_name or "rate" in attr_name or "value" in attr_name:
                    default_sys_decision[attr_name] = 0.5 
                elif "action" in attr_name.lower() and ("list" in attr_config.get("description","").lower() or "array" in attr_config.get("description","").lower()):
                    default_sys_decision[attr_name] = [] 
                elif "action" in attr_name.lower(): # single action might be a dict or string
                    default_sys_decision[attr_name] = {} # Or None, depends on expected type
                else:
                    default_sys_decision[attr_name] = None # Generic default
            decision[name] = default_sys_decision
        return decision

    def get_state_for_persistence(self) -> Dict[str, Any]:
        return {
            "static_attributes": self.static_attributes, # Still needed for initial save by scheduler
            "memory_system": self.memory_system.get_status_for_persistence(),
            "emotion_state": self.current_emotion,
            "environment_perception": self.last_perception, # Last observed environment
            "decision_output": self.last_decision, # Last generated decision
            "decision_input": self.last_prompt # The prompt that led to the last decision
        }

