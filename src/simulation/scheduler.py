import asyncio
import time as pytime
import os
import datetime
import importlib
import logging # Import logging module
from typing import Dict, Any, List, Type,Optional

from src.utils.config_loader import load_config, get_config_value
from src.utils.data_loader import load_agent_data, sample_agents, get_nested_value
from src.utils.logger import setup_logger, get_logger
from src.utils.persistence import SimulationDB
from src.utils.llm_clients import OpenaiModel, LLMConfig
from src.utils.llm_load_balancer import LLMLoadBalancer
from src.utils.embedding_clients import EmbeddingClient, get_embedding_client_from_config
from src.simulation.time import SimulationTime
from src.subsystems.base import SocialSystemBase
from src.agents.agent import SocialAgent
from src.agents.memory import MemorySystem


class SimpleBlackboard:
    """Simple blackboard implementation for data sharing between subsystems"""
    def __init__(self):
        self.data = {}
        
    def post_data(self, key: str, value: Any, source_system: str = None):
        """Post data to blackboard"""
        self.data[key] = value
        
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data from blackboard"""
        return self.data.get(key, default)


class SimulatorScheduler:
    def __init__(self, config_path: str = "../config/config.yaml"):
        self.config = load_config(config_path)
        self.simulation_id = self._generate_simulation_id()
        self.results_dir = os.path.join("results", self.simulation_id)
        os.makedirs(self.results_dir, exist_ok=True)
        
        log_file = os.path.join(self.results_dir, "simulation.log")
        log_level_str = self.config.get("log_level", "INFO").upper()
        log_level_int = logging.getLevelName(log_level_str) # Convert string to int level
        # Setup root logger handlers and level
        setup_logger(log_file, level=log_level_int) 
        # Get a specific logger for this class, it will use root's handlers
        self.logger = get_logger("GPLab_Simulation")
        # self.logger.setLevel(log_level_int) # Level is set on root, child loggers respect that unless set lower

        self.logger.info(f"Starting simulation: {self.simulation_id}")
        self.logger.info(f"Configuration loaded from: {config_path}")
        self.logger.info(f"Results will be saved to: {self.results_dir}")

        self.sim_time = SimulationTime(
            start_date_str=self.config.get("start_date", "2020-12-01"),
            epoch_duration_days=self.config.get("epoch_duration_days", 30)
        )
        self.num_epochs = self.config.get("num_epochs", 12)
        
        self.db = SimulationDB(os.path.join(self.results_dir, "results.db"))
        self.db.save_config(self.config)

        # Initialize blackboard for data sharing
        self.blackboard = SimpleBlackboard()
        self.logger.info("Blackboard initialized for inter-subsystem communication")

        self.agents: List[SocialAgent] = []
        self.subsystems: Dict[str, SocialSystemBase] = {}
        self.llm_clients: Dict[int, OpenaiModel] = {}
        self.llm_load_balancer: Optional[LLMLoadBalancer] = None
        self.embedding_client: Optional[EmbeddingClient] = None
        
        # QPS control configuration - semaphore will be created in async context
        self.max_concurrent_requests = self.config.get("max_concurrent_requests", 5)
        self.request_semaphore: Optional[asyncio.Semaphore] = None
        self.logger.info(f"QPS control configured: max concurrent requests = {self.max_concurrent_requests}")

    def _generate_simulation_id(self) -> str:
        prefix = self.config.get("simulation_name_prefix", "gplab_sim")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def _initialize_llm_clients(self):
        self.logger.info("Initializing LLM API clients...")
        api_configs = self.config.get('llm_api_configs', [])
        for i, api_conf_dict in enumerate(api_configs):
            try:
                llm_config = LLMConfig(**api_conf_dict)
                self.llm_clients[i] = OpenaiModel(llm_config)
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM client at index {i}: {e}")
        
        if not self.llm_clients:
            self.logger.error("No valid LLM clients could be initialized. Agent decisions may fail.")
        else:
            # Initialize load balancer with the LLM clients
            self.llm_load_balancer = LLMLoadBalancer(self.llm_clients, self.config)
            self.logger.info(f"LLM Load Balancer initialized with {len(self.llm_clients)} clients")
            
            # Log load balancer statistics
            stats = self.llm_load_balancer.get_stats()
            self.logger.info(f"Load Balancer Stats: {stats}")

    def _initialize_embedding_client(self):
        self.logger.info("Initializing Embedding client...")
        try:
            self.embedding_client = get_embedding_client_from_config(self.config)
        except Exception as e:
             self.logger.error(f"Failed to initialize Embedding Client: {e}")
             self.embedding_client = None # Ensure it's None if init fails

    def _initialize_subsystems(self):
        self.logger.info("Initializing active social subsystems...")
        active_system_class_names = self.config.get("active_subsystems", [])
        subsystem_configs = self.config.get("subsystem_configs", {})
        subsystem_directory_group = self.config.get("subsystem_directory_group")

        if not subsystem_directory_group:
            self.logger.error("Configuration key 'subsystem_directory_group' is missing or empty. "
                              "Cannot determine subsystem module path. No subsystems will be loaded.")
            return

        # Construct the path to the specific experiment's subsystem directory
        base_subsystems_path = os.path.join(os.path.dirname(__file__), "..", "subsystems")
        specific_subsystems_dir = os.path.join(base_subsystems_path, subsystem_directory_group)

        if not os.path.isdir(specific_subsystems_dir):
            self.logger.error(f"Subsystem directory '{specific_subsystems_dir}' not found. "
                              "Please check 'subsystem_directory_group' in config and directory structure. "
                              "No subsystems will be loaded.")
            return
        
        self.logger.debug(f"Scanning for subsystem modules in: {os.path.abspath(specific_subsystems_dir)}")

        for class_name in active_system_class_names:
            found_module = False
            for filename in os.listdir(specific_subsystems_dir): # Iterate over specific directory
                if filename.endswith(".py") and filename != "__init__.py" and filename != "base.py":
                    module_name_stem = filename[:-3]
                    # Update module path to include the experiment group
                    module_full_path = f"src.subsystems.{subsystem_directory_group}.{module_name_stem}"
                    # Dynamically import the individual subsystem module
                    subsystem_module = importlib.import_module(module_full_path)
                    if hasattr(subsystem_module, class_name):
                        SubsystemClass: Type[SocialSystemBase] = getattr(subsystem_module, class_name)
                        if not issubclass(SubsystemClass, SocialSystemBase):
                            self.logger.warning(f"Class {class_name} in {module_full_path} is not a subclass of SocialSystemBase. Skipping.")
                            continue
                        found_module = True
                        config_for_subsystem = subsystem_configs.get(class_name, {})
                        
                        instance_kwargs = {"name": class_name, "config": config_for_subsystem, "blackboard": self.blackboard}
                        if class_name == "OpinionSystemExp1": # Specific injections still needed if class has unique deps
                            instance_kwargs["embedding_client"] = self.embedding_client
                            # Use load balancer for LLM client selection
                            if self.llm_load_balancer:
                                sentiment_client = self.llm_load_balancer.get_other_llm_client("sentiment_analysis")
                                instance_kwargs["llm_client_for_sentiment"] = sentiment_client
                            else:
                                self.logger.warning(f"Load balancer not available for {class_name}, using fallback LLM client")
                                other_llm_idx = self.config.get("other_llm_index", 0)
                                if isinstance(other_llm_idx, list) and other_llm_idx:
                                    other_llm_idx = other_llm_idx[0]  # Use first index as fallback
                                instance_kwargs["llm_client_for_sentiment"] = self.llm_clients.get(other_llm_idx)
                        
                        instance = SubsystemClass(**instance_kwargs)
                        self.subsystems[class_name] = instance
                        self.logger.info(f"Successfully initialized subsystem: {class_name} from {module_full_path} with blackboard")
                        break # Found and loaded the class, move to next in active_system_class_names
            if not found_module:
                 self.logger.error(f"Subsystem class '{class_name}' listed in config was not found in any module in {specific_subsystems_dir}/")

        if not self.subsystems:
            self.logger.warning("No subsystems were initialized. Simulation might not be meaningful.")

    async def _initialize_agents_async(self):
        self.logger.info("Initializing social agents...")
        agent_data_path = self.config.get("agent_data_path", "data/cgss_agents.json")
        sample_size = self.config.get("agent_sample_size", 100)
        seed = self.config.get("agent_sample_seed", 42)
        method = self.config.get("agent_sampling_method", "random")

        try:
            all_agent_data = load_agent_data(agent_data_path)
            sampled_agent_data = sample_agents(all_agent_data, sample_size, seed, method)
            self.logger.info(f"Loaded and sampled {len(sampled_agent_data)} agents.")
        except Exception as e:
            self.logger.error(f"Failed to load or sample agent data: {e}", exc_info=True)
            return
            
        # Use load balancer for client selection instead of fixed indices
        if not self.llm_load_balancer:
            self.logger.error("LLM Load Balancer not available. Cannot initialize agents.")
            return

        mem_config = {
            "reflection_threshold": self.config.get("memory_reflection_threshold", 20),
            "reflection_count": self.config.get("memory_reflection_count", 2),
            "time_weight_lambda": self.config.get("memory_time_weight_lambda", 0.5),
            "retrieval_k": self.config.get("memory_retrieval_k", 5)
        }

        for data in sampled_agent_data:
            agent_id = str(get_nested_value(data, "id", f"agent_{len(self.agents)}"))
            
            # Get clients from load balancer
            decision_client = self.llm_load_balancer.get_agent_driver_client(agent_id)
            emotion_client = self.llm_load_balancer.get_other_llm_client("emotion_evaluation")
            reflection_client = self.llm_load_balancer.get_other_llm_client("memory_reflection")
            
            if not decision_client:
                self.logger.error(f"No decision LLM client available for agent {agent_id}. Skipping agent.")
                continue
                
            memory = MemorySystem(
                agent_id=agent_id,
                embedding_client=self.embedding_client,
                reflection_llm_client=reflection_client,
                **mem_config
            )
            agent = SocialAgent(
                agent_data=data,
                decision_llm_client=decision_client,
                emotion_llm_client=emotion_client if emotion_client else decision_client, # Fallback emotion to decision LLM
                memory_system=memory
            )
            self.agents.append(agent)
            # Save static attributes once upon initialization
            try:
                self.db.save_static_agent_attributes(agent.agent_id, agent.static_attributes)
            except Exception as e:
                self.logger.error(f"Failed to save static attributes for agent {agent.agent_id}: {e}")
        
        self.logger.info(f"Initialized {len(self.agents)} agents.")
        
        # Initialize subsystems with agent data
        for name, system in self.subsystems.items():
            system.init(sampled_agent_data)
                # If subsystem needs async init (like OpinionSystem embedding posts)
            if hasattr(system, 'init_async_resources'):
                    await system.init_async_resources()

    async def _run_agent_step_with_qps_control(self, agent, sim_time, subsystems, perception, decision_principles):
        """Run agent step with QPS control to prevent API rate limiting."""
        # Lazy initialize semaphore in async context if not already created
        if self.request_semaphore is None:
            self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            self.logger.debug(f"Created QPS semaphore with {self.max_concurrent_requests} slots")
        
        async with self.request_semaphore:
            return await agent.step(sim_time, subsystems, perception, decision_principles)

    async def run_simulation(self):
        # Initialize QPS control semaphore in async context
        if self.request_semaphore is None:
            self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            self.logger.info(f"QPS control enabled: max concurrent requests = {self.max_concurrent_requests}")
        
        self._initialize_llm_clients()
        self._initialize_embedding_client()
        self._initialize_subsystems()
        await self._initialize_agents_async()

        if not self.agents:
            self.logger.error("No agents initialized. Aborting simulation.")
            return

        start_sim_time = pytime.time()
        self.logger.info(f"Starting simulation run for {self.num_epochs} epochs.")

        for epoch in range(self.num_epochs):
            epoch_start_time = pytime.time()
            self.logger.info(f"--- Starting Epoch {epoch} ({self.sim_time.get_current_time_str()}) ---")

            # Update time in subsystems
            for system in self.subsystems.values():
                system.set_time(self.sim_time)
                system.update_time()
            
             # Precompute embeddings if needed (e.g., for new posts/agents)
            if self.embedding_client:
                self.logger.debug(f"Epoch {epoch}: Precomputing embeddings...")
                tasks = [sys.precompute_embeddings_if_needed() for sys in self.subsystems.values() if hasattr(sys, 'precompute_embeddings_if_needed')]
                await asyncio.gather(*tasks)
                self.logger.debug(f"Epoch {epoch}: Embeddings precomputed.")

            # 1. Get Environment Perception for each agent
            agent_perceptions: Dict[str, Dict[str, Any]] = {}
            for agent in self.agents:
                perception = {}
                for name, system in self.subsystems.items():
                    try:
                        perception[name] = system.get_system_information(agent.agent_id)
                    except Exception as e:
                        self.logger.error(f"Error getting perception for agent {agent.agent_id} from subsystem {name}: {e}", exc_info=True)
                        perception[name] = {"error": "Failed to get information"}
                agent_perceptions[agent.agent_id] = perception

            # 2. Run Agent Steps Asynchronously with QPS control
            # Get agent_decision_principles from config

            agent_decision_principles = self.config.get("agent_decision_principles", [])
            agent_tasks = [
                self._run_agent_step_with_qps_control(
                    agent, 
                    self.sim_time, 
                    self.subsystems, 
                    agent_perceptions[agent.agent_id], 
                    agent_decision_principles
                ) 
                for agent in self.agents
            ]
            agent_decisions_list = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Process decisions, handle potential errors from gather
            agent_decisions_map: Dict[str, Dict[str, Any]] = {}
            for i, result in enumerate(agent_decisions_list):
                agent = self.agents[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Agent {agent.agent_id} step failed: {result}", exc_info=result)
                    agent_decisions_map[agent.agent_id] = agent._generate_default_decision(self.subsystems) # Use default on error
                else:
                     agent_decisions_map[agent.agent_id] = result

            # 3. Update Subsystems Synchronously with all decisions
            for name, system in self.subsystems.items():
                try:
                    # Pass decisions relevant to this subsystem
                    # decisions_for_system = {agent_id: d.get(name, {}) for agent_id, d in agent_decisions_map.items()}
                    system.step(agent_decisions_map)
                except Exception as e:
                    self.logger.error(f"Error during subsystem {name} step: {e}", exc_info=True)

            # 4. Persist results for the epoch
            should_save_prompts = self.config.get("save_decision_prompts", True)
            for agent in self.agents:
                agent_state = agent.get_state_for_persistence()
                prompt_to_save = agent_state.get("decision_input") if should_save_prompts else None
                self.db.save_agent_result(
                    simulation_id=self.simulation_id, 
                    epoch=epoch, 
                    agent_id=agent.agent_id,
                    # static_attrs=agent_state["static_attributes"], # Removed, saved separately
                    memory=agent_state["memory_system"],
                    emotion=agent_state["emotion_state"],
                    perception=agent_state["environment_perception"],
                    decision=agent_state["decision_output"],
                    decision_input=prompt_to_save
                )
            
            for name, system in self.subsystems.items():
                 # Run evaluation within the loop if needed for dynamic state, or just save state
                 # For final evaluation, we call evaluate() after the loop
                 current_state = system.get_state_for_persistence()
                 # Placeholder for metrics if evaluated per epoch
                 epoch_metrics = system.get_evaluation_results() if hasattr(system, 'get_evaluation_results') else {}
                 
                 # Add epoch number to the metrics
                 combined_metrics = {
                     "current_epoch": epoch,
                     "state": current_state,
                     "metrics": epoch_metrics
                 }
                 
                 # Save metrics for this epoch
                 self.db.save_subsystem_result(
                     simulation_id=self.simulation_id,
                     subsystem_name=name,
                     metrics=combined_metrics
                 )

            epoch_duration = pytime.time() - epoch_start_time
            self.logger.info(f"--- Epoch {epoch} finished in {epoch_duration:.2f} seconds ---")
            
            # 5. Advance Time
            self.sim_time.advance_epoch()

        # Simulation End
        end_sim_time = pytime.time()
        total_duration = end_sim_time - start_sim_time
        self.logger.info(f"Simulation finished after {self.num_epochs} epochs.")
        self.logger.info(f"Total simulation time: {total_duration:.2f} seconds.")
        
        # Collect and report token usage statistics
        self._report_token_usage()

        # Final Evaluation
        self.logger.info("Performing final evaluations...")
        for name, system in self.subsystems.items():
            try:
                self.logger.info(f"Evaluating subsystem: {name}")
                # Ensure evaluation methods handle potential async requirements
                if asyncio.iscoroutinefunction(system.evaluate):
                    eval_results = await system.evaluate()
                else:
                    eval_results = system.evaluate()
                
                # Save final evaluation results
                final_metrics = {
                    "final_evaluation": eval_results,
                    "simulation_summary": {
                        "simulation_id": self.simulation_id,
                        "start_time": datetime.datetime.fromtimestamp(start_sim_time).isoformat(),
                        "end_time": datetime.datetime.fromtimestamp(end_sim_time).isoformat(),
                        "duration": total_duration,
                        "num_agents": len(self.agents),
                        "num_epochs": self.num_epochs,
                        "status": "Completed"
                    }
                }
                
                self.db.save_subsystem_result(
                    simulation_id=self.simulation_id,
                    subsystem_name=name,
                    metrics=final_metrics
                )

            except Exception as e:
                self.logger.error(f"Error during final evaluation of subsystem {name}: {e}", exc_info=True)
        
        self.db.close()
        self.logger.info("Simulation run complete. Results saved.")

    def evaluate_simulation_performance(self):
        # Placeholder for evaluating scheduler performance metrics
        # e.g., average epoch time, agent step time distribution, LLM call times
        self.logger.info("Evaluating simulation performance (Placeholder - Not Implemented)")
        # Visualization of performance metrics could be added here
        
    def _report_token_usage(self):
        """Collect and report token usage statistics from all LLM clients"""
        self.logger.info("Token Usage Report:")
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        
        # Collect statistics from all LLM clients
        for idx, client in self.llm_clients.items():
            if isinstance(client, OpenaiModel):
                stats = client.get_token_stats()
                model_name = client.config.description or client.config.model
                prompt_tokens = stats["prompt_tokens"]
                completion_tokens = stats["completion_tokens"]
                total_model_tokens = stats["total_tokens"]
                
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += total_model_tokens
                
                self.logger.info(f"  - Model {model_name}: {prompt_tokens:,} prompt + {completion_tokens:,} completion = {total_model_tokens:,} total tokens")
        
        # Log grand totals
        self.logger.info(f"Total Token Usage: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion = {total_tokens:,} total tokens")
        
        # Save token statistics to database
        try:
            token_metrics = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "models": {f"model_{idx}": client.get_token_stats() for idx, client in self.llm_clients.items() if isinstance(client, OpenaiModel)}
            }
            
            self.db.save_subsystem_result(
                simulation_id=self.simulation_id,
                subsystem_name="token_usage",
                metrics=token_metrics
            )
            self.logger.info("Token usage statistics saved to database")
        except Exception as e:
            self.logger.error(f"Failed to save token usage statistics: {e}")
