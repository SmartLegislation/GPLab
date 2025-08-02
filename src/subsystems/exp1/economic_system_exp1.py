from typing import Dict, Any, List
import statistics
import pandas as pd # Use pandas for easier aggregation
import datetime
from abc import ABC, abstractmethod

from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger
from src.utils.data_loader import get_nested_value
from src.simulation.time import SimulationTime


class EcoSystemExp1(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Any = None):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        self.base_consumption_rate = config.get("base_consumption_rate", 0.6)
        self.base_work_willingness = config.get("base_work_willingness", 0.7)
        self.economic_policies = config.get("economic_policies", {})

        # System state tracking
        self.agent_consumption_willingness: Dict[str, float] = {}
        self.agent_work_willingness: Dict[str, float] = {}
        self.average_consumption_willingness_history: List[float] = []
        self.average_work_willingness_history: List[float] = []
        
        # Store agent attributes needed for evaluation
        self.agent_attributes_for_eval: Dict[str, Dict[str, Any]] = {}
        self.all_epoch_decisions = [] # Store all decisions for final eval

        self.logger.info(f"Initialized EcoSystemExp1 with config: {config}")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        self.logger.info(f"Initializing EcoSystemExp1 state for {len(all_agent_data)} agents.")
        self.agent_consumption_willingness = {}
        self.agent_work_willingness = {}
        self.average_consumption_willingness_history = []
        self.average_work_willingness_history = []
        self.agent_attributes_for_eval = {}
        self.all_epoch_decisions = []
        
        for agent_data in all_agent_data:
            agent_id = get_nested_value(agent_data, "id")
            if agent_id is None:
                 self.logger.warning(f"Agent data missing 'id', skipping: {agent_data}")
                 continue
            agent_id = str(agent_id) # Ensure ID is string
            # Initialize willingness with base values or potentially agent-specific logic later
            self.agent_consumption_willingness[agent_id] = self.base_consumption_rate 
            self.agent_work_willingness[agent_id] = self.base_work_willingness
            
            # Store relevant attributes for later heterogeneous analysis
            attrs = {}
            for attr_key in self.required_agent_attributes:
                 attrs[attr_key] = get_nested_value(agent_data, attr_key, "Unknown")
            self.agent_attributes_for_eval[agent_id] = attrs

        # --- BEGIN MODIFICATION: Store initial average willingness for epoch -1 ---
        if self.agent_consumption_willingness:
            initial_avg_cons = statistics.mean(self.agent_consumption_willingness.values())
            self.average_consumption_willingness_history.append(initial_avg_cons)
        else:
            self.average_consumption_willingness_history.append(0.0)
        
        if self.agent_work_willingness:
            initial_avg_work = statistics.mean(self.agent_work_willingness.values())
            self.average_work_willingness_history.append(initial_avg_work)
        else:
            self.average_work_willingness_history.append(0.0)
        
        self.logger.info(f"Stored initial average willingness (pre-epoch 0): Cons={self.average_consumption_willingness_history[0]:.3f}, Work={self.average_work_willingness_history[0]:.3f}")
        # --- END MODIFICATION ---

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        current_epoch = self.current_time.get_current_epoch()
        self.logger.info(f"EcoSystemExp1 - Epoch {current_epoch} Step Start")
        
        epoch_decisions = []
        for agent_id, decisions in agent_decisions.items():
            eco_decision = decisions.get(self.name, {}) # Get decisions for this subsystem
            cons_willingness = eco_decision.get("consumption_willingness", self.agent_consumption_willingness.get(agent_id, self.base_consumption_rate))
            work_willingness = eco_decision.get("work_willingness", self.agent_work_willingness.get(agent_id, self.base_work_willingness))
            
            # Basic validation/clamping
            self.agent_consumption_willingness[agent_id] = max(0.0, min(1.0, float(cons_willingness)))
            self.agent_work_willingness[agent_id] = max(0.0, min(1.0, float(work_willingness)))
            
            epoch_decisions.append({
                "agent_id": agent_id,
                "epoch": current_epoch,
                "consumption_willingness": self.agent_consumption_willingness[agent_id],
                "work_willingness": self.agent_work_willingness[agent_id]
            })
            
        self.all_epoch_decisions.extend(epoch_decisions)

        # Update aggregate metrics for this epoch
        if self.agent_consumption_willingness:
            avg_cons = statistics.mean(self.agent_consumption_willingness.values())
            # For epoch 0 onwards, append. Initial state is already at index 0.
            if self.current_time.get_current_epoch() == 0 and len(self.average_consumption_willingness_history) == 1:
                # If it's epoch 0 and we only have the initial value, replace it if it was a placeholder, or append if it was the true initial
                # Given the new init logic, the first value is the true initial. So we append for epoch 0.
                self.average_consumption_willingness_history.append(avg_cons)
            elif self.current_time.get_current_epoch() > 0:
                 self.average_consumption_willingness_history.append(avg_cons)
            elif len(self.average_consumption_willingness_history) == 0: # Should not happen with new init logic
                 self.average_consumption_willingness_history.append(avg_cons)

        else:
            # Ensure history has an entry for every epoch, even if 0.0
            if len(self.average_consumption_willingness_history) <= self.current_time.get_current_epoch() +1: # +1 due to initial value
                 self.average_consumption_willingness_history.append(0.0)
        
        if self.agent_work_willingness:
            avg_work = statistics.mean(self.agent_work_willingness.values())
            if self.current_time.get_current_epoch() == 0 and len(self.average_work_willingness_history) == 1:
                self.average_work_willingness_history.append(avg_work)
            elif self.current_time.get_current_epoch() > 0:
                self.average_work_willingness_history.append(avg_work)
            elif len(self.average_work_willingness_history) == 0:
                self.average_work_willingness_history.append(avg_work)
        else:
            if len(self.average_work_willingness_history) <= self.current_time.get_current_epoch() +1:
                self.average_work_willingness_history.append(0.0)
            
        # Replace with a more sophisticated model if needed
        # Log index should be current_epoch + 1 because history now includes initial state at index 0
        log_idx = self.current_time.get_current_epoch() + 1
        self.logger.info(f"EcoSystemExp1 - Epoch {self.current_time.get_current_epoch()} Step End. Avg Cons: {self.average_consumption_willingness_history[log_idx]:.3f}, Avg Work: {self.average_work_willingness_history[log_idx]:.3f}")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        current_epoch_str = str(self.current_time.get_current_epoch())
        current_time_str = self.current_time.get_current_time_str("%Y-%m-%d")
        current_policy = self.economic_policies.get(current_epoch_str, "No specific economic policy update this month.")
        
        # Format historical policies with epoch information and time information
        historical_policies_info = {}
        for epoch, policy_text in self.economic_policies.items():
            # Calculate the date for this historical policy based on epoch
            # Each epoch is typically a month as per the documentation
            # Start with current simulation time and go back by (current_epoch - historical_epoch) months
            current_epoch = self.current_time.get_current_epoch()
            epoch_diff = current_epoch - int(epoch)
            # Calculate approximate date - this assumes regular month progression in simulation
            if epoch_diff > 0:
                historical_date = self.current_time.current_time - datetime.timedelta(days=30 * epoch_diff)
                historical_time_str = historical_date.strftime("%Y-%m-%d")
                historical_policies_info[f"date_{historical_time_str}"] = policy_text

        info = {
            "historical_economic_policies": historical_policies_info,
            "current_economic_policy": {
                "date": current_time_str,
                "policy_text": current_policy
            }
        }
        return info

    def evaluate(self) -> Dict[str, Any]:
        self.logger.info("Evaluating EcoSystemExp1 results.")
        
        if not self.all_epoch_decisions:
            self.logger.warning("No decisions recorded for evaluation.")
            return {}
        
        # Helper function for age binning
        def get_age_bin(age_value: Any) -> str:
            try:
                age = int(age_value) # Age might be string from DB keys
                if 18 <= age <= 30:
                    return '18-30'
                elif 31 <= age <= 45:
                    return '31-45'
                elif 46 <= age <= 60:
                    return '46-60'
                elif age > 60:
                    return '60+'
                else:
                    return 'Other'
            except ValueError:
                return 'Unknown' # If age cannot be converted to int

        # Helper for income binning into four categories
        def get_income_bin(income_value: Any) -> str:
            try:
                # Try to convert to float to handle various formats
                income = float(str(income_value).replace(',', ''))
                if income < 30000:
                    return 'Low Income (<30k)'
                elif 30000 <= income < 100000:
                    return 'Lower Middle Income (30k-100k)'
                elif 100000 <= income < 200000:
                    return 'Upper Middle Income (100k-200k)'
                else:
                    return 'High Income (>200k)'
            except (ValueError, TypeError):
                return 'Unknown' # If income cannot be converted to float
        # Combine decisions with static attributes for heterogeneity analysis
        df = pd.DataFrame(self.all_epoch_decisions)
        agent_attrs_df = pd.DataFrame.from_dict(self.agent_attributes_for_eval, orient='index')
        # Rename index to match agent_id column name if needed
        agent_attrs_df.index.name = 'agent_id' 
        df = pd.merge(df, agent_attrs_df, on='agent_id')

        # Apply binning for Age and Income directly to the DataFrame
        if "Basic Information.Age" in df.columns:
            df['age_group_binned'] = df["Basic Information.Age"].apply(get_age_bin)
        else:
            self.logger.warning("Column 'Basic Information.Age' not found for binning.")
            df['age_group_binned'] = 'Unknown' # Fallback column

        if "Economic Attributes.Annual Personal Income" in df.columns:
            df['income_group_binned'] = df["Economic Attributes.Annual Personal Income"].apply(get_income_bin)
        else:
            self.logger.warning("Column 'Economic Attributes.Annual Personal Income' not found for binning.")
            df['income_group_binned'] = 'Unknown' # Fallback column

        results = {
            "average_consumption_willingness_trend": self.average_consumption_willingness_history,
            "average_work_willingness_trend": self.average_work_willingness_history,
            "consumption_by_epoch_and_group": {},
            "average_consumption_by_group": {},
        }

        # Define grouping keys and how to access the (potentially binned) data for grouping
        # The key in the `grouping_map` is the original attribute name (as expected by visualizer config)
        # The value is the column name in the DataFrame to use for grouping (binned or original)
        grouping_map = {
            "Basic Information.Gender": "Basic Information.Gender",
            "Basic Information.Age": "age_group_binned", 
            "Basic Information.Region": "Basic Information.Region",
            "Basic Information.Marital Status": "Basic Information.Marital Status",
            "Basic Information.Education Level": "Basic Information.Education Level",
            "Economic Attributes.Annual Personal Income": "income_group_binned"
        }

        for original_key, df_column_for_grouping in grouping_map.items():
            if df_column_for_grouping in df.columns:
                try:
                    # Consumption by epoch and group
                    # Group by epoch and the binned/original data column
                    consumption_by_epoch_group = df.groupby(['epoch', df_column_for_grouping])['consumption_willingness'].mean().unstack().to_dict()
                    results["consumption_by_epoch_and_group"][original_key] = consumption_by_epoch_group
                    
                    # Average consumption by group (across all epochs)
                    avg_consumption_group = df.groupby(df_column_for_grouping)['consumption_willingness'].mean().to_dict()
                    results["average_consumption_by_group"][original_key] = avg_consumption_group
                    self.logger.info(f"Successfully processed heterogeneity for grouping key: {original_key} using column {df_column_for_grouping}")
                except Exception as e:
                    self.logger.error(f"Error processing heterogeneity for {original_key} using {df_column_for_grouping}: {e}")
            else:
                 self.logger.warning(f"Column '{df_column_for_grouping}' (for original key '{original_key}') not found in combined data for heterogeneity analysis.")
        
        self.evaluation_results = results # Store for get_evaluation_results
        self.logger.info(f"Final evaluation metrics for {self.name}: {results}")
        self.logger.info("EcoSystemExp1 evaluation complete.")
        return results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        # Return current snapshot of key state variables
        return {
            "average_consumption_willingness": self.average_consumption_willingness_history[-1] if self.average_consumption_willingness_history else None,
            "average_work_willingness": self.average_work_willingness_history[-1] if self.average_work_willingness_history else None,
            # Add other relevant state if needed
        }
