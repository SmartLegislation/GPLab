import sqlite3
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

class DatabaseManager:
    """Database manager for simulation results"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            if Path(self.db_path).exists():
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.conn.row_factory = sqlite3.Row  # Enable column access by name
            else:
                raise FileNotFoundError(f"Database file not found: {self.db_path}")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            self.conn = None
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get simulation configuration from database"""
        if not self.conn:
            return {}
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT config_key, config_value FROM simulation_config")
            config = {}
            for row in cursor.fetchall():
                key, value = row
                try:
                    # Try to parse as JSON first
                    config[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, store as string
                    config[key] = value
            return config
        except Exception as e:
            print(f"Error getting simulation config: {e}")
            return {}
    
    def get_agent_count(self) -> int:
        """Get total number of agents"""
        if not self.conn:
            return 0
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT agent_id) FROM agent_static_attributes")
            return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error getting agent count: {e}")
            return 0
    
    def get_max_epoch(self) -> Optional[int]:
        """Get maximum epoch number"""
        if not self.conn:
            return None
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT MAX(epoch) FROM agent_results")
            result = cursor.fetchone()[0]
            return result if result is not None else None
        except Exception as e:
            print(f"Error getting max epoch: {e}")
            return None
    
    def get_agent_list(self) -> List[Dict[str, Any]]:
        """Get list of all agents with their static attributes"""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT agent_id, attributes FROM agent_static_attributes")
            agents = []
            for row in cursor.fetchall():
                agent_id, attributes_json = row
                try:
                    attributes = json.loads(attributes_json)
                    agents.append({
                        'agent_id': agent_id,
                        'attributes': attributes
                    })
                except json.JSONDecodeError:
                    agents.append({
                        'agent_id': agent_id,
                        'attributes': {}
                    })
            return agents
        except Exception as e:
            print(f"Error getting agent list: {e}")
            return []
    
    def get_agent_static_attributes(self, agent_id: str) -> Dict[str, Any]:
        """Get static attributes for a specific agent"""
        if not self.conn:
            return {}
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT attributes FROM agent_static_attributes WHERE agent_id = ?", (agent_id,))
            result = cursor.fetchone()
            if result:
                return json.loads(result[0])
            return {}
        except Exception as e:
            print(f"Error getting agent static attributes: {e}")
            return {}
    
    def get_agent_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get historical data for a specific agent across all epochs"""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT simulation_id, epoch, memory_system, emotion_state, 
                       environment_perception, decision_output, decision_input
                FROM agent_results 
                WHERE agent_id = ? 
                ORDER BY epoch
            """, (agent_id,))
            
            history = []
            for row in cursor.fetchall():
                sim_id, epoch, memory, emotion, perception, decision, decision_input = row
                try:
                    history.append({
                        'simulation_id': sim_id,
                        'epoch': epoch,
                        'memory_system': json.loads(memory) if memory else {},
                        'emotion_state': emotion,
                        'environment_perception': json.loads(perception) if perception else {},
                        'decision_output': json.loads(decision) if decision else {},
                        'decision_input': decision_input
                    })
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for agent {agent_id} epoch {epoch}: {e}")
                    continue
            return history
        except Exception as e:
            print(f"Error getting agent history: {e}")
            return []
    
    def get_subsystem_names(self) -> List[str]:
        """Get list of all subsystem names"""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT subsystem_name FROM subsystem_results")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting subsystem names: {e}")
            return []
    
    def get_subsystem_metrics(self, simulation_id: str, subsystem_name: str) -> Dict[str, Any]:
        """Get metrics for a specific subsystem"""
        if not self.conn:
            return {}
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT metric_name, metric_value, metric_type
                FROM subsystem_results
                WHERE simulation_id = ? AND subsystem_name = ?
            """, (simulation_id, subsystem_name))
            
            metrics = {}
            for row in cursor.fetchall():
                name, value, value_type = row
                try:
                    if value_type in ("dict", "list"):
                        metrics[name] = json.loads(value)
                    elif value_type == "numeric":
                        try:
                            metrics[name] = float(value) if "." in value else int(value)
                        except ValueError:
                            metrics[name] = value
                    else:
                        metrics[name] = value
                except json.JSONDecodeError:
                    metrics[name] = value
            
            return metrics
        except Exception as e:
            print(f"Error getting subsystem metrics: {e}")
            return {}
    
    def get_agent_demographics(self) -> Dict[str, Any]:
        """Get demographic distribution of agents"""
        agents = self.get_agent_list()
        if not agents:
            return {}
        
        demographics = {
            'age_distribution': {},
            'gender_distribution': {},
            'education_distribution': {},
            'income_distribution': {},
            'residence_distribution': {}
        }
        
        for agent in agents:
            attrs = agent.get('attributes', {})
            basic_info = attrs.get('basic_info', {})
            economic_attrs = attrs.get('economic_attributes', {})
            
            # Age distribution
            age = basic_info.get('age', 'Unknown')
            age_group = self._get_age_group(age)
            demographics['age_distribution'][age_group] = demographics['age_distribution'].get(age_group, 0) + 1
            
            # Gender distribution
            gender = basic_info.get('gender', 'Unknown')
            demographics['gender_distribution'][gender] = demographics['gender_distribution'].get(gender, 0) + 1
            
            # Education distribution
            education = basic_info.get('education_level', 'Unknown')
            demographics['education_distribution'][education] = demographics['education_distribution'].get(education, 0) + 1
            
            # Income distribution
            income = economic_attrs.get('income_level', 'Unknown')
            demographics['income_distribution'][income] = demographics['income_distribution'].get(income, 0) + 1
            
            # Residence distribution
            residence = basic_info.get('residence_type', 'Unknown')
            demographics['residence_distribution'][residence] = demographics['residence_distribution'].get(residence, 0) + 1
        
        return demographics
    
    def _get_age_group(self, age) -> str:
        """Convert age to age group"""
        try:
            age_num = int(age)
            if age_num < 18:
                return "Under 18"
            elif age_num < 30:
                return "18-29"
            elif age_num < 40:
                return "30-39"
            elif age_num < 50:
                return "40-49"
            elif age_num < 60:
                return "50-59"
            else:
                return "60+"
        except (ValueError, TypeError):
            return "Unknown"
    
    def get_simulation_summary(self, simulation_id: str) -> Dict[str, Any]:
        """Get comprehensive simulation summary"""
        summary = {
            'simulation_id': simulation_id,
            'config': self.get_simulation_config(),
            'agent_count': self.get_agent_count(),
            'max_epoch': self.get_max_epoch(),
            'subsystems': self.get_subsystem_names(),
            'demographics': self.get_agent_demographics()
        }
        
        # Get subsystem summaries
        summary['subsystem_summaries'] = {}
        for subsystem in summary['subsystems']:
            if subsystem != "token_usage":
                summary['subsystem_summaries'][subsystem] = self.get_subsystem_metrics(simulation_id, subsystem)
        
        return summary
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close() 