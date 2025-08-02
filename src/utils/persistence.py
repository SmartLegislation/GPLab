import sqlite3
import json
import os
from typing import Dict, Any, List, Optional

from .logger import get_logger

logger = get_logger(__name__)

class SimulationDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Successfully connected to database: {self.db_path}")
            self._create_tables()
        except sqlite3.Error as e:
            logger.error(f"Error connecting to or creating database {self.db_path}: {e}")
            raise

    def _create_tables(self):
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulation_config (
                config_key TEXT PRIMARY KEY,
                config_value TEXT
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_static_attributes (
                agent_id TEXT PRIMARY KEY,
                attributes TEXT
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_results (
                simulation_id TEXT,
                epoch INTEGER,
                agent_id TEXT, 
                memory_system TEXT,     
                emotion_state TEXT,     
                environment_perception TEXT,
                decision_output TEXT,
                decision_input TEXT, 
                PRIMARY KEY (simulation_id, epoch, agent_id),
                FOREIGN KEY (agent_id) REFERENCES agent_static_attributes(agent_id)
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS subsystem_results (
                simulation_id TEXT,
                subsystem_name TEXT,
                metric_name TEXT,
                metric_value TEXT,
                metric_type TEXT,
                PRIMARY KEY (simulation_id, subsystem_name, metric_name)
            )
            """)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")

    def save_config(self, config_data: Dict[str, Any]):
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            for key, value in config_data.items():
                cursor.execute("INSERT OR REPLACE INTO simulation_config (config_key, config_value) VALUES (?, ?)",
                               (key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)))
            self.conn.commit()
            logger.info("Simulation configuration saved to database.")
        except sqlite3.Error as e:
            logger.error(f"Error saving configuration: {e}")

    def save_static_agent_attributes(self, agent_id: str, static_attrs: Dict[str, Any]):
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT OR IGNORE INTO agent_static_attributes (agent_id, attributes)
            VALUES (?, ?)
            """, (agent_id, json.dumps(static_attrs, ensure_ascii=False)))
            self.conn.commit()
            logger.debug(f"Static attributes for agent {agent_id} saved/ensured.")
        except sqlite3.Error as e:
            logger.error(f"Error saving static attributes for agent {agent_id}: {e}")

    def save_agent_result(self, simulation_id: str, epoch: int, agent_id: str, 
                            memory: Dict, emotion: str, 
                            perception: Dict, decision: Dict, decision_input: Optional[str]):
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO agent_results 
            (simulation_id, epoch, agent_id, memory_system, emotion_state, environment_perception, decision_output, decision_input)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (simulation_id, epoch, agent_id, 
                  json.dumps(memory, ensure_ascii=False), emotion,
                  json.dumps(perception, ensure_ascii=False), json.dumps(decision, ensure_ascii=False),
                  decision_input))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving agent result for agent {agent_id} at epoch {epoch}: {e}")

    def save_subsystem_result(self, simulation_id: str, subsystem_name: str, metrics: Dict[str, Any]):
        """
        Save subsystem metrics to the database.
        Each metric is stored as a separate row with its name and value.
        
        Args:
            simulation_id: Unique identifier for the simulation
            subsystem_name: Name of the subsystem
            metrics: Dictionary containing metrics to save
        """
        if not self.conn:
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Process and save each metric
            for metric_name, metric_value in self._flatten_metrics(metrics):
                # Determine metric type based on value
                if isinstance(metric_value, dict):
                    value_type = "dict"
                    value_json = json.dumps(metric_value, ensure_ascii=False)
                elif isinstance(metric_value, list):
                    value_type = "list"
                    value_json = json.dumps(metric_value, ensure_ascii=False)
                elif isinstance(metric_value, (int, float)):
                    value_type = "numeric"
                    value_json = str(metric_value)
                else:
                    value_type = "string"
                    value_json = str(metric_value)
                
                cursor.execute("""
                INSERT OR REPLACE INTO subsystem_results
                (simulation_id, subsystem_name, metric_name, metric_value, metric_type)
                VALUES (?, ?, ?, ?, ?)
                """, (simulation_id, subsystem_name, metric_name, value_json, value_type))
            
            self.conn.commit()
            logger.debug(f"Saved {len(metrics)} metrics for subsystem {subsystem_name}")
        except sqlite3.Error as e:
            logger.error(f"Error saving subsystem metrics for {subsystem_name}: {e}")

    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> List[tuple]:
        """
        Flatten nested metrics dictionary into a list of (name, value) tuples.
        For nested dictionaries, the keys are joined with dots.
        
        Args:
            metrics: Dictionary of metrics, potentially nested
            prefix: Prefix for nested keys
            
        Returns:
            List of (metric_name, metric_value) tuples
        """
        result = []
        
        for key, value in metrics.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict) and all(isinstance(k, (int, str)) for k in value.keys()):
                # If it's a dictionary with simple keys (like epoch numbers), store as a single metric
                result.append((full_key, value))
            elif isinstance(value, dict):
                # For other dictionaries, recurse to flatten
                result.extend(self._flatten_metrics(value, full_key))
            else:
                # For non-dict values, add directly
                result.append((full_key, value))
                
        return result

    def get_latest_simulation_id(self) -> Optional[str]:
        """Get the most recent simulation ID from the database."""
        if not self.conn:
            return None
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT DISTINCT simulation_id FROM agent_results
            ORDER BY simulation_id DESC LIMIT 1
            """)
            result = cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Error getting latest simulation ID: {e}")
            return None

    def get_subsystem_metrics(self, simulation_id: str, subsystem_name: str) -> Dict[str, Any]:
        """
        Get all metrics for a specific subsystem in a simulation.
        
        Args:
            simulation_id: ID of the simulation
            subsystem_name: Name of the subsystem
            
        Returns:
            Dictionary of metrics
        """
        if not self.conn:
            return {}
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT metric_name, metric_value, metric_type
            FROM subsystem_results
            WHERE simulation_id = ? AND subsystem_name = ?
            """, (simulation_id, subsystem_name))
            
            results = {}
            for name, value, value_type in cursor.fetchall():
                if value_type in ("dict", "list"):
                    results[name] = json.loads(value)
                elif value_type == "numeric":
                    try:
                        if "." in value:
                            results[name] = float(value)
                        else:
                            results[name] = int(value)
                    except ValueError:
                        results[name] = value
                else:
                    results[name] = value
                    
            return results
        except sqlite3.Error as e:
            logger.error(f"Error getting metrics for subsystem {subsystem_name}: {e}")
            return {}

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info(f"Database connection closed: {self.db_path}")





