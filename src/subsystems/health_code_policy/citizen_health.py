from typing import Dict, Any, List, Set, Tuple
import random
import networkx as nx
from collections import defaultdict, Counter
from enum import Enum

from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger


class HealthStatus(Enum):
    HEALTHY = "healthy"
    INFECTED = "infected"
    RECOVERED = "recovered"
    

class HealthCodeColor(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class CitizenHealthSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Any = None):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        
        # Health parameters
        self.health_params = config.get("health_params", {})
        self.initial_infection_rate = self.health_params.get("initial_infection_rate", 0.001)
        self.age_risk_factors = self.health_params.get("age_risk_factors", {})
        self.recovery_days_range = self.health_params.get("recovery_days_range", [7, 14])
        
        # Social network parameters
        self.social_network_params = config.get("social_network", {})
        self.avg_connections = self.social_network_params.get("avg_connections", 15)
        self.transmission_prob_by_type = self.social_network_params.get("transmission_probability_by_type", {})
        
        # Agent health tracking
        self.agent_health_status = {}  # agent_id -> HealthStatus
        self.agent_health_codes = {}  # agent_id -> HealthCodeColor
        self.infection_days = {}  # agent_id -> days since infection
        self.quarantine_status = {}  # agent_id -> bool
        
        # Social network
        self.social_network = nx.Graph()
        
        # Agent attributes
        self.agent_attributes = {}
        
        # Metrics tracking
        self.health_metrics_by_epoch = defaultdict(dict)
        self.quarantine_compliance_by_epoch = defaultdict(dict)
        
    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize with agent data and create social network"""
        self.logger.info(f"Initializing {self.name} with {len(all_agent_data)} agents")
        
        # Store agent attributes and initialize health status
        agent_ids = []
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            agent_ids.append(agent_id)
            
            self.agent_attributes[agent_id] = {
                "age": agent_data.get("basic_info", {}).get("age", 30),
                "chronic_conditions": agent_data.get("health_attributes", {}).get("chronic_conditions", [])
            }
            
            # Initialize all as healthy first
            self.agent_health_status[agent_id] = HealthStatus.HEALTHY
            self.agent_health_codes[agent_id] = HealthCodeColor.GREEN
            self.quarantine_status[agent_id] = False
        
        # Ensure minimum initial infections
        min_initial_infections = max(1, int(len(agent_ids) * self.initial_infection_rate))
        initial_infected = random.sample(agent_ids, min_initial_infections)
        
        for agent_id in initial_infected:
            self.agent_health_status[agent_id] = HealthStatus.INFECTED
            self.agent_health_codes[agent_id] = HealthCodeColor.RED
            self.infection_days[agent_id] = random.randint(0, 3)  # Already infected for 0-3 days
            self.quarantine_status[agent_id] = True  # Initially quarantined
        
        # Create social network
        self._create_social_network(agent_ids)
        
        # Post initial infected agents to blackboard
        infected_agents = {aid for aid, status in self.agent_health_status.items() 
                          if status == HealthStatus.INFECTED}
        self._post_to_blackboard("infected_agents", infected_agents)
        self._post_to_blackboard("active_infections", len(infected_agents))
        
        # Initialize system state
        self.system_state = {
            "total_agents": len(all_agent_data),
            "initial_infected": len(infected_agents),
            "network_density": nx.density(self.social_network)
        }
        
        self.logger.info(f"Initialized with {len(infected_agents)} initial infections")
        
    def _create_social_network(self, agent_ids: List[str]):
        """Create a social network between agents"""
        self.logger.info("Creating social network...")
        
        # Add all agents as nodes
        self.social_network.add_nodes_from(agent_ids)
        
        # Create connections based on small-world network model
        for agent_id in agent_ids:
            # Number of connections for this agent
            num_connections = max(1, int(random.gauss(self.avg_connections, 3)))
            num_connections = min(num_connections, len(agent_ids) - 1)
            
            # Select random connections
            potential_connections = [aid for aid in agent_ids if aid != agent_id 
                                   and not self.social_network.has_edge(agent_id, aid)]
            
            if potential_connections:
                connections = random.sample(potential_connections, 
                                          min(num_connections, len(potential_connections)))
                
                for connected_id in connections:
                    # Assign connection type
                    connection_type = random.choice(list(self.transmission_prob_by_type.keys()))
                    self.social_network.add_edge(agent_id, connected_id, 
                                               connection_type=connection_type)
        
        self.logger.info(f"Social network created with {self.social_network.number_of_nodes()} nodes "
                        f"and {self.social_network.number_of_edges()} edges")
        
    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process health decisions and update health states"""
        epoch = self.current_time.get_current_epoch() if self.current_time else 0
        self.logger.info(f"Processing citizen health for epoch {epoch}")
        
        # Get new infections from disease transmission system
        new_infection_list = self._get_from_blackboard("new_infection_list", [])
        
        # Update health states
        self._update_health_states(new_infection_list)
        
        # Process agent health decisions
        compliance_stats = Counter()
        symptom_reports = 0
        
        for agent_id, decisions in agent_decisions.items():
            health_decision = decisions.get(self.name, {})
            
            # Process health check frequency
            check_freq = health_decision.get("health_check_frequency", "when_required")
            
            # Process symptom reporting
            if health_decision.get("report_symptoms") == "yes":
                symptom_reports += 1
                # If reporting symptoms, might trigger testing
                if self.agent_health_status[agent_id] == HealthStatus.INFECTED:
                    self.agent_health_codes[agent_id] = HealthCodeColor.RED
            
            # Process quarantine compliance
            if self.quarantine_status[agent_id]:
                compliance_level = health_decision.get("follow_quarantine", "moderate")
                compliance_stats[compliance_level] += 1
        
        # Social network transmission
        network_infections = self._simulate_network_transmission()
        
        # Update health codes based on contact tracing
        self._update_health_codes()
        
        # Calculate and store metrics
        health_metrics = self._calculate_health_metrics()
        self.health_metrics_by_epoch[epoch] = health_metrics
        self.quarantine_compliance_by_epoch[epoch] = dict(compliance_stats)
        
        # Post to blackboard
        self._post_to_blackboard("infected_agents", health_metrics["infected_set"])
        self._post_to_blackboard("active_infections", health_metrics["total_infected"])
        self._post_to_blackboard("health_code_distribution", health_metrics["health_code_distribution"])
        
        # Log statistics
        self.logger.info(f"Epoch {epoch} health status: {health_metrics['status_distribution']}")
        self.logger.info(f"Health codes: {health_metrics['health_code_distribution']}")
        self.logger.info(f"Network infections: {network_infections}")
        self.logger.info(f"Symptom reports: {symptom_reports}")
        
    def _update_health_states(self, new_infections: List[str]):
        """Update health states including recovery"""
        # Process new infections
        for agent_id in new_infections:
            if self.agent_health_status[agent_id] == HealthStatus.HEALTHY:
                self.agent_health_status[agent_id] = HealthStatus.INFECTED
                self.infection_days[agent_id] = 0
                self.agent_health_codes[agent_id] = HealthCodeColor.RED
                self.quarantine_status[agent_id] = True
        
        # Process recovery
        recovered_agents = []
        for agent_id, days in list(self.infection_days.items()):
            self.infection_days[agent_id] = days + 1
            
            # Check for recovery
            recovery_days = random.randint(*self.recovery_days_range)
            if days >= recovery_days:
                self.agent_health_status[agent_id] = HealthStatus.RECOVERED
                self.agent_health_codes[agent_id] = HealthCodeColor.GREEN
                self.quarantine_status[agent_id] = False
                recovered_agents.append(agent_id)
                del self.infection_days[agent_id]
        
        if recovered_agents:
            self.logger.info(f"{len(recovered_agents)} agents recovered")
    
    def _simulate_network_transmission(self) -> int:
        """Simulate transmission through social network"""
        new_infections = set()
        
        infected_agents = {aid for aid, status in self.agent_health_status.items() 
                          if status == HealthStatus.INFECTED}
        
        for infected_id in infected_agents:
            # Get neighbors in social network
            neighbors = list(self.social_network.neighbors(infected_id))
            
            for neighbor_id in neighbors:
                if self.agent_health_status[neighbor_id] == HealthStatus.HEALTHY:
                    # Get connection type
                    edge_data = self.social_network.get_edge_data(infected_id, neighbor_id)
                    connection_type = edge_data.get("connection_type", "friend")
                    
                    # Get transmission probability
                    base_prob = self.transmission_prob_by_type.get(connection_type, 0.1)
                    
                    # Adjust for age risk
                    neighbor_age = self.agent_attributes[neighbor_id]["age"]
                    age_factor = self._get_age_risk_factor(neighbor_age)
                    
                    # Adjust for chronic conditions
                    chronic_factor = 1.0
                    if self.agent_attributes[neighbor_id]["chronic_conditions"]:
                        chronic_factor = self.health_params.get("chronic_condition_multiplier", 1.8)
                    
                    # Final transmission probability
                    transmission_prob = base_prob * age_factor * chronic_factor
                    
                    # Consider quarantine status
                    if self.quarantine_status[infected_id]:
                        transmission_prob *= 0.1  # 90% reduction if quarantined
                    
                    if random.random() < transmission_prob:
                        new_infections.add(neighbor_id)
        
        # Update health states for network infections
        for agent_id in new_infections:
            self.agent_health_status[agent_id] = HealthStatus.INFECTED
            self.infection_days[agent_id] = 0
            self.agent_health_codes[agent_id] = HealthCodeColor.RED
            self.quarantine_status[agent_id] = True
        
        return len(new_infections)
    
    def _get_age_risk_factor(self, age: int) -> float:
        """Get age-based risk factor"""
        if age <= 18:
            return self.age_risk_factors.get("0-18", 0.5)
        elif age <= 40:
            return self.age_risk_factors.get("19-40", 1.0)
        elif age <= 60:
            return self.age_risk_factors.get("41-60", 1.5)
        else:
            return self.age_risk_factors.get("61+", 2.0)
    
    def _update_health_codes(self):
        """Update health codes based on contact tracing"""
        # Identify close contacts of infected individuals
        yellow_code_agents = set()
        
        for agent_id, status in self.agent_health_status.items():
            if status == HealthStatus.INFECTED:
                # Mark close contacts as yellow code
                neighbors = self.social_network.neighbors(agent_id)
                for neighbor_id in neighbors:
                    if self.agent_health_status[neighbor_id] == HealthStatus.HEALTHY:
                        yellow_code_agents.add(neighbor_id)
        
        # Update health codes
        for agent_id in yellow_code_agents:
            if self.agent_health_codes[agent_id] == HealthCodeColor.GREEN:
                self.agent_health_codes[agent_id] = HealthCodeColor.YELLOW
    
    def _calculate_health_metrics(self) -> Dict[str, Any]:
        """Calculate current health metrics"""
        status_counts = Counter(status.value for status in self.agent_health_status.values())
        code_counts = Counter(code.value for code in self.agent_health_codes.values())
        
        infected_set = {aid for aid, status in self.agent_health_status.items() 
                       if status == HealthStatus.INFECTED}
        
        quarantine_count = sum(1 for q in self.quarantine_status.values() if q)
        
        return {
            "status_distribution": dict(status_counts),
            "health_code_distribution": dict(code_counts),
            "total_infected": len(infected_set),
            "infected_set": infected_set,
            "quarantine_count": quarantine_count,
            "infection_rate": len(infected_set) / len(self.agent_health_status)
        }
    
    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Return health information for a specific agent"""
        # Get neighbor infection status
        neighbor_infection_count = 0
        neighbors = list(self.social_network.neighbors(agent_id))
        
        for neighbor_id in neighbors:
            if self.agent_health_status[neighbor_id] == HealthStatus.INFECTED:
                neighbor_infection_count += 1
        
        # Get current health code restrictions
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        health_code = self.agent_health_codes[agent_id]
        
        # Determine access restrictions based on health code and epoch
        access_restrictions = self._get_access_restrictions(health_code, current_epoch)
        
        return {
            "health_status": self.agent_health_status[agent_id].value,
            "health_code_color": health_code.value,
            "neighbor_infection_status": {
                "total_neighbors": len(neighbors),
                "infected_neighbors": neighbor_infection_count,
                "infection_rate": neighbor_infection_count / max(len(neighbors), 1)
            },
            "quarantine_status": self.quarantine_status[agent_id],
            "days_infected": self.infection_days.get(agent_id, 0),
            "access_restrictions": access_restrictions,
            "health_guidance": self._get_health_guidance(agent_id)
        }
    
    def _get_access_restrictions(self, health_code: HealthCodeColor, epoch: int) -> Dict[str, bool]:
        """Get access restrictions based on health code and current policy epoch"""
        if epoch < 2:  # Before health code system
            return {
                "can_go_work": True,
                "can_go_shopping": True,
                "can_use_transport": True,
                "can_social_activities": True
            }
        elif epoch < 4:  # Strict health code period
            if health_code == HealthCodeColor.GREEN:
                return {
                    "can_go_work": True,
                    "can_go_shopping": True,
                    "can_use_transport": True,
                    "can_social_activities": True
                }
            elif health_code == HealthCodeColor.YELLOW:
                return {
                    "can_go_work": False if epoch >= 3 else True,
                    "can_go_shopping": False,
                    "can_use_transport": False,
                    "can_social_activities": False
                }
            else:  # RED
                return {
                    "can_go_work": False,
                    "can_go_shopping": False,
                    "can_use_transport": False,
                    "can_social_activities": False
                }
        else:  # Relaxed period (epoch 4+)
            if health_code == HealthCodeColor.GREEN:
                return {
                    "can_go_work": True,
                    "can_go_shopping": True,
                    "can_use_transport": True,
                    "can_social_activities": True
                }
            elif health_code == HealthCodeColor.YELLOW:
                return {
                    "can_go_work": True,  # With negative test
                    "can_go_shopping": True,
                    "can_use_transport": True,
                    "can_social_activities": False
                }
            else:  # RED
                return {
                    "can_go_work": False,
                    "can_go_shopping": False,
                    "can_use_transport": False,
                    "can_social_activities": False
                }
    
    def _get_health_guidance(self, agent_id: str) -> str:
        """Get health guidance for the agent"""
        health_status = self.agent_health_status[agent_id]
        health_code = self.agent_health_codes[agent_id]
        quarantine = self.quarantine_status[agent_id]
        
        if health_status == HealthStatus.INFECTED:
            return "You are infected. Stay home, isolate, and seek medical care if symptoms worsen."
        elif health_code == HealthCodeColor.RED:
            return "You have a red health code. You must quarantine and cannot access public places."
        elif health_code == HealthCodeColor.YELLOW:
            return "You have a yellow health code. Limit activities and get tested. Some restrictions apply."
        elif quarantine:
            return "You are in quarantine. Stay home and monitor your health."
        else:
            return "You are healthy. Follow prevention measures and monitor your health."
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the citizen health system"""
        self.logger.info("=== Citizen Health System Evaluation ===")
        
        # Compile time series data
        status_timeline = defaultdict(list)
        code_timeline = defaultdict(list)
        infection_rates = []
        quarantine_counts = []
        
        for epoch in sorted(self.health_metrics_by_epoch.keys()):
            metrics = self.health_metrics_by_epoch[epoch]
            
            # Status distribution over time
            for status in ["healthy", "infected", "recovered"]:
                count = metrics["status_distribution"].get(status, 0)
                status_timeline[status].append(count)
            
            # Health code distribution over time
            for code in ["green", "yellow", "red"]:
                count = metrics["health_code_distribution"].get(code, 0)
                code_timeline[code].append(count)
            
            infection_rates.append(metrics["infection_rate"])
            quarantine_counts.append(metrics["quarantine_count"])
        
        # Network analysis
        network_metrics = {
            "density": nx.density(self.social_network),
            "average_degree": sum(dict(self.social_network.degree()).values()) / self.social_network.number_of_nodes(),
            "clustering_coefficient": nx.average_clustering(self.social_network),
            "connected_components": nx.number_connected_components(self.social_network)
        }
        
        # Quarantine compliance analysis
        compliance_summary = defaultdict(int)
        for epoch_compliance in self.quarantine_compliance_by_epoch.values():
            for level, count in epoch_compliance.items():
                compliance_summary[level] += count
        
        # Age group analysis
        age_group_infections = defaultdict(lambda: {"total": 0, "infected": 0})
        for agent_id, attrs in self.agent_attributes.items():
            age = attrs["age"]
            age_group = "0-18" if age <= 18 else "19-40" if age <= 40 else "41-60" if age <= 60 else "61+"
            
            age_group_infections[age_group]["total"] += 1
            if self.agent_health_status[agent_id] in [HealthStatus.INFECTED, HealthStatus.RECOVERED]:
                age_group_infections[age_group]["infected"] += 1
        
        # Calculate infection rates by age group
        age_infection_rates = {}
        for age_group, data in age_group_infections.items():
            if data["total"] > 0:
                age_infection_rates[age_group] = data["infected"] / data["total"]
        
        evaluation_results = {
            "final_status_distribution": dict(Counter(status.value for status in self.agent_health_status.values())),
            "total_infections": sum(1 for status in self.agent_health_status.values() 
                                  if status in [HealthStatus.INFECTED, HealthStatus.RECOVERED]),
            "peak_infections": max(status_timeline["infected"]) if status_timeline["infected"] else 0,
            "status_timeline": dict(status_timeline),
            "health_code_timeline": dict(code_timeline),
            "infection_rate_timeline": infection_rates,
            "quarantine_timeline": quarantine_counts,
            "network_metrics": network_metrics,
            "quarantine_compliance": dict(compliance_summary),
            "age_group_infection_rates": age_infection_rates,
            "health_code_effectiveness": self._analyze_health_code_effectiveness()
        }
        
        # Log key metrics
        self.logger.info(f"Total infections: {evaluation_results['total_infections']}")
        self.logger.info(f"Peak infections: {evaluation_results['peak_infections']}")
        self.logger.info(f"Final status distribution: {evaluation_results['final_status_distribution']}")
        self.logger.info(f"Network density: {network_metrics['density']:.3f}")
        self.logger.info(f"Age group infection rates: {age_infection_rates}")
        
        # Log time series data
        self.logger.info("Infection timeline:")
        for epoch, count in enumerate(status_timeline["infected"]):
            self.logger.info(f"  Epoch {epoch}: {count} infected")
        
        self.logger.info("Health code distribution timeline:")
        for epoch in range(len(code_timeline["green"])):
            self.logger.info(f"  Epoch {epoch}: Green={code_timeline['green'][epoch]}, "
                           f"Yellow={code_timeline['yellow'][epoch]}, "
                           f"Red={code_timeline['red'][epoch]}")
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def _analyze_health_code_effectiveness(self) -> Dict[str, Any]:
        """Analyze the effectiveness of health code system"""
        effectiveness_analysis = {}
        
        # Compare infection spread with and without yellow codes
        yellow_code_infections = 0
        yellow_code_total = 0
        
        for agent_id, code in self.agent_health_codes.items():
            if code == HealthCodeColor.YELLOW:
                yellow_code_total += 1
                if self.agent_health_status[agent_id] == HealthStatus.INFECTED:
                    yellow_code_infections += 1
        
        if yellow_code_total > 0:
            yellow_to_red_rate = yellow_code_infections / yellow_code_total
            effectiveness_analysis["yellow_code_infection_rate"] = yellow_to_red_rate
            effectiveness_analysis["yellow_code_effectiveness"] = "high" if yellow_to_red_rate < 0.2 else "moderate" if yellow_to_red_rate < 0.5 else "low"
        
        # Analyze quarantine impact
        quarantine_effectiveness = sum(1 for q in self.quarantine_status.values() if q) / len(self.quarantine_status)
        effectiveness_analysis["quarantine_coverage"] = quarantine_effectiveness
        
        return effectiveness_analysis 