from typing import Dict, Any, List
import random
from collections import defaultdict, Counter

from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger


class DiseaseTransmissionSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Any = None):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        
        # Transmission parameters
        self.transmission_params = config.get("transmission_params", {})
        self.base_infection_rate = self.transmission_params.get("base_infection_rate", 0.02)
        
        # Health code impact parameters
        self.health_code_impact = config.get("health_code_impact", {})
        
        # Health code enforcement parameters - NEW
        self.health_code_enforcement = config.get("health_code_enforcement", {})
        
        # Agent mobility tracking
        self.agent_mobility = defaultdict(dict)  # agent_id -> {go_out, destination, transport}
        self.daily_contacts = defaultdict(list)  # agent_id -> [contact_ids]
        
        # Location crowding metrics
        self.location_crowding = defaultdict(int)
        self.transport_usage = defaultdict(int)
        
        # Infection tracking
        self.new_infections = defaultdict(int)  # epoch -> count
        self.cumulative_infections = 0
        
        # Agent attributes
        self.agent_attributes = {}
        
    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize with agent data"""
        self.logger.info(f"Initializing {self.name} with {len(all_agent_data)} agents")
        
        # Store relevant agent attributes
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            self.agent_attributes[agent_id] = {
                "age": agent_data.get("basic_info", {}).get("age", 30),
                "employment_status": agent_data.get("economic_attributes", {}).get("employment_status", "employed")
            }
        
        # Initialize system state
        self.system_state = {
            "total_agents": len(all_agent_data),
            "current_infection_rate": self.base_infection_rate,
            "restriction_level": 0
        }
        
    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent mobility decisions and calculate transmission"""
        epoch = self.current_time.get_current_epoch() if self.current_time else 0
        self.logger.info(f"Processing disease transmission for epoch {epoch}")
        
        # Get current health code policy impact
        current_impact = self.health_code_impact.get(str(epoch), {})
        restriction_level = current_impact.get("restriction_level", 0)
        transmission_reduction = current_impact.get("transmission_reduction", 0)
        mobility_reduction = current_impact.get("mobility_reduction", 0)
        
        # Get current health code enforcement rules
        current_enforcement = self.health_code_enforcement.get(str(epoch), {
            "green": True, "yellow": True, "red": True  # Default: all can go out
        })
        
        self.system_state["restriction_level"] = restriction_level
        
        # Reset daily tracking
        self.location_crowding.clear()
        self.transport_usage.clear()
        self.daily_contacts.clear()
        
        # Get health code distribution from blackboard
        health_code_distribution = self._get_from_blackboard("health_code_distribution", {})
        agent_health_codes = self._get_from_blackboard("agent_health_codes", {})
        
        # Process mobility decisions
        agents_going_out = []
        blocked_agents = []
        
        for agent_id, decisions in agent_decisions.items():
            mobility_decision = decisions.get(self.name, {})
            
            go_out = mobility_decision.get("go_out_decision", "no")
            destination = mobility_decision.get("destination_type", "none")
            transport = mobility_decision.get("transport_mode", "none")
            
            # Handle case where destination might be a list (fix TypeError)
            if isinstance(destination, list):
                destination = destination[0] if destination else "none"
            if isinstance(transport, list):
                transport = transport[0] if transport else "none"
            
            # Get agent's health code color
            health_code = agent_health_codes.get(agent_id, "green").lower()
            
            # Check if agent is allowed to go out based on health code and current policy
            allowed_to_go_out = current_enforcement.get(health_code, False)
            
            # Store original decision
            original_go_out = go_out == "yes"
            
            # Apply health code restrictions - this is the key change
            actual_go_out = original_go_out and allowed_to_go_out
            
            # Apply general mobility reduction due to policy (people voluntarily reducing mobility)
            if actual_go_out and random.random() < mobility_reduction:
                actual_go_out = False
                
            self.agent_mobility[agent_id] = {
                "go_out": actual_go_out,
                "destination": destination if actual_go_out else "none",
                "transport": transport if actual_go_out else "none",
                "blocked_by_policy": original_go_out and not actual_go_out
            }
            
            if actual_go_out:
                agents_going_out.append(agent_id)
                self.location_crowding[destination] += 1
                self.transport_usage[transport] += 1
            elif original_go_out and not actual_go_out:
                blocked_agents.append(agent_id)
        
        # Log policy impact
        if blocked_agents:
            self.logger.info(f"Health code policy blocked {len(blocked_agents)} agents from going out")
        
        # Calculate contacts and transmission
        self._calculate_contacts(agents_going_out)
        new_infection_count = self._simulate_transmission(epoch, transmission_reduction)
        
        # Update metrics
        self.new_infections[epoch] = new_infection_count
        self.cumulative_infections += new_infection_count
        
        # Calculate effective R0
        active_infections = self._get_from_blackboard("active_infections", 10)
        effective_r0 = new_infection_count / max(active_infections, 1)
        
        # Post to blackboard
        self._post_to_blackboard("new_infections", new_infection_count)
        self._post_to_blackboard("new_infection_list", [])  # Will be populated by _simulate_transmission
        self._post_to_blackboard("mobility_rate", len(agents_going_out) / len(agent_decisions))
        self._post_to_blackboard("blocked_mobility_rate", len(blocked_agents) / len(agent_decisions))
        self._post_to_blackboard("effective_r0", effective_r0)
        
        # Log statistics
        self.logger.info(f"Epoch {epoch}: {len(agents_going_out)}/{len(agent_decisions)} agents went out (rate: {len(agents_going_out)/len(agent_decisions):.3f})")
        self.logger.info(f"New infections: {new_infection_count}, Cumulative: {self.cumulative_infections}")
        self.logger.info(f"Location crowding: {dict(self.location_crowding)}")
        self.logger.info(f"Transport usage: {dict(self.transport_usage)}")
        
    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Return transmission risk information for a specific agent"""
        epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Get current policy restrictions
        current_impact = self.health_code_impact.get(str(epoch), {})
        restriction_level = current_impact.get("restriction_level", 0)
        
        # Get current health code enforcement
        current_enforcement = self.health_code_enforcement.get(str(epoch), {})
        
        # Get agent's health code color from blackboard
        agent_health_codes = self._get_from_blackboard("agent_health_codes", {})
        agent_health_code = agent_health_codes.get(agent_id, "green").lower()
        
        # Check if agent is allowed to go out based on health code
        allowed_to_go_out = current_enforcement.get(agent_health_code, True)
        
        # Calculate current infection rate in population
        active_infections = self._get_from_blackboard("active_infections", 0)
        total_agents = len(self.agent_attributes)
        current_infection_rate = active_infections / max(total_agents, 1)
        
        # Calculate location-specific risks
        location_risks = {}
        for location, multiplier_key in [
            ("work", "work_multiplier"),
            ("shopping", "shopping_multiplier"),
            ("social", "social_gathering_multiplier")
        ]:
            base_risk = self.transmission_params.get(multiplier_key, 1.0) * self.base_infection_rate
            adjusted_risk = base_risk * (1 - current_impact.get("transmission_reduction", 0))
            # Scale by actual infection rate in population
            final_risk = adjusted_risk * (1 + current_infection_rate * 10)  # Risk increases with more infections
            location_risks[location] = min(final_risk, 0.8)  # Cap at 80%
        
        # Public transport risk
        transport_risk = self.base_infection_rate * self.transmission_params.get("public_transport_multiplier", 3.0)
        transport_risk = transport_risk * (1 - current_impact.get("transmission_reduction", 0))
        transport_risk = transport_risk * (1 + current_infection_rate * 10)
        
        # Generate risk description
        if current_infection_rate < 0.01:
            risk_description = "Very low community transmission"
        elif current_infection_rate < 0.05:
            risk_description = "Low community transmission"
        elif current_infection_rate < 0.15:
            risk_description = "Moderate community transmission"
        else:
            risk_description = "High community transmission"
        
        return {
            "infection_rate": current_infection_rate,
            "risk_description": risk_description,
            "health_code_restrictions": {
                "level": restriction_level,
                "description": self._get_restriction_description(restriction_level),
                "allowed_to_go_out": allowed_to_go_out,
                "your_health_code": agent_health_code
            },
            "public_transport_risk": {
                "risk_level": min(transport_risk, 0.8),
                "crowding": self.transport_usage.get("public_transport", 0)
            },
            "location_risks": location_risks,
            "recent_outbreak_areas": self._get_outbreak_areas(),
            "daily_life_impact": self._get_daily_life_impact(restriction_level)
        }
    
    def _calculate_contacts(self, agents_going_out: List[str]):
        """Calculate contacts between agents based on their destinations"""
        # Group agents by destination
        destination_groups = defaultdict(list)
        transport_groups = defaultdict(list)
        
        for agent_id in agents_going_out:
            mobility = self.agent_mobility[agent_id]
            if mobility["destination"] != "none":
                destination_groups[mobility["destination"]].append(agent_id)
            if mobility["transport"] != "none":
                transport_groups[mobility["transport"]].append(agent_id)
        
        # Generate contacts within same locations
        for location, agents in destination_groups.items():
            if len(agents) > 1:
                # Each agent contacts a random subset of others at the location
                for agent in agents:
                    # Higher contact rates for certain locations
                    if location == "social":
                        num_contacts = min(random.randint(3, 8), len(agents) - 1)
                    elif location == "shopping":
                        num_contacts = min(random.randint(2, 6), len(agents) - 1)
                    elif location == "work":
                        num_contacts = min(random.randint(2, 5), len(agents) - 1)
                    else:
                        num_contacts = min(random.randint(1, 3), len(agents) - 1)
                    
                    if num_contacts > 0:
                        contacts = random.sample([a for a in agents if a != agent], num_contacts)
                        self.daily_contacts[agent].extend(contacts)
        
        # Additional contacts in public transport - higher contact rates
        public_transport_agents = transport_groups.get("public_transport", [])
        if len(public_transport_agents) > 1:
            for agent in public_transport_agents:
                num_contacts = min(random.randint(4, 12), len(public_transport_agents) - 1)
                if num_contacts > 0:
                    contacts = random.sample([a for a in public_transport_agents if a != agent], num_contacts)
                    self.daily_contacts[agent].extend(contacts)
        
        # Random encounters for agents going out (community transmission)
        if len(agents_going_out) > 2:
            for agent in agents_going_out:
                # Small chance of random encounters outside of main destinations
                if random.random() < 0.3:  # 30% chance
                    potential_encounters = [a for a in agents_going_out if a != agent]
                    if potential_encounters:
                        num_random_contacts = min(random.randint(1, 2), len(potential_encounters))
                        random_contacts = random.sample(potential_encounters, num_random_contacts)
                        self.daily_contacts[agent].extend(random_contacts)
    
    def _simulate_transmission(self, epoch: int, transmission_reduction: float) -> int:
        """Simulate disease transmission based on contacts"""
        # Get infected agents from blackboard
        infected_agents = self._get_from_blackboard("infected_agents", set())
        
        new_infections = set()
        
        # If no infected agents, introduce some random infections based on external sources
        if len(infected_agents) == 0 and epoch <= 2:
            # Simulate external introduction of infections (imported cases, etc.)
            num_external_infections = random.randint(1, 3)  # 1-3 external infections
            all_agents = list(self.agent_mobility.keys())
            if all_agents:
                external_infected = random.sample(all_agents, min(num_external_infections, len(all_agents)))
                new_infections.update(external_infected)
                self.logger.info(f"Introduced {len(external_infected)} external infections")
        
        # Simulate contact-based transmission
        for agent_id, contacts in self.daily_contacts.items():
            if agent_id in infected_agents:
                # This agent is infected and can transmit to contacts
                for contact_id in contacts:
                    if contact_id not in infected_agents and contact_id not in new_infections:
                        # Base transmission probability
                        base_prob = self.base_infection_rate
                        
                        # Reduce by policy effect
                        adjusted_prob = base_prob * (1 - transmission_reduction)
                        
                        # Actual transmission check
                        if random.random() < adjusted_prob:
                            new_infections.add(contact_id)
        
        # Post the new infections to blackboard for CitizenHealthSystem to use
        self._post_to_blackboard("new_infection_list", list(new_infections))
        
        return len(new_infections)
    
    def _get_restriction_description(self, level: float) -> str:
        """Get description of current restriction level"""
        if level < 0.1:
            return "No restrictions, normal movement allowed"
        elif level < 0.3:
            return "Light restrictions, avoid crowded places"
        elif level < 0.6:
            return "Moderate restrictions, health code required in public venues"
        elif level < 0.8:
            return "Strict restrictions, essential movement only"
        else:
            return "Very strict restrictions, stay home unless necessary"
    
    def _get_outbreak_areas(self) -> List[str]:
        """Get list of outbreak areas"""
        # In a real system, this would be dynamically updated based on infection clusters
        return ["Central District", "University Area", "Industrial Zone"]
    
    def _get_daily_life_impact(self, restriction_level: float) -> str:
        """Get description of impact on daily life"""
        if restriction_level < 0.1:
            return "Normal daily life, no significant impact"
        elif restriction_level < 0.3:
            return "Minor inconvenience, occasional checks"
        elif restriction_level < 0.6:
            return "Moderate impact, regular health code checks"
        elif restriction_level < 0.8:
            return "Significant impact, limited access to public places"
        else:
            return "Severe impact, most venues closed or restricted"
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the disease transmission system"""
        self.logger.info("=== Disease Transmission System Evaluation ===")
        
        # Calculate mobility metrics
        mobility_by_epoch = {}
        for epoch in range(self.current_time.get_current_epoch() + 1):
            # Get the decisions from that epoch
            epoch_decisions = self._get_from_blackboard(f"epoch_{epoch}_decisions", {})
            if not epoch_decisions:
                continue
                
            # Count agents who went out
            agents_out = 0
            for agent_id, agent_mobility in self.agent_mobility.items():
                if agent_mobility.get("go_out", False):
                    agents_out += 1
            
            mobility_rate = agents_out / len(epoch_decisions) if epoch_decisions else 0
            mobility_by_epoch[epoch] = mobility_rate
        
        # Calculate transmission metrics
        total_infections = sum(self.new_infections.values())
        peak_infections = max(self.new_infections.values()) if self.new_infections else 0
        peak_epoch = max(self.new_infections.items(), key=lambda x: x[1])[0] if self.new_infections else 0
        
        # Calculate average R0
        r0_values = []
        for epoch in range(self.current_time.get_current_epoch() + 1):
            active_infections = self._get_from_blackboard(f"epoch_{epoch}_active_infections", 0)
            new_infections = self.new_infections.get(epoch, 0)
            if active_infections > 0:
                r0 = new_infections / active_infections
                r0_values.append(r0)
        
        avg_r0 = sum(r0_values) / len(r0_values) if r0_values else 0
        
        # Calculate policy impact
        policy_impact = self._analyze_restriction_impact()
        
        # Prepare infection timeline
        infection_timeline = {}
        for epoch in range(self.current_time.get_current_epoch() + 1):
            infection_timeline[epoch] = self.new_infections.get(epoch, 0)
        
        # Log key metrics
        self.logger.info(f"Total infections from transmission system: {total_infections}")
        self.logger.info(f"Peak infections: {peak_infections} at epoch {peak_epoch}")
        self.logger.info(f"Average mobility rate: {sum(mobility_by_epoch.values()) / len(mobility_by_epoch) if mobility_by_epoch else 0}")
        self.logger.info(f"Average R0: {avg_r0}")
        
        # Log infection timeline
        self.logger.info("Infection timeline:")
        for epoch, count in infection_timeline.items():
            self.logger.info(f"  Epoch {epoch}: {count} new infections")
        
        # Log mobility rates
        self.logger.info("Mobility rates by epoch:")
        for epoch, rate in mobility_by_epoch.items():
            self.logger.info(f"  Epoch {epoch}: {rate}")
        
        return {
            "total_infections": total_infections,
            "peak_infections": peak_infections,
            "peak_epoch": peak_epoch,
            "average_mobility_rate": sum(mobility_by_epoch.values()) / len(mobility_by_epoch) if mobility_by_epoch else 0,
            "mobility_by_epoch": mobility_by_epoch,
            "average_r0": avg_r0,
            "infection_timeline": infection_timeline,
            "policy_impact": policy_impact
        }
    
    def _analyze_restriction_impact(self) -> Dict[str, Any]:
        """Analyze the impact of restrictions on mobility and transmission"""
        impact_by_epoch = {}
        
        for epoch in range(self.current_time.get_current_epoch() + 1):
            # Get policy parameters for this epoch
            policy = self.health_code_impact.get(str(epoch), {})
            restriction_level = policy.get("restriction_level", 0)
            transmission_reduction = policy.get("transmission_reduction", 0)
            mobility_reduction = policy.get("mobility_reduction", 0)
            
            # Get mobility rate for this epoch
            mobility_rate = self._get_from_blackboard(f"epoch_{epoch}_mobility_rate", 0)
            blocked_rate = self._get_from_blackboard(f"epoch_{epoch}_blocked_mobility_rate", 0)
            
            # Get infection data for this epoch
            new_infections = self.new_infections.get(epoch, 0)
            
            impact_by_epoch[epoch] = {
                "restriction_level": restriction_level,
                "transmission_reduction": transmission_reduction,
                "mobility_reduction": mobility_reduction,
                "actual_mobility_rate": mobility_rate,
                "blocked_mobility_rate": blocked_rate,
                "new_infections": new_infections
            }
        
        return impact_by_epoch 