import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class HealthcareUtilizationSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)
        
        # Hospital capacity configuration
        self.hospital_capacities = config.get("hospital_capacities", {
            "community_clinic": 500,
            "district_hospital": 300,
            "city_hospital": 200
        })
        
        # Current utilization tracking
        self.current_utilization = {
            "community_clinic": 0,
            "district_hospital": 0,
            "city_hospital": 0
        }
        
        # Service quality metrics (affected by crowding)
        self.service_quality = {
            "community_clinic": 0.8,
            "district_hospital": 0.85,
            "city_hospital": 0.9
        }
        
        # Historical tracking
        self.utilization_history = []
        self.wait_time_history = []
        self.quality_history = []
        
        self.logger.info("HealthcareUtilizationSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize system with agent data"""
        self.total_agents = len(all_agent_data)
        self.logger.info(f"Initialized with {self.total_agents} agents in the system")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide hospital crowding and service quality information"""
        # Calculate crowding levels
        crowding_levels = {}
        wait_times = {}
        
        for tier, capacity in self.hospital_capacities.items():
            utilization_rate = self.current_utilization[tier] / capacity
            
            if utilization_rate < 0.5:
                crowding_levels[tier] = "low"
                wait_times[tier] = "15-30 minutes"
            elif utilization_rate < 0.8:
                crowding_levels[tier] = "moderate"
                wait_times[tier] = "30-60 minutes"
            elif utilization_rate < 1.0:
                crowding_levels[tier] = "high"
                wait_times[tier] = "1-2 hours"
            else:
                crowding_levels[tier] = "overcrowded"
                wait_times[tier] = "2+ hours"
        
        return {
            "hospital_crowding": crowding_levels,
            "wait_times": wait_times,
            "service_quality": {
                tier: f"{quality * 100:.0f}% satisfaction rate"
                for tier, quality in self.service_quality.items()
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Update utilization based on agent decisions"""
        # Reset current utilization
        self.current_utilization = {
            "community_clinic": 0,
            "district_hospital": 0,
            "city_hospital": 0
        }
        
        # Count visits from medical insurance system decisions
        for agent_id, decisions in agent_decisions.items():
            if "MedicalInsuranceSystem" in decisions:
                decision = decisions["MedicalInsuranceSystem"]
                if decision.get("seek_medical_care") == "yes":
                    hospital_tier = decision.get("hospital_tier_choice", "none")
                    if hospital_tier in self.current_utilization:
                        self.current_utilization[hospital_tier] += 1
        
        # Update service quality based on crowding
        for tier, capacity in self.hospital_capacities.items():
            utilization_rate = self.current_utilization[tier] / capacity
            
            # Base quality degradation due to crowding
            if utilization_rate < 0.5:
                quality_factor = 1.0
            elif utilization_rate < 0.8:
                quality_factor = 0.95
            elif utilization_rate < 1.0:
                quality_factor = 0.85
            else:
                quality_factor = 0.7
            
            # Update service quality
            base_quality = {"community_clinic": 0.8, "district_hospital": 0.85, "city_hospital": 0.9}
            self.service_quality[tier] = base_quality[tier] * quality_factor
        
        # Calculate average wait times
        avg_wait_times = {}
        for tier, utilization in self.current_utilization.items():
            capacity = self.hospital_capacities[tier]
            utilization_rate = utilization / capacity
            
            # Simple wait time model (in minutes)
            base_wait = {"community_clinic": 20, "district_hospital": 30, "city_hospital": 40}
            avg_wait_times[tier] = base_wait[tier] * (1 + utilization_rate ** 2)
        
        # Record history
        self.utilization_history.append(dict(self.current_utilization))
        self.wait_time_history.append(avg_wait_times)
        self.quality_history.append(dict(self.service_quality))
        
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        self.logger.info(f"Epoch {current_epoch}: Utilization - Community: {self.current_utilization['community_clinic']}, "
                        f"District: {self.current_utilization['district_hospital']}, "
                        f"City: {self.current_utilization['city_hospital']}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate healthcare system performance"""
        # Calculate average utilization rates
        avg_utilization = {tier: 0 for tier in self.hospital_capacities}
        for record in self.utilization_history:
            for tier, count in record.items():
                avg_utilization[tier] += count
        
        for tier in avg_utilization:
            avg_utilization[tier] /= len(self.utilization_history) if self.utilization_history else 1
            avg_utilization[tier] /= self.hospital_capacities[tier]  # Convert to rate
        
        # Analyze wait time trends
        avg_wait_times = {tier: [] for tier in self.hospital_capacities}
        for record in self.wait_time_history:
            for tier, wait in record.items():
                avg_wait_times[tier].append(wait)
        
        # Calculate overcrowding incidents
        overcrowding_incidents = {tier: 0 for tier in self.hospital_capacities}
        for record in self.utilization_history:
            for tier, count in record.items():
                if count > self.hospital_capacities[tier]:
                    overcrowding_incidents[tier] += 1
        
        # Analyze quality trends
        quality_trends = {tier: [] for tier in self.hospital_capacities}
        for record in self.quality_history:
            for tier, quality in record.items():
                quality_trends[tier].append(quality)
        
        # Convert numpy types to native Python types for JSON serialization
        evaluation_results = {
            "utilization_metrics": {
                "average_utilization_rates": {
                    tier: float(rate) for tier, rate in avg_utilization.items()
                },
                "peak_utilization": {
                    tier: float(max(record[tier] for record in self.utilization_history) / self.hospital_capacities[tier])
                    for tier in self.hospital_capacities
                },
                "overcrowding_incidents": {
                    tier: int(count) for tier, count in overcrowding_incidents.items()
                }
            },
            "wait_time_analysis": {
                "average_wait_times": {
                    tier: float(np.mean(times)) if times else 0.0
                    for tier, times in avg_wait_times.items()
                },
                "max_wait_times": {
                    tier: float(max(times)) if times else 0.0
                    for tier, times in avg_wait_times.items()
                }
            },
            "service_quality": {
                "final_quality_scores": {
                    tier: float(score) for tier, score in self.service_quality.items()
                },
                "quality_trends": {
                    tier: {
                        "start": float(quality_trends[tier][0]) if quality_trends[tier] else 0.0,
                        "end": float(quality_trends[tier][-1]) if quality_trends[tier] else 0.0,
                        "average": float(np.mean(quality_trends[tier])) if quality_trends[tier] else 0.0
                    }
                    for tier in self.hospital_capacities
                }
            },
            "system_efficiency": {
                "balanced_utilization": bool(float(np.std(list(avg_utilization.values()))) < 0.2),
                "community_clinic_improvement": bool(
                    self.utilization_history[-1]["community_clinic"] > self.utilization_history[0]["community_clinic"]
                    if self.utilization_history else False
                )
            },
            "time_series": {
                "utilization_history": [
                    {tier: int(count) for tier, count in record.items()}
                    for record in self.utilization_history
                ],
                "wait_time_history": [
                    {tier: float(wait) for tier, wait in record.items()}
                    for record in self.wait_time_history
                ]
            }
        }
        
        self.logger.info(f"Healthcare system evaluation complete. "
                        f"evaluation_results {evaluation_results}")
        
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        return {
            "current_utilization": self.current_utilization,
            "service_quality": self.service_quality,
            "utilization_rates": {
                tier: self.current_utilization[tier] / self.hospital_capacities[tier]
                for tier in self.hospital_capacities
            },
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 