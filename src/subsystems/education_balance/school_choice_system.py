import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class SchoolChoiceSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)
        
        # School quality parameters
        self.initial_school_qualities = config.get("initial_school_qualities", {
            "top_tier": 0.9,
            "good_quality": 0.7,
            "average": 0.5,
            "weak": 0.3
        })
        
        # Housing market correlation with school quality
        self.school_housing_correlation = config.get("school_housing_correlation", 0.7)
        
        # Agent residence tracking
        self.agent_residences = {}  # agent_id -> district_type
        self.residence_history = defaultdict(list)  # agent_id -> [(epoch, district_type), ...]
        
        # Housing market statistics
        self.district_housing_prices = {
            "top_tier": 1.5,      # Multiplier relative to average
            "good_quality": 1.2,
            "average": 1.0,
            "weak": 0.8
        }
        self.housing_price_history = []
        
        # Choice behavior tracking
        self.choice_motivations = defaultdict(int)  # motivation -> count
        self.school_satisfaction = {}  # agent_id -> satisfaction score
        self.district_population = defaultdict(int)  # district_type -> population
        
        # Historical data
        self.district_movement_history = []  # List of movement counts per epoch
        self.housing_affordability_index = []  # Track affordability over time
        
        self.logger.info("SchoolChoiceSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent residences and housing market"""
        district_types = list(self.initial_school_qualities.keys())
        
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            
            # Assign initial residence based on income level
            income_level = agent_data.get("economic_attributes", {}).get("income_level", "")
            
            # Determine residence district based on income
            if "high" in income_level.lower():
                district_type = random.choices(district_types, weights=[0.5, 0.3, 0.15, 0.05])[0]
            elif "moderate" in income_level.lower():
                district_type = random.choices(district_types, weights=[0.15, 0.35, 0.35, 0.15])[0]
            else:
                district_type = random.choices(district_types, weights=[0.05, 0.15, 0.3, 0.5])[0]
            
            self.agent_residences[agent_id] = district_type
            self.district_population[district_type] += 1
            
            # Initialize satisfaction with school and district
            school_quality = self.initial_school_qualities.get(district_type, 0.5)
            income_factor = 0.8 if "high" in income_level.lower() else 0.6 if "moderate" in income_level.lower() else 0.4
            
            # Satisfaction is based on school quality and affordability
            housing_cost = self.district_housing_prices.get(district_type, 1.0)
            affordability_factor = max(0.1, min(1.0, income_factor / housing_cost))
            
            self.school_satisfaction[agent_id] = (school_quality * 0.6) + (affordability_factor * 0.4)
            
            # Share income level in system state for other subsystems
            self.system_state[f"income_level_{agent_id}"] = income_level
            self.system_state[f"residence_district_{agent_id}"] = district_type
        
        # Record initial housing market state
        self.housing_price_history.append(dict(self.district_housing_prices))
        
        # Calculate initial affordability index (average price / average income)
        avg_price = np.mean(list(self.district_housing_prices.values()))
        self.housing_affordability_index.append(avg_price)
        
        # Share initial housing prices with other subsystems
        self._share_housing_market_state()
        
        self.logger.info(f"Initialized residences for {len(all_agent_data)} agents")

    def _update_housing_prices(self, education_policy: Dict[str, Any] = None):
        """Update housing prices based on school quality and demand"""
        # Get school qualities from shared state (assuming EducationResourceSystem updated it)
        school_qualities = {}
        for district_type in self.district_housing_prices.keys():
            quality_key = f"school_quality_{district_type}"
            school_qualities[district_type] = self.system_state.get(quality_key, self.initial_school_qualities.get(district_type, 0.5))
        
        # Update housing prices based on school quality and population demand
        total_population = sum(self.district_population.values())
        avg_population = total_population / len(self.district_population) if self.district_population else 1
        
        for district_type in self.district_housing_prices.keys():
            # Base price influenced by school quality
            quality_factor = school_qualities.get(district_type, 0.5) * self.school_housing_correlation
            
            # Demand pressure based on population
            population_ratio = self.district_population.get(district_type, 0) / avg_population if avg_population > 0 else 1
            demand_factor = min(1.5, max(0.8, population_ratio))
            
            # Calculate new price (with some inertia to prevent wild swings)
            current_price = self.district_housing_prices[district_type]
            target_price = 0.5 + (quality_factor * 1.0) + (demand_factor * 0.3)
            
            # Apply policy effects if applicable
            if education_policy and education_policy.get("resource_allocation") == "equalized":
                # Equalized education reduces housing price disparities
                avg_price = np.mean(list(self.district_housing_prices.values()))
                target_price = current_price + (avg_price - current_price) * 0.25
            
            # Apply gradual change (20% toward target)
            self.district_housing_prices[district_type] = current_price + (target_price - current_price) * 0.3
        
        # Record updated prices
        self.housing_price_history.append(dict(self.district_housing_prices))
        
        # Update affordability index
        avg_price = np.mean(list(self.district_housing_prices.values()))
        self.housing_affordability_index.append(avg_price)

    def _share_housing_market_state(self):
        """Share housing market state with other subsystems"""
        for district, price in self.district_housing_prices.items():
            self.system_state[f"housing_price_{district}"] = price
        
        for district, population in self.district_population.items():
            self.system_state[f"district_population_{district}"] = population
            
        # Share affordability index
        self.system_state["housing_affordability_index"] = self.housing_affordability_index[-1] if self.housing_affordability_index else 1.0
        
        # Share price gap
        prices = list(self.district_housing_prices.values())
        if prices:
            self.system_state["housing_price_gap"] = max(prices) - min(prices)

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide housing and school choice information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Get agent's current residence
        current_district = self.agent_residences.get(agent_id)
        
        # Get school quality from shared state or default
        school_qualities = {}
        quality_changes = {}
        
        for district_type in self.district_housing_prices.keys():
            quality_key = f"school_quality_{district_type}"
            current_quality = self.system_state.get(quality_key, self.initial_school_qualities.get(district_type, 0.5))
            school_qualities[district_type] = current_quality
            
            # Calculate quality change if we have history
            if current_epoch > 0:
                previous_quality = self.system_state.get(f"previous_{quality_key}", self.initial_school_qualities.get(district_type, 0.5))
                quality_changes[district_type] = current_quality - previous_quality
        
        # Get current housing prices
        current_prices = self.district_housing_prices
        
        # Calculate price trends
        price_trends = {}
        if len(self.housing_price_history) > 1:
            previous_prices = self.housing_price_history[-2]
            for district, price in current_prices.items():
                previous = previous_prices.get(district, price)
                price_trends[district] = (price - previous) / previous if previous > 0 else 0
        
        # Get agent's satisfaction
        satisfaction = self.school_satisfaction.get(agent_id, 0.5)
        
        # Calculate neighborhood competition for schools
        enrollment_competition = {}
        for district in self.district_population.keys():
            enrollment_key = f"enrollment_pressure_{district}"
            enrollment_competition[district] = self.system_state.get(enrollment_key, 1.0)
        
        # Calculate commute impact for different districts
        commute_impact = {
            current_district: 0.0  # No additional commute for current district
        }
        
        for district in self.district_housing_prices.keys():
            if district != current_district:
                # Simulate commute impact (higher for bigger quality gaps)
                quality_gap = abs(school_qualities.get(district, 0.5) - school_qualities.get(current_district, 0.5))
                commute_impact[district] = min(1.0, quality_gap * 1.0)
        
        return {
            "housing_market": {
                "current_district": current_district,
                "district_prices": current_prices,
                "price_trends": price_trends,
                "affordability_index": self.housing_affordability_index[-1] if self.housing_affordability_index else 1.0
            },
            "school_quality_changes": {
                "current_qualities": school_qualities,
                "recent_changes": quality_changes,
                "your_satisfaction": satisfaction
            },
            "enrollment_competition": {
                "district_competition": enrollment_competition,
                "population_distribution": {k: v / sum(self.district_population.values()) 
                                           for k, v in self.district_population.items()}
                                           if sum(self.district_population.values()) > 0 else {}
            },
            "housing_market_impact": {
                "commute_impact": commute_impact,
                "price_to_quality_ratio": {
                    district: price / school_qualities.get(district, 0.5) 
                    for district, price in current_prices.items()
                },
                "moving_costs": 0.1  # Relative cost of moving (could be more dynamic)
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent housing and school choice decisions"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Get current education policy from shared state
        education_policy = {}
        policy_key = f"education_policy_{current_epoch}"
        if policy_key in self.system_state:
            education_policy = self.system_state[policy_key]
        
        # Track district movements
        district_movements = defaultdict(int)
        
        # Process agent decisions
        for agent_id, decisions in agent_decisions.items():
            if "EducationResourceSystem" not in decisions:
                continue
            
            decision = decisions["EducationResourceSystem"]
            action = decision.get("school_choice_action", "stay_current")
            preferred_type = decision.get("preferred_school_type", "average")
            
            current_district = self.agent_residences.get(agent_id)
            
            # Process residence changes
            if action == "move_residence" and current_district != preferred_type:
                # Record movement
                district_movements[f"{current_district}_to_{preferred_type}"] += 1
                
                # Update district populations
                self.district_population[current_district] -= 1
                self.district_population[preferred_type] += 1
                
                # Update agent residence
                self.agent_residences[agent_id] = preferred_type
                self.residence_history[agent_id].append((current_epoch, preferred_type))
                
                # Record motivation
                motivation = decision.get("choice_motivation", "school_quality")
                self.choice_motivations[motivation] += 1
                
                # Update satisfaction based on new district
                school_quality = self.system_state.get(f"school_quality_{preferred_type}", 
                                                     self.initial_school_qualities.get(preferred_type, 0.5))
                housing_cost = self.district_housing_prices.get(preferred_type, 1.0)
                
                # Simple satisfaction model based on quality and affordability
                income_level = self.system_state.get(f"income_level_{agent_id}", "moderate")
                income_factor = 0.8 if "high" in income_level.lower() else 0.6 if "moderate" in income_level.lower() else 0.4
                affordability_factor = max(0.1, min(1.0, income_factor / housing_cost))
                
                self.school_satisfaction[agent_id] = (school_quality * 0.6) + (affordability_factor * 0.4)
                
                # Update residence in system state for other subsystems
                self.system_state[f"residence_district_{agent_id}"] = preferred_type
        
        # Update housing prices based on new district populations
        self._update_housing_prices(education_policy)
        
        # Record district movements for this epoch
        self.district_movement_history.append(dict(district_movements))
        
        # Share updated state with other subsystems
        self._share_housing_market_state()
        
        # Log summary
        total_movements = sum(district_movements.values())
        self.logger.info(f"Epoch {current_epoch}: Housing movements={total_movements}, "
                        f"Avg price index={self.housing_affordability_index[-1]:.2f}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate housing market and school choice dynamics"""
        # Calculate housing price divergence
        initial_prices = self.housing_price_history[0] if self.housing_price_history else {}
        final_prices = self.housing_price_history[-1] if self.housing_price_history else {}
        
        price_changes = {}
        for district in final_prices.keys():
            initial = float(initial_prices.get(district, 1.0))
            final = float(final_prices.get(district, 1.0))
            price_changes[district] = float((final - initial) / initial if initial > 0 else 0)
        
        # Calculate price gap
        if final_prices:
            initial_price_gap = float(max(initial_prices.values()) - min(initial_prices.values()) if initial_prices else 0)
            final_price_gap = float(max(final_prices.values()) - min(final_prices.values()))
            price_gap_change = float(final_price_gap - initial_price_gap)
        else:
            initial_price_gap = final_price_gap = price_gap_change = 0.0
        
        # Calculate residential segregation index (simplified)
        population_shares = {}
        total_population = sum(self.district_population.values())
        
        for district, population in self.district_population.items():
            population_shares[district] = float(population / total_population if total_population > 0 else 0)
        
        # Calculate Gini coefficient for population distribution
        def calculate_gini(distribution):
            values = list(distribution.values())
            if not values or sum(values) == 0:
                return 0.0
            sorted_values = sorted(values)
            n = len(sorted_values)
            cumsum = np.cumsum(sorted_values)
            return float((2 * np.sum((np.arange(1, n+1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n)
        
        segregation_index = float(calculate_gini(self.district_population))
        
        # Analyze choice motivations
        top_motivations = sorted([(str(motivation), int(count)) for motivation, count in self.choice_motivations.items()], 
                               key=lambda x: x[1], reverse=True)
        
        # Calculate total movements across all epochs
        total_movements = int(sum(sum(epoch.values()) for epoch in self.district_movement_history))
        
        # Calculate affordability trend
        initial_affordability = float(self.housing_affordability_index[0] if self.housing_affordability_index else 1.0)
        final_affordability = float(self.housing_affordability_index[-1] if self.housing_affordability_index else 1.0)
        affordability_change = float(final_affordability - initial_affordability)
        
        # Get education quality gap from system state
        quality_gap = float(self.system_state.get("quality_gap", 0))
        
        evaluation_results = {
            "housing_market_metrics": {
                "initial_price_gap": float(initial_price_gap),
                "final_price_gap": float(final_price_gap), 
                "price_gap_change": float(price_gap_change),
                "district_price_changes": {str(k): float(v) for k,v in price_changes.items()},
                "affordability_index_change": float(affordability_change)
            },
            "residential_patterns": {
                "district_populations": {str(k): int(v) for k,v in self.district_population.items()},
                "population_distribution": {str(k): float(v) for k,v in population_shares.items()},
                "segregation_index": float(segregation_index),
                "total_district_movements": int(total_movements)
            },
            "choice_behavior": {
                "top_choice_motivations": [{"motivation": str(m), "count": int(c)} for m, c in top_motivations[:3]] if top_motivations else [],
                "average_satisfaction": float(np.mean(list(self.school_satisfaction.values())) if self.school_satisfaction else 0.5),
                "satisfaction_distribution": {
                    "high": int(sum(1 for s in self.school_satisfaction.values() if s >= 0.7)),
                    "medium": int(sum(1 for s in self.school_satisfaction.values() if 0.4 <= s < 0.7)),
                    "low": int(sum(1 for s in self.school_satisfaction.values() if s < 0.4))
                }
            },
            "policy_impact": {
                "reduced_housing_inequality": bool(price_gap_change < 0),
                "improved_affordability": bool(affordability_change < 0),
                "reduced_residential_segregation": bool(segregation_index < 0.3),
                "education_quality_gap": float(quality_gap),
                "quality_housing_correlation": float(self._calculate_quality_price_correlation()),
                "overall_success": bool(price_gap_change < 0 and affordability_change < 0)
            },
            "time_series": {
                "housing_price_history": [[str(k), float(v)] for hist in self.housing_price_history for k,v in hist.items()],
                "affordability_index_history": [float(x) for x in self.housing_affordability_index],
                "movement_history": [{str(k): int(v) for k,v in hist.items()} for hist in self.district_movement_history]
            }
        }
        
        self.logger.info(f"evaluation_results={evaluation_results}")
        
        return evaluation_results
    
    def _calculate_quality_price_correlation(self) -> float:
        """Calculate correlation between school quality and housing prices"""
        qualities = []
        prices = []
        
        for district in self.district_housing_prices.keys():
            quality = self.system_state.get(f"school_quality_{district}", self.initial_school_qualities.get(district, 0.5))
            price = self.district_housing_prices.get(district, 1.0)
            
            qualities.append(quality)
            prices.append(price)
        
        if len(qualities) <= 1:
            return 0.0
            
        # Calculate correlation coefficient
        try:
            correlation = np.corrcoef(qualities, prices)[0, 1]
            return correlation
        except:
            return 0.0

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        # Calculate average satisfaction
        avg_satisfaction = np.mean(list(self.school_satisfaction.values())) if self.school_satisfaction else 0.5
        
        # Calculate price gap
        current_prices = self.district_housing_prices
        price_gap = max(current_prices.values()) - min(current_prices.values()) if current_prices else 0
        
        return {
            "district_housing_prices": dict(self.district_housing_prices),
            "district_populations": dict(self.district_population),
            "housing_price_gap": price_gap,
            "average_satisfaction": avg_satisfaction,
            "total_district_movements": sum(sum(epoch.values()) for epoch in self.district_movement_history),
            "affordability_index": self.housing_affordability_index[-1] if self.housing_affordability_index else 1.0,
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 