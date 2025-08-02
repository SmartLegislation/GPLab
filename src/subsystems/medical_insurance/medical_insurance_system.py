import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class MedicalInsuranceSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)
        
        # Insurance policy parameters
        self.insurance_policies = config.get("insurance_policies", {})
        
        # Agent medical records
        self.agent_medical_expenses = defaultdict(float)  # agent_id -> total annual expenses
        self.agent_reimbursements = defaultdict(float)   # agent_id -> total reimbursements
        self.agent_visit_history = defaultdict(list)     # agent_id -> list of visits
        
        # System statistics
        self.monthly_visits = {"community_clinic": 0, "district_hospital": 0, "city_hospital": 0}
        self.monthly_expenses = 0
        self.monthly_reimbursements = 0
        
        # Historical data
        self.visit_history = []
        self.expense_history = []
        self.reimbursement_history = []
        
        self.logger.info("MedicalInsuranceSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent health profiles"""
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            age = agent_data.get("basic_info", {}).get("age", 30)
            
            # Initialize health risk based on age
            if age < 18:
                health_risk = 0.1
            elif age < 40:
                health_risk = 0.2
            elif age < 60:
                health_risk = 0.4
            else:
                health_risk = 0.6
            
            # Store in system state for reference
            self.system_state[f"health_risk_{agent_id}"] = health_risk
        
        self.logger.info(f"Initialized health profiles for {len(all_agent_data)} agents")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide insurance coverage information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policy = self.insurance_policies.get(str(current_epoch), 
                                                    self.insurance_policies.get("0", {}))
        
        # Calculate agent's current year expenses
        current_year_expenses = self.agent_medical_expenses.get(agent_id, 0)
        deductible_remaining = max(0, current_policy.get("annual_deductible", 500) - current_year_expenses)
        
        # Estimate costs for different hospital tiers
        base_costs = {
            "community_clinic": random.uniform(100, 300),
            "district_hospital": random.uniform(300, 800),
            "city_hospital": random.uniform(800, 2000)
        }
        
        # Calculate out-of-pocket costs after reimbursement
        out_of_pocket = {}
        for tier, base_cost in base_costs.items():
            reimbursement_rate = current_policy.get(f"{tier}_reimbursement", 0.5)
            if deductible_remaining > 0:
                # Need to pay deductible first
                if base_cost <= deductible_remaining:
                    out_of_pocket[tier] = base_cost
                else:
                    out_of_pocket[tier] = deductible_remaining + (base_cost - deductible_remaining) * (1 - reimbursement_rate)
            else:
                out_of_pocket[tier] = base_cost * (1 - reimbursement_rate)
        
        # Check if policy changed
        policy_changed = False
        if current_epoch > 0:
            prev_policy = self.insurance_policies.get(str(current_epoch - 1), {})
            if prev_policy != current_policy:
                policy_changed = True
        
        return {
            "insurance_coverage": {
                "community_clinic_reimbursement_rate": current_policy.get("community_clinic_reimbursement", 0.8),
                "district_hospital_reimbursement_rate": current_policy.get("district_hospital_reimbursement", 0.7),
                "city_hospital_reimbursement_rate": current_policy.get("city_hospital_reimbursement", 0.6),
                "annual_deductible": current_policy.get("annual_deductible", 500),
                "your_deductible_remaining": deductible_remaining
            },
            "reimbursement_rates": current_policy,
            "hospital_info": {
                "estimated_costs": base_costs,
                "estimated_out_of_pocket": out_of_pocket
            },
            "policy_changes": {
                "policy_updated_this_month": policy_changed,
                "message": "New policy encourages community clinic visits with higher reimbursement rates" if policy_changed and current_epoch >= 2 else ""
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent medical care decisions"""
        # Reset monthly statistics
        self.monthly_visits = {"community_clinic": 0, "district_hospital": 0, "city_hospital": 0}
        self.monthly_expenses = 0
        self.monthly_reimbursements = 0
        
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policy = self.insurance_policies.get(str(current_epoch), 
                                                    self.insurance_policies.get("0", {}))
        
        for agent_id, decisions in agent_decisions.items():
            if "MedicalInsuranceSystem" not in decisions:
                continue
            
            decision = decisions["MedicalInsuranceSystem"]
            seek_care = decision.get("seek_medical_care", "no")
            hospital_tier = decision.get("hospital_tier_choice", "none")
            treatment_intensity = decision.get("treatment_intensity", "basic")
            
            if seek_care == "yes" and hospital_tier != "none":
                # Calculate treatment cost based on tier and intensity
                base_costs = {
                    "community_clinic": {"basic": 150, "standard": 250, "comprehensive": 400},
                    "district_hospital": {"basic": 400, "standard": 600, "comprehensive": 1000},
                    "city_hospital": {"basic": 1000, "standard": 1500, "comprehensive": 2500}
                }
                
                treatment_cost = base_costs.get(hospital_tier, {}).get(treatment_intensity, 200)
                
                # Add some randomness
                treatment_cost *= random.uniform(0.8, 1.2)
                
                # Calculate reimbursement
                reimbursement_rate = current_policy.get(f"{hospital_tier}_reimbursement", 0.5)
                deductible = current_policy.get("annual_deductible", 500)
                
                # Check if deductible has been met
                current_year_expenses = self.agent_medical_expenses.get(agent_id, 0)
                
                if current_year_expenses < deductible:
                    # Still need to meet deductible
                    deductible_payment = min(treatment_cost, deductible - current_year_expenses)
                    reimbursable_amount = max(0, treatment_cost - deductible_payment)
                    reimbursement = reimbursable_amount * reimbursement_rate
                else:
                    # Deductible already met
                    reimbursement = treatment_cost * reimbursement_rate
                
                out_of_pocket = treatment_cost - reimbursement
                
                # Update records
                self.agent_medical_expenses[agent_id] += treatment_cost
                self.agent_reimbursements[agent_id] += reimbursement
                self.agent_visit_history[agent_id].append({
                    "epoch": current_epoch,
                    "hospital_tier": hospital_tier,
                    "treatment_intensity": treatment_intensity,
                    "cost": treatment_cost,
                    "reimbursement": reimbursement,
                    "out_of_pocket": out_of_pocket
                })
                
                # Update monthly statistics
                self.monthly_visits[hospital_tier] += 1
                self.monthly_expenses += treatment_cost
                self.monthly_reimbursements += reimbursement
                
                self.logger.debug(f"Agent {agent_id} visited {hospital_tier} for {treatment_intensity} treatment. "
                                f"Cost: {treatment_cost:.2f}, Reimbursement: {reimbursement:.2f}")
        
        # Record historical data
        self.visit_history.append(dict(self.monthly_visits))
        self.expense_history.append(self.monthly_expenses)
        self.reimbursement_history.append(self.monthly_reimbursements)
        
        self.logger.info(f"Epoch {current_epoch}: Total visits={sum(self.monthly_visits.values())}, "
                        f"Expenses={self.monthly_expenses:.2f}, Reimbursements={self.monthly_reimbursements:.2f}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the impact of medical insurance reform"""
        # Calculate visit distribution changes
        total_visits_by_tier = {"community_clinic": 0, "district_hospital": 0, "city_hospital": 0}
        for visit_record in self.visit_history:
            for tier, count in visit_record.items():
                total_visits_by_tier[tier] += count
        
        # Compare early vs late distribution
        early_visits = self.visit_history[:2] if len(self.visit_history) > 2 else []
        late_visits = self.visit_history[-2:] if len(self.visit_history) > 2 else []
        
        early_community_ratio = 0
        late_community_ratio = 0
        
        if early_visits:
            early_total = sum(sum(v.values()) for v in early_visits)
            early_community = sum(v.get("community_clinic", 0) for v in early_visits)
            early_community_ratio = early_community / early_total if early_total > 0 else 0
        
        if late_visits:
            late_total = sum(sum(v.values()) for v in late_visits)
            late_community = sum(v.get("community_clinic", 0) for v in late_visits)
            late_community_ratio = late_community / late_total if late_total > 0 else 0
        
        # Calculate financial metrics
        total_expenses = sum(self.expense_history)
        total_reimbursements = sum(self.reimbursement_history)
        avg_reimbursement_rate = total_reimbursements / total_expenses if total_expenses > 0 else 0
        
        # Analyze individual burden
        individual_burdens = []
        for agent_id, expenses in self.agent_medical_expenses.items():
            reimbursements = self.agent_reimbursements.get(agent_id, 0)
            out_of_pocket = expenses - reimbursements
            individual_burdens.append(out_of_pocket)
        
        evaluation_results = {
            "visit_distribution": {
                "total_visits_by_tier": total_visits_by_tier,
                "community_clinic_ratio_change": float(late_community_ratio - early_community_ratio),
                "policy_goal_achieved": bool(late_community_ratio > early_community_ratio + 0.1)
            },
            "financial_impact": {
                "total_medical_expenses": float(total_expenses),
                "total_reimbursements": float(total_reimbursements),
                "average_reimbursement_rate": float(avg_reimbursement_rate),
                "government_burden_change": float(self.reimbursement_history[-1] - self.reimbursement_history[0]) if self.reimbursement_history else 0
            },
            "individual_impact": {
                "average_out_of_pocket": float(np.mean(individual_burdens)) if individual_burdens else 0,
                "median_out_of_pocket": float(np.median(individual_burdens)) if individual_burdens else 0,
                "max_out_of_pocket": float(max(individual_burdens)) if individual_burdens else 0
            },
            "time_series": {
                "visit_history": self.visit_history,
                "expense_history": [float(x) for x in self.expense_history],
                "reimbursement_history": [float(x) for x in self.reimbursement_history]
            },
            "policy_effectiveness": {
                "increased_community_clinic_usage": bool(late_community_ratio > early_community_ratio),
                "reduced_individual_burden": bool(float(np.mean(individual_burdens)) < 1000) if individual_burdens else False,
                "cost_control_achieved": bool(total_expenses < len(self.agent_medical_expenses) * 2000)
            }
        }
        
        self.logger.info(f"medical insurance system evaluation complete. "
                        f"evaluation_results {evaluation_results}")
        
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        return {
            "monthly_visits": self.monthly_visits,
            "monthly_expenses": self.monthly_expenses,
            "monthly_reimbursements": self.monthly_reimbursements,
            "total_agents_treated": len([a for a in self.agent_visit_history if self.agent_visit_history[a]]),
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 