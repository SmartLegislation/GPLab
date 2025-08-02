import random
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class EducationResourceSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)
        
        # Education policy configurations
        self.education_policies = config.get("education_policies", {})
        
        # School quality tracking
        self.school_qualities = {
            "top_tier": 0.9,
            "good_quality": 0.7,
            "average": 0.5,
            "weak": 0.3
        }
        
        # Agent school assignments
        self.agent_schools = {}  # agent_id -> school_type
        self.agent_school_history = defaultdict(list)
        
        # School enrollment statistics
        self.school_enrollments = defaultdict(int)
        self.transfer_requests = defaultdict(int)
        self.transfer_success = defaultdict(int)
        
        # Student achievement tracking
        self.student_achievement = {}  # agent_id -> achievement score
        self.achievement_by_school_type = defaultdict(list)  # school_type -> list of achievement scores
        self.achievement_history = []  # track average achievement over time
        
        # Socioeconomic impact tracking
        self.socioeconomic_mobility = {}  # agent_id -> mobility score
        self.mobility_by_school_type = defaultdict(list)
        
        # Facility improvement tracking
        self.facility_quality = {
            "top_tier": 0.9,
            "good_quality": 0.7,
            "average": 0.5,
            "weak": 0.3
        }
        
        # Historical tracking
        self.quality_gap_history = []
        self.enrollment_distribution_history = []
        self.facility_improvement_history = []
        
        self.logger.info("EducationResourceSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent school assignments based on residence"""
        school_types = list(self.school_qualities.keys())
        
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            
            # Check if agent has school-age children
            age = agent_data.get("basic_info", {}).get("age", 30)
            marital_status = agent_data.get("basic_info", {}).get("marital_status", "")
            
            has_children = False
            if "married" in marital_status.lower() and 25 <= age <= 50:
                has_children = random.random() < 0.6
            
            if has_children:
                # Assign school based on income level (initial inequality)
                income_level = agent_data.get("economic_attributes", {}).get("income_level", "")
                
                if "high" in income_level.lower():
                    school_type = random.choices(school_types, weights=[0.6, 0.3, 0.1, 0.0])[0]
                elif "moderate" in income_level.lower():
                    school_type = random.choices(school_types, weights=[0.2, 0.4, 0.3, 0.1])[0]
                else:
                    school_type = random.choices(school_types, weights=[0.1, 0.2, 0.4, 0.3])[0]
                
                self.agent_schools[agent_id] = school_type
                self.school_enrollments[school_type] += 1
                
                # Initialize student achievement based on school quality and random factors
                base_achievement = self.school_qualities[school_type] * 0.7
                random_factor = random.uniform(-0.1, 0.1)
                self.student_achievement[agent_id] = max(0.1, min(1.0, base_achievement + random_factor))
                self.achievement_by_school_type[school_type].append(self.student_achievement[agent_id])
                
                # Initialize socioeconomic mobility potential
                income_factor = 0.7 if "high" in income_level.lower() else 0.5 if "moderate" in income_level.lower() else 0.3
                school_factor = self.school_qualities[school_type]
                self.socioeconomic_mobility[agent_id] = (income_factor * 0.3) + (school_factor * 0.7)
                self.mobility_by_school_type[school_type].append(self.socioeconomic_mobility[agent_id])
            else:
                self.agent_schools[agent_id] = None
        
        # Record initial achievement averages
        avg_achievement = np.mean(list(self.student_achievement.values())) if self.student_achievement else 0.5
        self.achievement_history.append(avg_achievement)
        
        self.logger.info(f"Initialized {len([a for a in self.agent_schools.values() if a])} agents with school-age children")

    def _update_school_qualities(self, current_policy: Dict[str, Any]):
        """Update school qualities based on policy"""
        resource_allocation = current_policy.get("resource_allocation", "traditional")
        
        if resource_allocation == "traditional":
            # Maintain inequality
            pass
        elif resource_allocation == "balanced":
            # Start reducing gap
            gap_reduction = 0.05
            avg_quality = np.mean(list(self.school_qualities.values()))
            
            for school_type in self.school_qualities:
                current = self.school_qualities[school_type]
                self.school_qualities[school_type] = current + (avg_quality - current) * gap_reduction
                
        elif resource_allocation == "equalized":
            # Aggressive gap reduction
            gap_reduction = 0.15
            target_quality = 0.7  # Target quality for all schools
            
            for school_type in self.school_qualities:
                current = self.school_qualities[school_type]
                self.school_qualities[school_type] = current + (target_quality - current) * gap_reduction
        
        # Apply teacher rotation effect
        if current_policy.get("teacher_rotation", False):
            avg_quality_overall = np.mean(list(self.school_qualities.values()))
            for school_type in self.school_qualities:
                if self.school_qualities[school_type] < avg_quality_overall:
                    # Move 5% towards the average
                    self.school_qualities[school_type] += (avg_quality_overall - self.school_qualities[school_type]) * 0.05
        
        # Apply facility upgrade effect if fund is available
        facility_fund = current_policy.get("facility_upgrade_fund", 0)
        if facility_fund > 0:
            # Prioritize schools with lowest facility quality
            sorted_facilities = sorted(self.facility_quality.items(), key=lambda x: x[1])
            
            # Distribute funds to improve facilities (weighted towards worst facilities)
            fund_per_improvement = 1000000  # 1M per 0.1 improvement
            remaining_fund = facility_fund
            
            for school_type, quality in sorted_facilities:
                if remaining_fund <= 0:
                    break
                    
                # Calculate possible improvement with available funds
                max_possible = min(0.9 - quality, remaining_fund / fund_per_improvement * 0.1)
                if max_possible > 0:
                    self.facility_quality[school_type] += max_possible
                    used_funds = max_possible / 0.1 * fund_per_improvement
                    remaining_fund -= used_funds
                    
                    # Facility improvements also affect school quality
                    self.school_qualities[school_type] += max_possible * 0.75
        
        # Apply subsidy effects to school quality
        subsidy = current_policy.get("subsidy_per_student", 0)
        if subsidy > 0:
            # Subsidies have more impact on lower-quality schools
            for school_type in self.school_qualities:
                if self.school_qualities[school_type] < 0.7:
                    # Calculate impact based on subsidy amount
                    impact = min(0.05, subsidy / 2000 * 0.05)
                    self.school_qualities[school_type] += impact
        
        # Ensure school qualities stay within bounds
        for school_type in self.school_qualities:
            self.school_qualities[school_type] = max(0.1, min(0.95, self.school_qualities[school_type]))

    def _update_student_achievements(self):
        """Update student achievement scores based on school quality"""
        for agent_id, school_type in self.agent_schools.items():
            if school_type is None or school_type == "private":
                continue
                
            current_achievement = self.student_achievement.get(agent_id, 0.5)
            school_quality = self.school_qualities.get(school_type, 0.5)
            
            # Achievement gradually moves toward school quality
            achievement_gap = school_quality - current_achievement
            adjustment = achievement_gap * 0.2  # 20% adjustment per epoch
            
            # Add small random factor
            random_factor = random.uniform(-0.05, 0.05)
            
            new_achievement = current_achievement + adjustment + random_factor
            self.student_achievement[agent_id] = max(0.1, min(0.95, new_achievement))
            
            # Update achievement by school type
            self.achievement_by_school_type[school_type] = [
                score for score in self.achievement_by_school_type[school_type] 
                if score != current_achievement
            ]
            self.achievement_by_school_type[school_type].append(new_achievement)

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide education system information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policy = self.education_policies.get(str(current_epoch), 
                                                    self.education_policies.get("0", {}))
        
        current_school = self.agent_schools.get(agent_id)
        
        # Calculate quality gap
        quality_gap = max(self.school_qualities.values()) - min(self.school_qualities.values())
        
        # School rankings with updated qualities
        school_rankings = sorted(self.school_qualities.items(), key=lambda x: x[1], reverse=True)
        
        # Get student achievement if applicable
        student_achievement = self.student_achievement.get(agent_id, None)
        achievement_percentile = None
        
        if student_achievement is not None and current_school:
            # Calculate percentile among all students
            all_achievements = list(self.student_achievement.values())
            if all_achievements:
                achievement_percentile = sum(1 for a in all_achievements if a <= student_achievement) / len(all_achievements) * 100
        
        # Get facility quality for current school
        facility_quality = self.facility_quality.get(current_school, 0) if current_school else 0
        
        # Check if new policies are being announced
        policy_changes = {}
        if current_epoch > 0:
            prev_policy = self.education_policies.get(str(current_epoch - 1), {})
            for key, value in current_policy.items():
                if key not in prev_policy or prev_policy[key] != value:
                    policy_changes[key] = value
        
        return {
            "school_rankings": [
                {"type": school[0], "quality_score": school[1]} 
                for school in school_rankings
            ],
            "education_policies": {
                "resource_allocation_mode": current_policy.get("resource_allocation", "traditional"),
                "transfer_allowed": current_policy.get("transfer_allowed", True),
                "transfer_quota": current_policy.get("transfer_quota", 0.1),
                "subsidy_amount": current_policy.get("subsidy_per_student", 0),
                "teacher_rotation": current_policy.get("teacher_rotation", False),
                "facility_upgrades": current_policy.get("facility_upgrade_fund", 0) > 0,
                "policy_changes": policy_changes
            },
            "school_resources": {
                "your_current_school": current_school,
                "current_school_quality": self.school_qualities.get(current_school, 0) if current_school else 0,
                "facility_quality": facility_quality,
                "quality_improvement": current_policy.get("teacher_rotation", False)
            },
            "student_performance": {
                "achievement_score": student_achievement,
                "achievement_percentile": achievement_percentile,
                "school_average": np.mean(self.achievement_by_school_type.get(current_school, [0.5])) if current_school else None
            },
            "district_info": {
                "quality_gap": quality_gap,
                "enrollment_pressure": {
                    school_type: self.school_enrollments[school_type] 
                    for school_type in self.school_qualities
                },
                "transfer_success_rate": self._calculate_transfer_success_rate()
            }
        }

    def _calculate_transfer_success_rate(self) -> Dict[str, float]:
        """Calculate transfer success rates by school type"""
        success_rates = {}
        for school_type in self.school_qualities.keys():
            requests = self.transfer_requests.get(school_type, 0)
            successes = self.transfer_success.get(school_type, 0)
            success_rates[school_type] = successes / requests if requests > 0 else 0
        return success_rates

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent education decisions"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policy = self.education_policies.get(str(current_epoch), 
                                                    self.education_policies.get("0", {}))
        
        # Update school qualities based on policy
        self._update_school_qualities(current_policy)
        
        # Reset transfer statistics
        self.transfer_requests = defaultdict(int)
        self.transfer_success = defaultdict(int)
        
        # Process agent decisions
        for agent_id, decisions in agent_decisions.items():
            if "EducationResourceSystem" not in decisions:
                continue
            
            if self.agent_schools.get(agent_id) is None:
                continue  # No school-age children
            
            decision = decisions["EducationResourceSystem"]
            action = decision.get("school_choice_action", "stay_current")
            preferred_type = decision.get("preferred_school_type", "average")
            
            current_school = self.agent_schools[agent_id]
            
            if action == "apply_transfer" and current_policy.get("transfer_allowed", True):
                self.transfer_requests[preferred_type] += 1
                
                # Process transfer based on quota
                transfer_quota = current_policy.get("transfer_quota", 0.1)
                current_enrollment = self.school_enrollments[preferred_type]
                max_transfers = int(current_enrollment * transfer_quota)
                
                if self.transfer_success[preferred_type] < max_transfers:
                    # Transfer successful
                    self.school_enrollments[current_school] -= 1
                    self.school_enrollments[preferred_type] += 1
                    self.agent_schools[agent_id] = preferred_type
                    self.transfer_success[preferred_type] += 1
                    self.agent_school_history[agent_id].append((current_epoch, preferred_type))
                    
                    self.logger.debug(f"Agent {agent_id} transferred from {current_school} to {preferred_type}")
            
            elif action == "private_school":
                # Move to private school (outside the public system)
                self.school_enrollments[current_school] -= 1
                self.agent_schools[agent_id] = "private"
                self.logger.debug(f"Agent {agent_id} moved to private school")
            
            elif action == "move_residence":
                # Simulate moving to a district with preferred school type
                if current_school != preferred_type:
                    self.school_enrollments[current_school] -= 1
                    self.school_enrollments[preferred_type] += 1
                    self.agent_schools[agent_id] = preferred_type
                    self.agent_school_history[agent_id].append((current_epoch, preferred_type))
                    self.logger.debug(f"Agent {agent_id} moved residence to attend {preferred_type} school")
        
        # Update student achievements based on school quality
        self._update_student_achievements()
        
        # Update socioeconomic mobility potential
        self._update_socioeconomic_mobility()
        
        # Calculate and record metrics
        quality_gap = max(self.school_qualities.values()) - min(self.school_qualities.values())
        self.quality_gap_history.append(quality_gap)
        
        enrollment_dist = dict(self.school_enrollments)
        self.enrollment_distribution_history.append(enrollment_dist)
        
        facility_improvement = sum(self.facility_quality.values()) / len(self.facility_quality)
        self.facility_improvement_history.append(facility_improvement)
        
        # Record average achievement
        avg_achievement = np.mean(list(self.student_achievement.values())) if self.student_achievement else 0.5
        self.achievement_history.append(avg_achievement)
        
        # Share state with other subsystems
        self._share_system_state(current_epoch, current_policy)
        
        self.logger.info(f"Epoch {current_epoch}: Quality gap={quality_gap:.3f}, "
                        f"Transfer requests={sum(self.transfer_requests.values())}, "
                        f"Successful transfers={sum(self.transfer_success.values())}, "
                        f"Avg achievement={avg_achievement:.3f}")

    def _share_system_state(self, current_epoch: int, current_policy: Dict[str, Any]):
        """Share state with other subsystems via system_state"""
        # Share school qualities
        for school_type, quality in self.school_qualities.items():
            # Store current quality
            self.system_state[f"school_quality_{school_type}"] = quality
            
            # Store previous quality for change tracking
            if current_epoch > 0:
                self.system_state[f"previous_school_quality_{school_type}"] = self.system_state.get(f"school_quality_{school_type}", quality)
        
        # Share facility qualities
        for school_type, quality in self.facility_quality.items():
            self.system_state[f"facility_quality_{school_type}"] = quality
        
        # Share enrollment pressures
        total_enrollment = sum(self.school_enrollments.values())
        avg_enrollment = total_enrollment / len(self.school_enrollments) if self.school_enrollments and len(self.school_enrollments) > 0 else 1
        
        for school_type, enrollment in self.school_enrollments.items():
            pressure = enrollment / avg_enrollment if avg_enrollment > 0 else 1
            self.system_state[f"enrollment_pressure_{school_type}"] = pressure
        
        # Share current policy
        self.system_state[f"education_policy_{current_epoch}"] = current_policy
        
        # Share achievement metrics
        for school_type, scores in self.achievement_by_school_type.items():
            if scores:
                self.system_state[f"achievement_avg_{school_type}"] = np.mean(scores)
        
        # Share overall metrics
        self.system_state["quality_gap"] = max(self.school_qualities.values()) - min(self.school_qualities.values())
        self.system_state["avg_achievement"] = np.mean(list(self.student_achievement.values())) if self.student_achievement else 0.5

    def _update_socioeconomic_mobility(self):
        """Update socioeconomic mobility potential based on school quality and achievement"""
        for agent_id, school_type in self.agent_schools.items():
            if school_type is None or school_type == "private":
                continue
                
            current_mobility = self.socioeconomic_mobility.get(agent_id, 0.5)
            school_quality = self.school_qualities.get(school_type, 0.5)
            achievement = self.student_achievement.get(agent_id, 0.5)
            
            # Mobility is influenced by both school quality and student achievement
            new_mobility = (school_quality * 0.4) + (achievement * 0.6)
            
            # Gradual adjustment
            adjustment = (new_mobility - current_mobility) * 0.3
            self.socioeconomic_mobility[agent_id] = current_mobility + adjustment
            
            # Update mobility by school type
            self.mobility_by_school_type[school_type] = [
                score for score in self.mobility_by_school_type[school_type] 
                if score != current_mobility
            ]
            self.mobility_by_school_type[school_type].append(self.socioeconomic_mobility[agent_id])

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate education balance policy effectiveness"""
        # Calculate quality convergence
        initial_gap = self.quality_gap_history[0] if self.quality_gap_history else 0.0
        final_gap = self.quality_gap_history[-1] if self.quality_gap_history else 0.0
        gap_reduction = float((initial_gap - final_gap) / initial_gap if initial_gap > 0 else 0.0)
        
        # Analyze enrollment distribution changes
        initial_dist = self.enrollment_distribution_history[0] if self.enrollment_distribution_history else {}
        final_dist = self.enrollment_distribution_history[-1] if self.enrollment_distribution_history else {}
        
        # Calculate Gini coefficient for enrollment distribution
        def calculate_gini(distribution):
            values = list(distribution.values())
            if not values or sum(values) == 0:
                return 0.0
            sorted_values = sorted(values)
            n = len(sorted_values)
            cumsum = np.cumsum(sorted_values)
            return float((2 * np.sum((np.arange(1, n+1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n)
        
        initial_gini = calculate_gini(initial_dist)
        final_gini = calculate_gini(final_dist)
        
        # Calculate achievement metrics
        initial_achievement = float(self.achievement_history[0] if self.achievement_history else 0.5)
        final_achievement = float(self.achievement_history[-1] if self.achievement_history else 0.5)
        achievement_improvement = float(final_achievement - initial_achievement)
        
        # Calculate achievement gap between school types
        achievement_by_type = {}
        achievement_gap = 0.0
        if self.achievement_by_school_type:
            for school_type, scores in self.achievement_by_school_type.items():
                if scores:
                    achievement_by_type[school_type] = float(np.mean(scores))
            
            if achievement_by_type:
                achievement_gap = float(max(achievement_by_type.values()) - min(achievement_by_type.values()))
        
        # Calculate mobility metrics
        mobility_by_type = {}
        mobility_gap = 0.0
        if self.mobility_by_school_type:
            for school_type, scores in self.mobility_by_school_type.items():
                if scores:
                    mobility_by_type[school_type] = float(np.mean(scores))
            
            if mobility_by_type:
                mobility_gap = float(max(mobility_by_type.values()) - min(mobility_by_type.values()))
        
        # Calculate transfer request success rate
        total_requests = sum(self.transfer_requests.values())
        total_success = sum(self.transfer_success.values())
        transfer_success_rate = float(total_success / total_requests if total_requests > 0 else 0.0)
        
        # Convert numpy arrays to lists for JSON serialization
        quality_gap_history = [float(x) for x in self.quality_gap_history] if self.quality_gap_history else []
        achievement_history = [float(x) for x in self.achievement_history] if self.achievement_history else []
        facility_improvement_history = [float(x) for x in self.facility_improvement_history] if self.facility_improvement_history else []
        
        evaluation_results = {
            "quality_metrics": {
                "initial_quality_gap": float(initial_gap),
                "final_quality_gap": float(final_gap),
                "gap_reduction_percentage": float(gap_reduction * 100),
                "final_school_qualities": {k: float(v) for k, v in self.school_qualities.items()},
                "facility_quality_improvement": float(facility_improvement_history[-1] - facility_improvement_history[0]) 
                    if facility_improvement_history else 0.0
            },
            "enrollment_metrics": {
                "total_transfer_requests": int(total_requests),
                "successful_transfers": int(total_success),
                "transfer_success_rate": transfer_success_rate,
                "enrollment_gini_coefficient": {
                    "initial": float(initial_gini),
                    "final": float(final_gini),
                    "improvement": float(initial_gini - final_gini)
                },
                "private_school_enrollment": int(self.school_enrollments.get("private", 0))
            },
            "achievement_metrics": {
                "initial_average_achievement": initial_achievement,
                "final_average_achievement": final_achievement,
                "achievement_improvement": achievement_improvement,
                "achievement_by_school_type": achievement_by_type,
                "achievement_gap": achievement_gap,
                "achievement_gap_reduction": float(1 - (achievement_gap / (max(achievement_by_type.values()) if achievement_by_type else 1)))
            },
            "socioeconomic_metrics": {
                "mobility_by_school_type": mobility_by_type,
                "mobility_gap": mobility_gap,
                "mobility_improvement": float(1 - (mobility_gap / (max(mobility_by_type.values()) if mobility_by_type else 1)))
            },
            "policy_effectiveness": {
                "quality_equalization_achieved": bool(final_gap < 0.2),
                "enrollment_balance_improved": bool(final_gini < initial_gini),
                "achievement_improved": bool(achievement_improvement > 0.1),
                "socioeconomic_mobility_improved": bool(mobility_gap < 0.3),
                "overall_success": bool(gap_reduction > 0.5 and final_gini < initial_gini and achievement_improvement > 0)
            },
            "time_series": {
                "quality_gap_history": quality_gap_history,
                "enrollment_history": [{k: int(v) for k, v in d.items()} for d in self.enrollment_distribution_history],
                "achievement_history": achievement_history,
                "facility_improvement_history": facility_improvement_history
            }
        }
        
        self.logger.info(f"evaluation_results={evaluation_results}")
        
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        # Calculate average achievement
        avg_achievement = np.mean(list(self.student_achievement.values())) if self.student_achievement else 0.5
        
        # Calculate achievement gap
        achievement_by_type = {}
        for school_type, scores in self.achievement_by_school_type.items():
            if scores:
                achievement_by_type[school_type] = np.mean(scores)
        
        achievement_gap = max(achievement_by_type.values()) - min(achievement_by_type.values()) if achievement_by_type else 0
        
        return {
            "school_qualities": dict(self.school_qualities),
            "facility_qualities": dict(self.facility_quality),
            "quality_gap": max(self.school_qualities.values()) - min(self.school_qualities.values()),
            "enrollment_distribution": dict(self.school_enrollments),
            "average_achievement": avg_achievement,
            "achievement_gap": achievement_gap,
            "transfer_activity": {
                "requests": sum(self.transfer_requests.values()),
                "success": sum(self.transfer_success.values()),
                "success_rate": sum(self.transfer_success.values()) / sum(self.transfer_requests.values()) 
                    if sum(self.transfer_requests.values()) > 0 else 0
            },
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 