import random
import numpy as np
from typing import Dict, Any, List, Optional
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class EmploymentBehaviorSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Optional[Any] = None, **kwargs):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        
        # Employment parameters
        self.base_employment_rate = config.get("base_employment_rate", 0.92)
        self.entrepreneurship_rate = config.get("entrepreneurship_rate", 0.05)
        self.job_market_elasticity = config.get("job_market_elasticity", 0.3)
        
        # Tax incentive effects on employment
        self.entrepreneur_tax_multiplier = config.get("entrepreneur_tax_multiplier", 1.5)
        self.employment_incentives = config.get("employment_incentives", {})
        
        # Agent employment tracking
        self.agent_employment = {}  # agent_id -> employment status
        self.agent_entrepreneurship = {}  # agent_id -> entrepreneurship status
        self.agent_productivity = {}  # agent_id -> productivity level
        self.agent_job_satisfaction = {}  # agent_id -> job satisfaction level
        
        # Employment statistics
        self.employment_rate = self.base_employment_rate
        self.entrepreneurship_count = 0
        self.job_creation_rate = 0.01
        self.employment_history = [self.base_employment_rate]
        self.entrepreneurship_history = [0]
        self.job_creation_history = [0]
        self.productivity_history = [1.0]  # Index 1.0 as baseline
        
        self.logger.info(f"EmploymentBehaviorSystem initialized with base employment rate: {self.base_employment_rate}")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent employment status based on their attributes"""
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            
            # Initialize employment based on economic status and occupation
            income_level = agent_data.get("economic_attributes", {}).get("income_level", "")
            occupation = agent_data.get("economic_attributes", {}).get("occupation", "")
            work_status = agent_data.get("economic_attributes", {}).get("work_status", "")
            
            # Determine employment status
            if "unemployed" in work_status.lower():
                employment_status = "unemployed"
            elif "part-time" in work_status.lower():
                employment_status = "part_time"
            else:
                employment_status = "full_time"
                
            # Determine entrepreneurship status
            is_entrepreneur = False
            if any(term in occupation.lower() for term in ["entrepreneur", "business owner", "self-employed", "freelance"]):
                is_entrepreneur = True
                self.entrepreneurship_count += 1
            
            # Set initial productivity based on income level
            if "high" in income_level.lower() or "wealthy" in income_level.lower():
                productivity = random.uniform(1.3, 2.0)
                job_satisfaction = random.uniform(0.7, 1.0)
            elif "mid" in income_level.lower():
                productivity = random.uniform(0.9, 1.3)
                job_satisfaction = random.uniform(0.5, 0.8)
            else:
                productivity = random.uniform(0.6, 0.9)
                job_satisfaction = random.uniform(0.3, 0.6)
            
            self.agent_employment[agent_id] = employment_status
            self.agent_entrepreneurship[agent_id] = is_entrepreneur
            self.agent_productivity[agent_id] = productivity
            self.agent_job_satisfaction[agent_id] = job_satisfaction
        
        # Calculate initial employment rate
        employed_count = sum(1 for status in self.agent_employment.values() if status != "unemployed")
        self.employment_rate = employed_count / max(len(self.agent_employment), 1)
        
        # Record initial history
        self.employment_history[0] = self.employment_rate
        self.entrepreneurship_history[0] = self.entrepreneurship_count / max(len(self.agent_employment), 1)
        
        # Calculate average productivity
        avg_productivity = sum(self.agent_productivity.values()) / max(len(self.agent_productivity), 1)
        self.productivity_history[0] = avg_productivity
        
        self.logger.info(f"Initialized {len(self.agent_employment)} agents with employment status")
        self.logger.info(f"Initial employment rate: {self.employment_rate:.2f}, entrepreneurs: {self.entrepreneurship_count}")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide employment information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_incentives = self.employment_incentives.get(str(current_epoch), 
                                                         self.employment_incentives.get("0", {}))
        
        # Calculate employment trend
        employment_trend = "stable"
        if len(self.employment_history) > 1:
            recent_change = self.employment_history[-1] - self.employment_history[-2]
            if recent_change > 0.01:
                employment_trend = "improving"
            elif recent_change < -0.01:
                employment_trend = "worsening"
        
        # Get agent's current employment status
        employment_status = self.agent_employment.get(agent_id, "unemployed")
        is_entrepreneur = self.agent_entrepreneurship.get(agent_id, False)
        productivity = self.agent_productivity.get(agent_id, 1.0)
        job_satisfaction = self.agent_job_satisfaction.get(agent_id, 0.5)
        
        # Calculate applicable tax incentives
        entrepreneur_tax_benefit = current_incentives.get("entrepreneur_tax_credit", 0.0)
        employment_tax_benefit = current_incentives.get("employment_tax_deduction", 0.0)
        
        return {
            "labor_market": {
                "employment_rate": self.employment_rate,
                "employment_trend": employment_trend,
                "entrepreneurship_rate": self.entrepreneurship_history[-1],
                "job_creation_rate": self.job_creation_rate,
                "average_productivity": self.productivity_history[-1]
            },
            "tax_incentives": {
                "entrepreneur_tax_credit": f"{entrepreneur_tax_benefit * 100:.1f}%",
                "employment_tax_deduction": f"{employment_tax_benefit * 100:.1f}%",
                "self_employment_benefits": current_incentives.get("self_employment_benefits", "None"),
                "policy_description": current_incentives.get("policy_description", "No active employment incentives")
            },
            "personal_employment": {
                "your_status": employment_status,
                "entrepreneur_status": "Yes" if is_entrepreneur else "No",
                "productivity_level": productivity,
                "job_satisfaction": job_satisfaction,
                "applicable_benefits": "Entrepreneur benefits" if is_entrepreneur and entrepreneur_tax_benefit > 0 else 
                                       "Employee benefits" if employment_status != "unemployed" and employment_tax_benefit > 0 else
                                       "None"
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent employment decisions and update labor market state"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_incentives = self.employment_incentives.get(str(current_epoch), 
                                                         self.employment_incentives.get("0", {}))
        
        # Get policy parameters
        entrepreneur_tax_credit = current_incentives.get("entrepreneur_tax_credit", 0.0)
        employment_tax_deduction = current_incentives.get("employment_tax_deduction", 0.0)
        
        # Get economic indicators from blackboard if available - add error handling
        try:
            economic_gdp = self._get_from_blackboard("economic_gdp", None)
            social_sentiment = self._get_from_blackboard("social_sentiment", 0.0)
        except Exception as e:
            # Log the error and use default values
            self.logger.debug(f"Could not retrieve data from blackboard: {e}")
            economic_gdp = None
            social_sentiment = 0.0
        
        # Economic feedback effect on employment - 更合理的就业响应
        economic_employment_effect = 0
        if economic_gdp is not None and len(self.employment_history) > 1:
            prev_gdp = self._get_from_blackboard("previous_gdp_employment", economic_gdp)
            if prev_gdp and prev_gdp > 0:
                gdp_growth = (economic_gdp - prev_gdp) / max(prev_gdp, 1)
                # 限制GDP增长率的影响范围
                gdp_growth = max(min(gdp_growth, 0.1), -0.1)
                economic_employment_effect = gdp_growth * self.job_market_elasticity
            self._post_to_blackboard("previous_gdp_employment", economic_gdp)
        
        # 社会舆论对就业的影响
        sentiment_employment_effect = 0
        if social_sentiment is not None:
            sentiment_employment_effect = social_sentiment * 0.02  # 舆论积极对就业有轻微正面影响
        
        # 政策效应计算 - 增强税收政策对创业的影响
        entrepreneur_policy_effect = 1.0 + (entrepreneur_tax_credit * 3.0)  # 强化创业税收优惠效果
        employment_policy_effect = 1.0 + (employment_tax_deduction * 2.0)   # 强化就业税收优惠效果
        
        # Reset counters for this epoch
        new_employment_count = 0
        new_entrepreneurship_count = 0
        total_productivity = 0
        job_creation = 0
        
        # Process each agent's decision
        for agent_id, decisions in agent_decisions.items():
            if "EmploymentBehaviorSystem" not in decisions:
                continue
                
            decision = decisions["EmploymentBehaviorSystem"]
            career_change_intention = float(decision.get("career_change_intention", 0.1))
            entrepreneurship_intention = float(decision.get("entrepreneurship_intention", 0.1))
            productivity_effort = float(decision.get("productivity_effort", 0.5))
            
            # Current status
            current_status = self.agent_employment.get(agent_id, "unemployed")
            is_entrepreneur = self.agent_entrepreneurship.get(agent_id, False)
            current_productivity = self.agent_productivity.get(agent_id, 1.0)
            current_satisfaction = self.agent_job_satisfaction.get(agent_id, 0.5)
            
            # 更新生产力 - 基于个人努力和政策激励
            productivity_boost = productivity_effort * 0.5  # 个人努力提高生产力
            
            # 税收政策对生产力的影响
            policy_productivity_effect = 0
            if is_entrepreneur and entrepreneur_tax_credit > 0:
                policy_productivity_effect = entrepreneur_tax_credit * 0.8  # 创业税收优惠对生产力的影响
            elif current_status != "unemployed" and employment_tax_deduction > 0:
                policy_productivity_effect = employment_tax_deduction * 0.6  # 就业税收优惠对生产力的影响
            
            # 新的生产力水平 - 累积增长模型
            new_productivity = current_productivity * (1.0 + 0.05 + productivity_boost + policy_productivity_effect)
            
            # 限制生产力增长幅度
            max_productivity_growth = 0.15  # 最大15%的生产力增长
            productivity_change = (new_productivity / current_productivity) - 1.0
            if productivity_change > max_productivity_growth:
                new_productivity = current_productivity * (1.0 + max_productivity_growth)
            
            # 更新就业满意度
            new_satisfaction = current_satisfaction
            if productivity_effort > 0.7:
                # 高努力可能导致满意度小幅下降
                new_satisfaction = max(0.1, current_satisfaction - 0.05)
            elif productivity_effort < 0.3:
                # 低努力可能导致满意度小幅上升
                new_satisfaction = min(1.0, current_satisfaction + 0.05)
            
            # 政策对满意度的影响
            if is_entrepreneur and entrepreneur_tax_credit > 0:
                new_satisfaction = min(1.0, new_satisfaction + entrepreneur_tax_credit * 0.3)
            elif current_status != "unemployed" and employment_tax_deduction > 0:
                new_satisfaction = min(1.0, new_satisfaction + employment_tax_deduction * 0.2)
            
            # Update employment status based on decisions and economic conditions
            new_status = current_status
            if random.random() < career_change_intention:
                # Agent is considering a career change
                if current_status == "unemployed":
                    # Unemployed agent looking for work
                    employment_chance = 0.3 + economic_employment_effect + sentiment_employment_effect
                    employment_chance *= employment_policy_effect  # 就业政策影响
                    if random.random() < employment_chance:
                        new_status = "part_time" if random.random() < 0.3 else "full_time"
                elif current_status == "part_time" and economic_employment_effect > 0:
                    # Part-time worker looking for full-time
                    full_time_chance = 0.4 + economic_employment_effect + sentiment_employment_effect
                    full_time_chance *= employment_policy_effect  # 就业政策影响
                    if random.random() < full_time_chance:
                        new_status = "full_time"
                elif economic_employment_effect < -0.05:
                    # Economic downturn risk of job loss
                    job_loss_risk = 0.1 - economic_employment_effect  # 经济下行时失业风险增加
                    # 政策保护就业
                    job_loss_risk /= employment_policy_effect
                    if random.random() < job_loss_risk:
                        if current_status == "full_time":
                            new_status = "part_time"  # Downgrade to part-time first
                        else:
                            new_status = "unemployed"
            
            # Update entrepreneurship status based on intention and economic conditions
            new_entrepreneur = is_entrepreneur
            if random.random() < entrepreneurship_intention:
                # Agent considering entrepreneurship
                if not is_entrepreneur:
                    # Starting a business
                    entrepreneur_chance = 0.05 + economic_employment_effect + (social_sentiment * 0.1)
                    # 创业税收优惠大幅提高创业概率
                    entrepreneur_chance *= entrepreneur_policy_effect
                    if random.random() < entrepreneur_chance:
                        new_entrepreneur = True
                else:
                    # Potentially giving up entrepreneurship in bad economy
                    failure_risk = 0.05 - economic_employment_effect
                    # 税收优惠降低创业失败风险
                    failure_risk /= entrepreneur_policy_effect
                    if random.random() < failure_risk:
                        new_entrepreneur = False
            
            # Calculate job creation (only entrepreneurs create jobs)
            agent_job_creation = 0
            if new_entrepreneur:
                # Entrepreneurs with high productivity create more jobs
                base_job_creation = 0.01 * (new_productivity - 1.0) * 5  # 提高生产率对创造就业的贡献
                # 创业税收优惠提高创业者创造就业的能力
                policy_job_creation_boost = entrepreneur_tax_credit * 0.2
                agent_job_creation = base_job_creation + policy_job_creation_boost
                job_creation += agent_job_creation
            
            # Update agent data
            self.agent_employment[agent_id] = new_status
            self.agent_entrepreneurship[agent_id] = new_entrepreneur
            self.agent_productivity[agent_id] = new_productivity
            self.agent_job_satisfaction[agent_id] = new_satisfaction
            
            # Count employment status
            if new_status != "unemployed":
                new_employment_count += 1
            if new_entrepreneur:
                new_entrepreneurship_count += 1
                
            total_productivity += new_productivity
        
        # Calculate new employment rate
        self.employment_rate = new_employment_count / max(len(self.agent_employment), 1)
        self.entrepreneurship_count = new_entrepreneurship_count
        self.job_creation_rate = job_creation / max(len(self.agent_employment), 1)
        
        # Calculate average productivity
        avg_productivity = total_productivity / max(len(self.agent_productivity), 1)
        
        # Update history
        self.employment_history.append(self.employment_rate)
        self.entrepreneurship_history.append(self.entrepreneurship_count / max(len(self.agent_employment), 1))
        self.job_creation_history.append(self.job_creation_rate)
        self.productivity_history.append(avg_productivity)
        
        self.logger.info(f"Epoch {current_epoch}: Employment={self.employment_rate:.2f}, "
                        f"Entrepreneurs={self.entrepreneurship_count}, "
                        f"Jobs Created={self.job_creation_rate:.3f}, "
                        f"Productivity={avg_productivity:.2f}")
        
        # Post key metrics to blackboard for other subsystems - add error handling
        try:
            self._post_to_blackboard("employment_rate", self.employment_rate)
            self._post_to_blackboard("entrepreneurship_rate", self.entrepreneurship_count / max(len(self.agent_employment), 1))
            self._post_to_blackboard("productivity_index", avg_productivity)
            
            # 为每个代理单独发布收入信息，供消费系统使用
            for agent_id, productivity in self.agent_productivity.items():
                # 基于就业状态和生产力估算收入水平
                status = self.agent_employment.get(agent_id, "unemployed")
                is_entrepreneur = self.agent_entrepreneurship.get(agent_id, False)
                
                if is_entrepreneur:
                    # 创业者收入基于生产力和税收优惠
                    entrepreneur_income_factor = productivity * self.entrepreneur_tax_multiplier
                    entrepreneur_income_factor *= entrepreneur_policy_effect  # 税收政策提高创业收入
                    estimated_income = 8000 * entrepreneur_income_factor
                elif status == "full_time":
                    # 全职员工收入基于生产力和税收优惠
                    employment_income_factor = productivity * employment_policy_effect
                    estimated_income = 6000 * employment_income_factor
                elif status == "part_time":
                    # 兼职收入
                    employment_income_factor = productivity * employment_policy_effect
                    estimated_income = 3000 * employment_income_factor
                else:
                    # 失业救济
                    estimated_income = 1500
                
                # 发布到黑板
                self._post_to_blackboard(f"income_{agent_id}", estimated_income)
        except Exception as e:
            # Log the error but continue execution
            self.logger.debug(f"Could not post employment data to blackboard: {e}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the effectiveness of tax incentives on employment and entrepreneurship"""
        # Calculate key metrics
        employment_change = (self.employment_history[-1] - self.employment_history[0]) * 100  # Percentage points
        entrepreneurship_change = (self.entrepreneurship_history[-1] - self.entrepreneurship_history[0]) * 100  # Percentage points
        productivity_growth = ((self.productivity_history[-1] / self.productivity_history[0]) - 1) * 100  # Percentage
        
        # Calculate job creation total
        total_jobs_created = sum(self.job_creation_history)
        
        # Calculate employment volatility
        employment_volatility = np.std(self.employment_history) * 100  # Convert to percentage points
        
        # Calculate correlation between entrepreneurship and job creation
        if len(self.entrepreneurship_history) > 2 and len(self.entrepreneurship_history) == len(self.job_creation_history):
            # Simple correlation calculation
            entrepreneur_job_correlation = np.corrcoef(self.entrepreneurship_history, self.job_creation_history)[0, 1]
        else:
            # Make sure arrays are the same length before correlation
            min_length = min(len(self.entrepreneurship_history), len(self.job_creation_history))
            if min_length > 2:
                entrepreneur_job_correlation = np.corrcoef(
                    self.entrepreneurship_history[-min_length:], 
                    self.job_creation_history[-min_length:]
                )[0, 1]
            else:
                entrepreneur_job_correlation = 0
            
        # Calculate employment distribution
        employment_distribution = {
            "unemployed": 0,
            "part_time": 0,
            "full_time": 0
        }
        
        for status in self.agent_employment.values():
            if status in employment_distribution:
                employment_distribution[status] += 1
                
        # Calculate job satisfaction statistics
        satisfaction_levels = list(self.agent_job_satisfaction.values())
        avg_satisfaction = np.mean(satisfaction_levels) if satisfaction_levels else 0
        satisfaction_std = np.std(satisfaction_levels) if len(satisfaction_levels) > 1 else 0
        
        evaluation_results = {
            "employment_metrics": {
                "final_employment_rate": self.employment_history[-1],
                "employment_change_percentage_points": employment_change,
                "employment_volatility": employment_volatility
            },
            "entrepreneurship_metrics": {
                "final_entrepreneurship_rate": self.entrepreneurship_history[-1],
                "entrepreneurship_growth_percentage_points": entrepreneurship_change,
                "total_jobs_created": total_jobs_created,
                "entrepreneur_job_creation_correlation": entrepreneur_job_correlation
            },
            "productivity_metrics": {
                "final_productivity_index": self.productivity_history[-1],
                "productivity_growth_percent": productivity_growth
            },
            "distribution_metrics": {
                "employment_distribution": employment_distribution,
                "average_job_satisfaction": avg_satisfaction,
                "satisfaction_inequality": satisfaction_std
            },
            "time_series": {
                "employment_history": self.employment_history,
                "entrepreneurship_history": self.entrepreneurship_history,
                "job_creation_history": self.job_creation_history,
                "productivity_history": self.productivity_history
            }
        }
        
        self.logger.info(f"Employment Evaluation Results: {evaluation_results}")
        
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        return {
            "employment_rate": self.employment_rate,
            "entrepreneurship_count": self.entrepreneurship_count,
            "job_creation_rate": self.job_creation_rate,
            "average_productivity": self.productivity_history[-1] if self.productivity_history else 1.0,
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 