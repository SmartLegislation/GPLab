import random
import numpy as np
from typing import Dict, Any, List, Optional
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class EconomicBehaviorSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Optional[Any] = None, **kwargs):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        
        # Economic parameters
        self.consumption_multiplier = config.get("consumption_multiplier", 1.2)
        self.investment_multiplier = config.get("investment_multiplier", 1.3)
        self.saving_rate_base = config.get("saving_rate_base", 0.2)
        
        # Tax incentive policies by epoch
        self.tax_policies = config.get("tax_policies", {})
        
        # Agent economic behavior tracking
        self.agent_consumption = {}  # agent_id -> consumption level
        self.agent_investment = {}   # agent_id -> investment level
        self.agent_savings = {}      # agent_id -> savings level
        self.agent_income = {}       # agent_id -> income level
        
        # Market statistics
        self.total_consumption = 0
        self.total_investment = 0
        self.total_savings = 0
        self.consumption_history = []
        self.investment_history = []
        self.savings_history = []
        self.gdp_estimate = 0
        self.gdp_history = []
        
        self.logger.info(f"EconomicBehaviorSystem initialized with base consumption multiplier: {self.consumption_multiplier}")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent economic behaviors based on their economic status"""
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            
            # Initialize economic behaviors based on income and assets
            income_level = agent_data.get("economic_attributes", {}).get("income_level", "")
            
            # Simple heuristic for initial consumption and investment levels
            if "high" in income_level.lower() or "wealthy" in income_level.lower():
                consumption_base = random.uniform(5000, 10000)
                investment_base = random.uniform(3000, 8000)
                savings_base = random.uniform(10000, 20000)
                income_base = random.uniform(10000, 15000)
            elif "mid" in income_level.lower():
                consumption_base = random.uniform(3000, 5000)
                investment_base = random.uniform(1000, 3000)
                savings_base = random.uniform(5000, 10000)
                income_base = random.uniform(6000, 10000)
            else:
                consumption_base = random.uniform(1000, 3000)
                investment_base = random.uniform(500, 1500)
                savings_base = random.uniform(1000, 5000)
                income_base = random.uniform(3000, 6000)
            
            self.agent_consumption[agent_id] = consumption_base
            self.agent_investment[agent_id] = investment_base
            self.agent_savings[agent_id] = savings_base
            self.agent_income[agent_id] = income_base
        
        # Calculate initial totals
        self.total_consumption = sum(self.agent_consumption.values())
        self.total_investment = sum(self.agent_investment.values())
        self.total_savings = sum(self.agent_savings.values())
        self.gdp_estimate = self.total_consumption + self.total_investment
        
        # Record initial history
        self.consumption_history.append(self.total_consumption)
        self.investment_history.append(self.total_investment)
        self.savings_history.append(self.total_savings)
        self.gdp_history.append(self.gdp_estimate)
        
        self.logger.info(f"Initialized {len(self.agent_consumption)} agents with economic behaviors")
        self.logger.info(f"Initial consumption: {self.total_consumption}, investment: {self.total_investment}, savings: {self.total_savings}")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide economic information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policies = self.tax_policies.get(str(current_epoch), 
                                               self.tax_policies.get("0", {}))
        
        # Calculate economic trends
        consumption_trend = "stable"
        investment_trend = "stable"
        
        if len(self.consumption_history) > 1:
            recent_consumption_change = (self.consumption_history[-1] - self.consumption_history[-2]) / max(self.consumption_history[-2], 1)
            if recent_consumption_change > 0.05:
                consumption_trend = "rising"
            elif recent_consumption_change < -0.05:
                consumption_trend = "falling"
                
        if len(self.investment_history) > 1:
            recent_investment_change = (self.investment_history[-1] - self.investment_history[-2]) / max(self.investment_history[-2], 1)
            if recent_investment_change > 0.05:
                investment_trend = "rising"
            elif recent_investment_change < -0.05:
                investment_trend = "falling"
        
        # Agent's current economic status
        agent_consumption = self.agent_consumption.get(agent_id, 0)
        agent_investment = self.agent_investment.get(agent_id, 0)
        agent_savings = self.agent_savings.get(agent_id, 0)
        agent_income = self.agent_income.get(agent_id, 0)
        
        # Calculate tax benefits based on current policies
        consumption_tax_benefit = current_policies.get("consumption_tax_reduction", 0.0)
        investment_tax_benefit = current_policies.get("investment_tax_deduction", 0.0)
        
        return {
            "economic_indicators": {
                "gdp_growth": ((self.gdp_history[-1] - self.gdp_history[0]) / max(self.gdp_history[0], 1) * 100) if len(self.gdp_history) > 1 else 0,
                "consumption_trend": consumption_trend,
                "investment_trend": investment_trend,
                "overall_economic_climate": "expanding" if self.gdp_history[-1] >= self.gdp_history[0] else "contracting"
            },
            "tax_incentives": {
                "consumption_tax_reduction": f"{consumption_tax_benefit * 100:.1f}%",
                "investment_tax_deduction": f"{investment_tax_benefit * 100:.1f}%",
                "tax_free_savings_threshold": current_policies.get("tax_free_savings_threshold", 0),
                "policy_description": current_policies.get("policy_description", "No active tax incentive policy")
            },
            "personal_economic_status": {
                "your_income_level": agent_income,
                "your_consumption_level": agent_consumption,
                "your_investment_level": agent_investment,
                "your_savings_level": agent_savings,
                "estimated_tax_benefit": round(agent_consumption * consumption_tax_benefit + agent_investment * investment_tax_benefit, 2)
            },
            "market_trends": {
                "total_consumption": self.total_consumption,
                "total_investment": self.total_investment,
                "consumption_growth": ((self.consumption_history[-1] - self.consumption_history[0]) / max(self.consumption_history[0], 1) * 100) if len(self.consumption_history) > 1 else 0,
                "investment_growth": ((self.investment_history[-1] - self.investment_history[0]) / max(self.investment_history[0], 1) * 100) if len(self.investment_history) > 1 else 0
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent economic decisions and update economic state"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policies = self.tax_policies.get(str(current_epoch), 
                                               self.tax_policies.get("0", {}))
        
        # Get policy parameters
        consumption_tax_reduction = current_policies.get("consumption_tax_reduction", 0.0)
        investment_tax_deduction = current_policies.get("investment_tax_deduction", 0.0)
        tax_free_savings_threshold = current_policies.get("tax_free_savings_threshold", 0)
        
        # Update consumption and investment multipliers based on tax incentives
        # 更合理的税收乘数效应
        consumption_policy_effect = 1.0 + (consumption_tax_reduction * 1.5)
        investment_policy_effect = 1.0 + (investment_tax_deduction * 2.0)
        
        # Reset totals for this epoch
        new_total_consumption = 0
        new_total_investment = 0
        new_total_savings = 0
        
        # Get economic factors from other subsystems - add error handling
        try:
            employment_rate = self._get_from_blackboard("employment_rate", 0.95)
            social_sentiment = self._get_from_blackboard("social_sentiment", 0.0)
            productivity_index = self._get_from_blackboard("productivity_index", 1.0)
        except Exception as e:
            # Log the error and use default values
            self.logger.debug(f"Could not retrieve data from blackboard: {e}")
            employment_rate = 0.95
            social_sentiment = 0.0
            productivity_index = 1.0
        
        # Economic growth factor - based on productivity and sentiment
        economic_growth_factor = 1.0 + (0.02 * (productivity_index - 1.0)) + (0.01 * social_sentiment)
        
        # Process each agent's decision
        for agent_id, decisions in agent_decisions.items():
            if "EconomicBehaviorSystem" not in decisions:
                continue
                
            decision = decisions["EconomicBehaviorSystem"]
            consumption_intention = float(decision.get("consumption_intention", 0.5))
            investment_intention = float(decision.get("investment_intention", 0.3))
            saving_intention = float(decision.get("saving_intention", 0.2))
            
            # Normalize intentions to sum to 1.0
            total_intention = consumption_intention + investment_intention + saving_intention
            if total_intention > 0:
                consumption_intention /= total_intention
                investment_intention /= total_intention
                saving_intention /= total_intention
            
            # Get current levels
            current_consumption = self.agent_consumption.get(agent_id, 0)
            current_investment = self.agent_investment.get(agent_id, 0)
            current_savings = self.agent_savings.get(agent_id, 0)
            current_income = self.agent_income.get(agent_id, 0)
            
            # 收入稳定增长 - 确保经济不会崩溃
            # 基于现有收入的增长模型，包括投资回报和经济环境因素
            investment_return = current_investment * 0.08  # 8% 投资回报率
            savings_interest = current_savings * 0.02     # 2% 储蓄利率
            
            # 收入增长模型 - 考虑经济环境、生产率和消费刺激
            new_income = current_income * economic_growth_factor
            new_income += investment_return + savings_interest
            
            # 收入波动控制 - 防止极端变化
            min_income = current_income * 0.95  # 最大下降5%
            max_income = current_income * 1.15  # 最大上升15%
            new_income = max(min(new_income, max_income), min_income)
            
            # 确保最低收入水平
            new_income = max(new_income, 2000)
            
            # 更新代理收入
            self.agent_income[agent_id] = new_income
            
            # 应用税收激励政策和个人意愿确定新的消费、投资和储蓄水平
            # 消费水平 = 收入 * 消费意愿 * 基础消费乘数 * 政策影响
            new_consumption = new_income * consumption_intention * self.consumption_multiplier * consumption_policy_effect
            
            # 投资水平 = 收入 * 投资意愿 * 基础投资乘数 * 政策影响
            new_investment = new_income * investment_intention * self.investment_multiplier * investment_policy_effect
            
            # 储蓄水平 = 收入 * 储蓄意愿 + 储蓄政策影响
            new_savings_contribution = new_income * saving_intention
            
            # 应用储蓄阈值激励
            if current_savings < tax_free_savings_threshold:
                new_savings_contribution *= 1.2  # 储蓄低于阈值时的额外激励
            
            # 更新代理消费和投资水平
            self.agent_consumption[agent_id] = new_consumption
            self.agent_investment[agent_id] = new_investment
            
            # 更新储蓄 - 累加新的储蓄贡献，减去部分消费影响
            # 消费只消耗一小部分储蓄
            savings_reduction = new_consumption * 0.05  # 减少消费对储蓄的侵蚀
            self.agent_savings[agent_id] = current_savings + new_savings_contribution - savings_reduction
            
            # 确保储蓄不会为负
            self.agent_savings[agent_id] = max(self.agent_savings[agent_id], 0)
            
            # 累加总量
            new_total_consumption += new_consumption
            new_total_investment += new_investment
            new_total_savings += self.agent_savings[agent_id]
        
        # 更新系统总量
        self.total_consumption = new_total_consumption
        self.total_investment = new_total_investment
        self.total_savings = new_total_savings
        self.gdp_estimate = self.total_consumption + self.total_investment
        
        # 更新历史记录
        self.consumption_history.append(self.total_consumption)
        self.investment_history.append(self.total_investment)
        self.savings_history.append(self.total_savings)
        self.gdp_history.append(self.gdp_estimate)
        
        self.logger.info(f"Epoch {current_epoch}: GDP={self.gdp_estimate:.0f}, "
                        f"Consumption={self.total_consumption:.0f}, Investment={self.total_investment:.0f}, "
                        f"Savings={self.total_savings:.0f}")
        
        # 将关键指标发布到黑板供其他子系统使用 - add error handling
        try:
            self._post_to_blackboard("economic_consumption", self.total_consumption)
            self._post_to_blackboard("economic_investment", self.total_investment)
            self._post_to_blackboard("economic_gdp", self.gdp_estimate)
        except Exception as e:
            # Log the error but continue execution
            self.logger.debug(f"Could not post economic data to blackboard: {e}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the effectiveness of tax incentive policies on economic behaviors"""
        # Calculate key metrics
        consumption_growth = ((self.consumption_history[-1] - self.consumption_history[0]) / max(self.consumption_history[0], 1)) * 100
        investment_growth = ((self.investment_history[-1] - self.investment_history[0]) / max(self.investment_history[0], 1)) * 100
        gdp_growth = ((self.gdp_history[-1] - self.gdp_history[0]) / max(self.gdp_history[0], 1)) * 100
        savings_change = ((self.savings_history[-1] - self.savings_history[0]) / max(self.savings_history[0], 1)) * 100
        
        # Calculate volatility
        consumption_volatility = np.std(self.consumption_history) / max(np.mean(self.consumption_history), 1) * 100
        investment_volatility = np.std(self.investment_history) / max(np.mean(self.investment_history), 1) * 100
        
        # Analyze economic distribution
        consumption_distribution = {}
        for agent_id, consumption in self.agent_consumption.items():
            bracket = int(consumption / 1000) * 1000
            consumption_distribution[bracket] = consumption_distribution.get(bracket, 0) + 1
        
        # Calculate inequality measures (Gini coefficient)
        investment_values = list(self.agent_investment.values())
        savings_values = list(self.agent_savings.values())
        investment_gini = self._calculate_gini(investment_values)
        savings_gini = self._calculate_gini(savings_values)
        
        # Calculate policy effectiveness
        # Tax elasticity: percent change in outcome / percent change in tax rate
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policies = self.tax_policies.get(str(current_epoch), 
                                               self.tax_policies.get("0", {}))
        
        consumption_tax_elasticity = 0
        investment_tax_elasticity = 0
        
        consumption_tax_reduction = current_policies.get("consumption_tax_reduction", 0.0)
        if consumption_tax_reduction > 0:
            consumption_tax_elasticity = consumption_growth / (consumption_tax_reduction * 100)
            
        investment_tax_deduction = current_policies.get("investment_tax_deduction", 0.0)
        if investment_tax_deduction > 0:
            investment_tax_elasticity = investment_growth / (investment_tax_deduction * 100)
        
        overall_effectiveness = (consumption_tax_elasticity + investment_tax_elasticity) / 2 if (consumption_tax_reduction > 0 or investment_tax_deduction > 0) else 0
        
        results = {
            "growth_metrics": {
                "gdp_growth_percent": gdp_growth,
                "consumption_growth_percent": consumption_growth,
                "investment_growth_percent": investment_growth,
                "savings_change_percent": savings_change
            },
            "volatility_metrics": {
                "consumption_volatility": consumption_volatility,
                "investment_volatility": investment_volatility
            },
            "distribution_metrics": {
                "consumption_distribution": consumption_distribution,
                "investment_gini": investment_gini,
                "savings_gini": savings_gini
            },
            "policy_effectiveness": {
                "consumption_tax_elasticity": consumption_tax_elasticity,
                "investment_tax_elasticity": investment_tax_elasticity,
                "overall_policy_effectiveness": overall_effectiveness
            },
            "time_series": {
                "gdp_history": self.gdp_history,
                "consumption_history": self.consumption_history,
                "investment_history": self.investment_history,
                "savings_history": self.savings_history
            }
        }
        
        self.logger.info(f"EconomicBehaviorSystem evaluation results: {results}")
        return results
        

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return the current state for database persistence"""
        return {
            "agent_consumption": self.agent_consumption,
            "agent_investment": self.agent_investment,
            "agent_savings": self.agent_savings,
            "agent_income": self.agent_income,
            "total_consumption": self.total_consumption,
            "total_investment": self.total_investment,
            "total_savings": self.total_savings,
            "gdp_estimate": self.gdp_estimate,
            "consumption_history": self.consumption_history,
            "investment_history": self.investment_history,
            "savings_history": self.savings_history,
            "gdp_history": self.gdp_history
        }

    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient as a measure of inequality"""
        if not values or sum(values) == 0:
            return 1.0
            
        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(sorted_values))
        
        return gini 