import random
import numpy as np
from typing import Dict, Any, List, Optional
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class ConsumptionBehaviorSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Optional[Any] = None, **kwargs):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        
        # Consumption parameters
        self.base_consumption_level = config.get("base_consumption_level", 1000)
        self.consumption_volatility = config.get("consumption_volatility", 0.1)
        self.luxury_goods_threshold = config.get("luxury_goods_threshold", 5000)
        
        # Tax incentive effects on consumption
        self.consumption_tax_policies = config.get("consumption_tax_policies", {})
        
        # Agent consumption tracking
        self.agent_consumption = {}  # agent_id -> total consumption amount
        self.agent_consumption_categories = {}  # agent_id -> {category: amount}
        self.agent_disposable_income = {}  # agent_id -> disposable income estimate
        
        # Consumption categories
        self.categories = ["essentials", "leisure", "durables", "luxury"]
        
        # Market statistics
        self.total_consumption = 0
        self.category_consumption = {category: 0 for category in self.categories}
        self.consumption_history = []
        self.category_history = {category: [] for category in self.categories}
        
        self.logger.info(f"ConsumptionBehaviorSystem initialized with base consumption level: {self.base_consumption_level}")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent consumption patterns based on their economic status"""
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            
            # Initialize consumption based on income level and consumption preferences
            income_level = agent_data.get("economic_attributes", {}).get("income_level", "")
            consumption_level = agent_data.get("economic_attributes", {}).get("consumption_level", "")
            
            # Estimate disposable income based on income level
            if "high" in income_level.lower() or "wealthy" in income_level.lower():
                base_income = random.uniform(10000, 20000)
                essentials_ratio = 0.3
                luxury_ratio = 0.2
            elif "mid" in income_level.lower():
                base_income = random.uniform(5000, 10000)
                essentials_ratio = 0.4
                luxury_ratio = 0.1
            else:
                base_income = random.uniform(2000, 5000)
                essentials_ratio = 0.6
                luxury_ratio = 0.05
                
            # Adjust consumption level based on described behavior
            consumption_multiplier = 1.0
            if "high" in consumption_level.lower() or "luxury" in consumption_level.lower():
                consumption_multiplier = 1.3
            elif "moderate" in consumption_level.lower() or "selective" in consumption_level.lower():
                consumption_multiplier = 1.0
            elif "frugal" in consumption_level.lower() or "minimal" in consumption_level.lower():
                consumption_multiplier = 0.7
                
            # Calculate total consumption and category breakdown
            total_consumption = base_income * consumption_multiplier * 0.8  # 80% of income goes to consumption
            
            # Initialize category distribution
            categories_consumption = {
                "essentials": total_consumption * essentials_ratio,
                "leisure": total_consumption * (0.3 - luxury_ratio/2),
                "durables": total_consumption * (0.2 - luxury_ratio/2),
                "luxury": total_consumption * luxury_ratio
            }
            
            self.agent_disposable_income[agent_id] = base_income
            self.agent_consumption[agent_id] = total_consumption
            self.agent_consumption_categories[agent_id] = categories_consumption
        
        # Calculate initial totals
        self.total_consumption = sum(self.agent_consumption.values())
        for category in self.categories:
            self.category_consumption[category] = sum(
                agent_cats.get(category, 0) 
                for agent_cats in self.agent_consumption_categories.values()
            )
        
        # Record initial history
        self.consumption_history.append(self.total_consumption)
        for category in self.categories:
            self.category_history[category].append(self.category_consumption[category])
        
        self.logger.info(f"Initialized {len(self.agent_consumption)} agents with consumption patterns")
        self.logger.info(f"Initial total consumption: {self.total_consumption:.0f}")
        self.logger.info(f"Initial category breakdown: {self.category_consumption}")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide consumption information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policies = self.consumption_tax_policies.get(str(current_epoch), 
                                                          self.consumption_tax_policies.get("0", {}))
        
        # Calculate consumption trend
        consumption_trend = "stable"
        if len(self.consumption_history) > 1:
            recent_change = (self.consumption_history[-1] - self.consumption_history[-2]) / self.consumption_history[-2]
            if recent_change > 0.05:
                consumption_trend = "increasing"
            elif recent_change < -0.05:
                consumption_trend = "decreasing"
        
        # Get agent's current consumption data
        total_consumption = self.agent_consumption.get(agent_id, 0)
        category_breakdown = self.agent_consumption_categories.get(agent_id, {})
        disposable_income = self.agent_disposable_income.get(agent_id, 0)
        
        # Calculate applicable tax incentives
        category_incentives = {}
        for category in self.categories:
            incentive = current_policies.get(f"{category}_tax_reduction", 0.0)
            category_incentives[category] = f"{incentive * 100:.1f}%"
        
        # Calculate savings rate
        savings_rate = 1.0 - (total_consumption / disposable_income) if disposable_income > 0 else 0
        
        return {
            "market_trends": {
                "overall_consumption": self.total_consumption,
                "consumption_trend": consumption_trend,
                "category_trends": {
                    category: "growing" if len(self.category_history[category]) > 1 and 
                                           self.category_history[category][-1] > self.category_history[category][-2]
                                else "shrinking"
                    for category in self.categories
                }
            },
            "tax_incentives": {
                "category_incentives": category_incentives,
                "general_consumption_tax": f"{current_policies.get('general_consumption_tax', 0.0) * 100:.1f}%",
                "special_promotions": current_policies.get("special_promotions", "None"),
                "policy_description": current_policies.get("policy_description", "No active consumption incentives")
            },
            "personal_consumption": {
                "your_total_spending": total_consumption,
                "your_category_breakdown": category_breakdown,
                "your_savings_rate": f"{savings_rate * 100:.1f}%",
                "estimated_tax_savings": sum(
                    amount * current_policies.get(f"{category}_tax_reduction", 0.0)
                    for category, amount in category_breakdown.items()
                )
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent consumption decisions and update consumption patterns"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policies = self.consumption_tax_policies.get(str(current_epoch), 
                                                          self.consumption_tax_policies.get("0", {}))
        
        # Get policy parameters
        general_consumption_tax = current_policies.get("general_consumption_tax", 0.0)
        category_incentives = {
            category: current_policies.get(f"{category}_tax_reduction", 0.0)
            for category in self.categories
        }
        
        # Get economic indicators from blackboard if available
        economic_gdp = self._get_from_blackboard("economic_gdp", None)
        employment_rate = self._get_from_blackboard("employment_rate", None)
        social_sentiment = self._get_from_blackboard("social_sentiment", 0.0)
        
        # Economic feedback effect on consumption - 更平滑的GDP对消费的影响
        economic_consumption_effect = 0
        if economic_gdp is not None and len(self.consumption_history) > 1:
            prev_gdp = self._get_from_blackboard("previous_gdp_consumption", economic_gdp)
            if prev_gdp and prev_gdp > 0:
                gdp_growth = (economic_gdp - prev_gdp) / max(prev_gdp, 1)
                # 限制GDP增长率的影响范围
                gdp_growth = max(min(gdp_growth, 0.1), -0.1)
                economic_consumption_effect = gdp_growth * 0.5  # 适度增强GDP对消费的影响
            self._post_to_blackboard("previous_gdp_consumption", economic_gdp)
        
        # Employment effect on consumption - 就业率对消费的影响
        employment_effect = 0
        if employment_rate is not None:
            # Higher employment increases consumption
            employment_effect = (employment_rate - 0.9) * 0.5  # 适度增强就业率的影响
        
        # Social sentiment effect on consumption - 添加社会舆论对消费的影响
        sentiment_effect = 0
        if social_sentiment is not None:
            sentiment_effect = social_sentiment * 0.1  # 社会舆论积极时轻微提升消费
        
        # Reset category totals for this epoch
        new_total_consumption = 0
        new_category_consumption = {category: 0 for category in self.categories}
        
        # Process each agent's decision
        for agent_id, decisions in agent_decisions.items():
            if "ConsumptionBehaviorSystem" not in decisions:
                continue
                
            decision = decisions["ConsumptionBehaviorSystem"]
            consumption_level = float(decision.get("consumption_level", 0.5))
            category_preferences = decision.get("category_preferences", {})
            
            # Get current consumption and disposable income
            current_total = self.agent_consumption.get(agent_id, 0)
            current_categories = self.agent_consumption_categories.get(agent_id, {})
            disposable_income = self.agent_disposable_income.get(agent_id, 0)
            
            # 获取最新收入（如果可用）
            economic_income = self._get_from_blackboard(f"income_{agent_id}", None)
            if economic_income:
                # 更新可支配收入 - 如果经济系统提供了收入信息
                disposable_income = economic_income * 0.8  # 假设80%的收入可支配
                self.agent_disposable_income[agent_id] = disposable_income
            else:
                # 基于经济状况的收入调整
                disposable_income = max(disposable_income * (1.0 + economic_consumption_effect * 0.5), 
                                      disposable_income * 0.95)  # 防止收入大幅下降
            
            # 确保消费不会过度波动 - 设置消费变化的上下限
            base_consumption_change = (consumption_level - 0.5) * 0.3  # 增强消费意愿的影响
            
            # 添加经济、就业和社会舆论因素的影响
            total_effect = base_consumption_change + economic_consumption_effect + employment_effect + sentiment_effect
            
            # 限制总体变化率在合理范围内
            total_effect = max(min(total_effect, 0.1), -0.08)  # 允许消费增长比下降更快
            
            # 计算新的消费总额
            new_total = current_total * (1.0 + total_effect)
            
            # 应用消费税减免的影响 - 增强税收政策的效果
            tax_incentive_multiplier = 1.0
            if general_consumption_tax > 0:
                # 消费税基础减免效应
                tax_incentive_multiplier -= general_consumption_tax * 0.5
            
            # 计算类别特定的税收优惠总效应
            category_tax_effect = sum(category_incentives.values()) / len(self.categories)
            
            # 税收优惠的消费刺激效应 - 增强税收优惠对消费的积极影响
            tax_stimulus = 1.0 + category_tax_effect * 2.0
            
            # 应用税收效应
            new_total *= tax_stimulus
            
            # 特殊促销活动的影响
            if "special_promotions" in current_policies and current_policies["special_promotions"] != "None":
                new_total *= 1.05  # 促销活动额外提升5%消费
            
            # 确保消费总额不低于基本生活水平且不超过可支配收入
            new_total = max(min(new_total, disposable_income * 0.9), disposable_income * 0.3)
            
            # 分配到各个消费类别
            if not category_preferences:
                # Default preferences if not specified
                category_preferences = {
                    "essentials": 0.4,
                    "leisure": 0.3,
                    "durables": 0.2,
                    "luxury": 0.1
                }
            
            # Normalize preferences
            total_pref = sum(category_preferences.values())
            if total_pref > 0:
                category_preferences = {k: v / total_pref for k, v in category_preferences.items()}
            
            # Apply tax incentives to category preferences - 增强税收优惠对类别选择的影响
            adjusted_preferences = {}
            for category, preference in category_preferences.items():
                # Increase preference for categories with tax incentives
                tax_boost = category_incentives.get(category, 0.0) * 3.0  # 增强税收优惠的效果
                adjusted_preferences[category] = preference * (1.0 + tax_boost)
            
            # Re-normalize adjusted preferences
            total_adjusted = sum(adjusted_preferences.values())
            if total_adjusted > 0:
                adjusted_preferences = {k: v / total_adjusted for k, v in adjusted_preferences.items()}
            
            # Calculate new category spending
            new_categories = {
                category: new_total * pref
                for category, pref in adjusted_preferences.items()
            }
            
            # Update agent's consumption data
            self.agent_consumption[agent_id] = new_total
            self.agent_consumption_categories[agent_id] = new_categories
            
            # Add to category totals
            new_total_consumption += new_total
            for category, amount in new_categories.items():
                if category in new_category_consumption:
                    new_category_consumption[category] += amount
        
        # Update system totals
        self.total_consumption = new_total_consumption
        self.category_consumption = new_category_consumption
        
        # Update history
        self.consumption_history.append(self.total_consumption)
        for category in self.categories:
            self.category_history[category].append(self.category_consumption.get(category, 0))
        
        self.logger.info(f"Epoch {current_epoch}: Total Consumption={self.total_consumption:.0f}")
        for category in self.categories:
            self.logger.info(f"  {category.capitalize()} Consumption: {self.category_consumption.get(category, 0):.0f}")
        
        # Post key metrics to blackboard for other subsystems
        try:
            self._post_to_blackboard("total_consumption", self.total_consumption)
            for category in self.categories:
                self._post_to_blackboard(f"consumption_{category}", self.category_consumption[category])
        except Exception as e:
            # Log the error but continue execution
            self.logger.debug(f"Could not post consumption data to blackboard: {e}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the effectiveness of tax incentives on consumption patterns"""
        # Calculate key metrics
        consumption_growth = ((self.consumption_history[-1] - self.consumption_history[0]) / self.consumption_history[0]) * 100
        
        # Calculate category growth rates
        category_growth = {}
        for category in self.categories:
            if self.category_history[category][0] > 0:
                growth = ((self.category_history[category][-1] - self.category_history[category][0]) / 
                         self.category_history[category][0]) * 100
                category_growth[category] = growth
            else:
                category_growth[category] = 0
        
        # Calculate consumption volatility
        consumption_volatility = np.std(self.consumption_history) / np.mean(self.consumption_history) * 100
        
        # Calculate category share changes
        initial_shares = {}
        final_shares = {}
        for category in self.categories:
            initial_shares[category] = self.category_history[category][0] / max(self.consumption_history[0], 1) * 100
            final_shares[category] = self.category_history[category][-1] / max(self.consumption_history[-1], 1) * 100
        
        # Calculate consumption inequality (simplified Gini)
        consumption_values = list(self.agent_consumption.values())
        consumption_gini = self._calculate_gini(consumption_values)
        
        # Calculate policy effectiveness on targeted categories
        policy_effectiveness = {}
        for epoch, policies in self.consumption_tax_policies.items():
            if epoch.isdigit() and int(epoch) < len(self.consumption_history) - 1:
                epoch_num = int(epoch)
                effectiveness = {}
                
                for category in self.categories:
                    incentive = policies.get(f"{category}_tax_reduction", 0.0)
                    if incentive > 0:
                        # Calculate growth after policy implementation
                        before = self.category_history[category][epoch_num]
                        after = self.category_history[category][epoch_num + 1]
                        if before > 0:
                            growth = (after - before) / before * 100
                            effectiveness[category] = growth / (incentive * 100) if incentive > 0 else 0
                
                policy_effectiveness[f"epoch_{epoch}"] = effectiveness
        
        evaluation_results = {
            "consumption_metrics": {
                "total_consumption_growth_percent": consumption_growth,
                "consumption_volatility": consumption_volatility,
                "consumption_gini_coefficient": consumption_gini
            },
            "category_metrics": {
                "category_growth_percent": category_growth,
                "initial_category_shares": initial_shares,
                "final_category_shares": final_shares,
                "share_changes": {cat: final_shares[cat] - initial_shares[cat] for cat in self.categories}
            },
            "policy_effectiveness": {
                "policy_elasticity": policy_effectiveness,
                "overall_consumption_stimulus": consumption_growth / 10  # Assuming 10% tax reduction baseline
            },
            "time_series": {
                "consumption_history": self.consumption_history,
                "category_history": self.category_history
            }
        }
        
        self.logger.info(f"Consumption Evaluation Results: {evaluation_results}")
        
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        return {
            "total_consumption": self.total_consumption,
            "category_consumption": self.category_consumption,
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        }
        
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient as a measure of inequality"""
        if not values or len(values) <= 1:
            return 0.0
            
        values = sorted(values)
        n = len(values)
        cumsum = 0
        for i, x in enumerate(values):
            cumsum += (n - i) * x
        return (n + 1 - 2 * cumsum / (n * sum(values))) / n if sum(values) > 0 else 0 