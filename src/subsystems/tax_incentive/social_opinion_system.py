import random
import numpy as np
from typing import Dict, Any, List, Optional
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class SocialOpinionSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Optional[Any] = None, **kwargs):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        
        # Opinion parameters
        self.base_approval_rate = config.get("base_approval_rate", 0.5)
        self.opinion_volatility = config.get("opinion_volatility", 0.05)
        
        # Policy announcements by epoch
        self.policy_announcements = config.get("policy_announcements", {})
        
        # Agent opinion tracking
        self.agent_opinions = {}  # agent_id -> opinion score (-1 to 1)
        self.agent_influence = {}  # agent_id -> influence score
        
        # Social media posts
        self.posts = []  # List of {epoch, agent_id, content, sentiment, influence}
        self.post_count_by_epoch = {}
        self.sentiment_by_epoch = {}
        
        # Opinion statistics
        self.overall_sentiment = self.base_approval_rate
        self.sentiment_history = [self.base_approval_rate]
        self.post_volume_history = [0]
        
        self.logger.info(f"SocialOpinionSystem initialized with base approval rate: {self.base_approval_rate}")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent opinions based on their attributes"""
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            
            # Initialize opinions based on economic status and personality
            income_level = agent_data.get("economic_attributes", {}).get("income_level", "")
            personality = agent_data.get("psychological_attributes", {}).get("personality_traits", [])
            
            # Base opinion slightly biased by income level
            if "high" in income_level.lower() or "wealthy" in income_level.lower():
                base_opinion = random.uniform(0.3, 0.8)  # Higher income tends to favor tax incentives
            elif "mid" in income_level.lower():
                base_opinion = random.uniform(0.2, 0.7)  # Middle income somewhat favorable
            else:
                base_opinion = random.uniform(-0.2, 0.5)  # Lower income more skeptical
                
            # Adjust based on personality traits
            if any(trait.lower() in ["conservative", "traditional"] for trait in personality):
                base_opinion += random.uniform(0.1, 0.2)  # More favorable to tax cuts
            if any(trait.lower() in ["progressive", "liberal"] for trait in personality):
                base_opinion -= random.uniform(0.1, 0.2)  # Less favorable to tax cuts
                
            # Clamp to valid range
            base_opinion = max(-1.0, min(1.0, base_opinion))
            
            # Set initial opinion and influence
            self.agent_opinions[agent_id] = base_opinion
            
            # Determine social influence based on social attributes
            social_influence = agent_data.get("social_attributes", {}).get("social_influence", "")
            if "leader" in social_influence.lower() or "influencer" in social_influence.lower():
                influence = random.uniform(0.7, 1.0)
            elif "independent" in social_influence.lower():
                influence = random.uniform(0.4, 0.7)
            else:
                influence = random.uniform(0.1, 0.4)
                
            self.agent_influence[agent_id] = influence
        
        # Calculate initial overall sentiment
        self.overall_sentiment = self._calculate_weighted_sentiment()
        self.sentiment_history[0] = self.overall_sentiment
        
        self.logger.info(f"Initialized {len(self.agent_opinions)} agents with opinions")
        self.logger.info(f"Initial overall sentiment: {self.overall_sentiment:.2f}")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide social opinion information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_announcement = self.policy_announcements.get(str(current_epoch), 
                                                           self.policy_announcements.get("0", {}))
        
        # Get recent posts (last 10 or fewer)
        recent_posts = self.posts[-10:] if len(self.posts) > 10 else self.posts
        
        # Calculate sentiment trend
        sentiment_trend = "stable"
        if len(self.sentiment_history) > 1:
            recent_change = self.sentiment_history[-1] - self.sentiment_history[-2]
            if recent_change > 0.05:
                sentiment_trend = "improving"
            elif recent_change < -0.05:
                sentiment_trend = "worsening"
        
        # Get agent's own opinion
        agent_opinion = self.agent_opinions.get(agent_id, 0)
        agent_influence = self.agent_influence.get(agent_id, 0)
        
        # Calculate how agent's opinion compares to overall sentiment
        opinion_alignment = "neutral"
        opinion_diff = agent_opinion - self.overall_sentiment
        if opinion_diff > 0.2:
            opinion_alignment = "more positive than average"
        elif opinion_diff < -0.2:
            opinion_alignment = "more negative than average"
        
        return {
            "public_opinion": {
                "overall_sentiment": self.overall_sentiment,
                "sentiment_trend": sentiment_trend,
                "post_volume": len(self.posts),
                "post_volume_trend": "increasing" if self.post_volume_history[-1] > (self.post_volume_history[-2] if len(self.post_volume_history) > 1 else 0) else "decreasing"
            },
            "policy_announcement": {
                "current_announcement": current_announcement.get("text", "No active policy announcement"),
                "announcement_date": current_announcement.get("date", "N/A"),
                "announcement_source": current_announcement.get("source", "Government")
            },
            "social_media": {
                "recent_posts": [{"content": p["content"], "sentiment": p["sentiment"]} for p in recent_posts],
                "trending_topics": self._get_trending_topics()
            },
            "personal_opinion": {
                "your_sentiment": agent_opinion,
                "your_social_influence": agent_influence,
                "opinion_alignment": opinion_alignment
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent social media decisions and update opinion state"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_announcement = self.policy_announcements.get(str(current_epoch), 
                                                           self.policy_announcements.get("0", {}))
        
        # Reset post count for this epoch
        self.post_count_by_epoch[current_epoch] = 0
        self.sentiment_by_epoch[current_epoch] = []
        
        # Get economic indicators from blackboard if available - add error handling
        try:
            economic_consumption = self._get_from_blackboard("economic_consumption", None)
            economic_investment = self._get_from_blackboard("economic_investment", None)
            economic_gdp = self._get_from_blackboard("economic_gdp", None)
            employment_rate = self._get_from_blackboard("employment_rate", 0.95)
            productivity_index = self._get_from_blackboard("productivity_index", 1.0)
        except Exception as e:
            # Log the error and use default values
            self.logger.debug(f"Could not retrieve data from blackboard: {e}")
            economic_consumption = None
            economic_investment = None
            economic_gdp = None
            employment_rate = 0.95
            productivity_index = 1.0
        
        # Economic feedback effect on opinions - 更合理的经济对舆论的影响
        economic_sentiment_effect = 0
        gdp_growth = 0
        if economic_gdp is not None and len(self.sentiment_history) > 1:
            # 计算经济增长率
            prev_gdp = self._get_from_blackboard("previous_gdp", economic_gdp)
            if prev_gdp and prev_gdp > 0:
                gdp_growth = (economic_gdp - prev_gdp) / max(prev_gdp, 1)
                # 限制GDP增长率的影响范围
                gdp_growth = max(min(gdp_growth, 0.1), -0.1)
                # 经济增长提升舆论，下滑降低舆论
                economic_sentiment_effect = gdp_growth * 1.0  # 降低GDP波动对舆论的过度影响
            self._post_to_blackboard("previous_gdp", economic_gdp)
        
        # 就业率对舆论的影响
        employment_sentiment_effect = 0
        if employment_rate is not None:
            # 高就业率提升舆论
            employment_sentiment_effect = (employment_rate - 0.9) * 0.3
        
        # 生产力对舆论的影响
        productivity_sentiment_effect = 0
        if productivity_index is not None:
            # 生产力提高改善舆论
            productivity_sentiment_effect = (productivity_index - 1.0) * 0.2
        
        # 政策公告的影响
        announcement_effect = 0
        if current_announcement and "text" in current_announcement:
            # 分析政策公告的积极程度
            text = current_announcement["text"].lower()
            
            # 公告中的积极因素
            positive_terms = ["benefit", "incentive", "reduction", "boost", "growth", "stimulus"]
            positive_count = sum(1 for term in positive_terms if term in text)
            
            # 公告中的消极因素
            negative_terms = ["increase tax", "higher tax", "cut", "reduce", "limitation"]
            negative_count = sum(1 for term in negative_terms if term in text)
            
            # 税收优惠力度对公告影响的放大
            tax_benefit_magnitude = 0
            if "tax" in text and "%" in text:
                # 尝试提取税收优惠百分比
                try:
                    # 简单的百分比提取
                    import re
                    percentages = re.findall(r'(\d+)%', text)
                    if percentages:
                        # 取最大的百分比作为影响因子
                        tax_benefit_magnitude = max([int(p) for p in percentages]) / 100
                except:
                    tax_benefit_magnitude = 0.05  # 默认影响
            
            # 综合影响，积极项多且税收优惠大则公告影响更积极
            announcement_effect = (positive_count - negative_count) * 0.03
            announcement_effect += tax_benefit_magnitude * 0.5
        
        # Process each agent's decision
        for agent_id, decisions in agent_decisions.items():
            if "SocialOpinionSystem" not in decisions:
                continue
                
            decision = decisions["SocialOpinionSystem"]
            post_intention = float(decision.get("post_intention", 0.3))
            post_sentiment = float(decision.get("post_sentiment", 0))
            post_content = decision.get("post_content", "")
            
            # Update agent's opinion based on economic conditions and social influence
            current_opinion = self.agent_opinions.get(agent_id, 0)
            
            # Opinion evolves based on:
            # 1. Economic conditions
            # 2. Social influence (other agents' opinions)
            # 3. Policy announcements
            # 4. Random fluctuation
            
            # 经济影响 - 经济向好提升舆论，向差降低舆论
            economic_effect = economic_sentiment_effect + employment_sentiment_effect + productivity_sentiment_effect
            # 限制经济因素的总体影响
            economic_effect = max(min(economic_effect, 0.1), -0.1)
            
            # 社会影响效应 - 靠近整体舆论
            social_effect = (self.overall_sentiment - current_opinion) * 0.15
            
            # 公告效应
            announcement_impact = announcement_effect
            
            # 随机波动 - 减小随机性
            random_effect = random.uniform(-self.opinion_volatility, self.opinion_volatility)
            
            # 计算最终舆论变化
            opinion_shift = economic_effect + social_effect + announcement_impact + random_effect
            
            # 更新舆论
            new_opinion = current_opinion + opinion_shift
            new_opinion = max(-1.0, min(1.0, new_opinion))
            self.agent_opinions[agent_id] = new_opinion
            
            # Process social media posts
            if random.random() < post_intention:
                # Agent makes a post
                if not post_content:
                    # Generate post content if not provided
                    post_content = self._generate_post_content(post_sentiment, current_announcement)
                
                # Adjust sentiment based on current overall sentiment (calibration)
                calibrated_sentiment = post_sentiment * 0.7 + new_opinion * 0.3
                calibrated_sentiment = max(-1.0, min(1.0, calibrated_sentiment))
                
                # Create post
                post = {
                    "epoch": current_epoch,
                    "agent_id": agent_id,
                    "content": post_content,
                    "sentiment": calibrated_sentiment,
                    "influence": self.agent_influence.get(agent_id, 0.1)
                }
                
                self.posts.append(post)
                self.post_count_by_epoch[current_epoch] = self.post_count_by_epoch.get(current_epoch, 0) + 1
                self.sentiment_by_epoch[current_epoch].append(calibrated_sentiment)
        
        # Calculate new overall sentiment with weighted average based on influence
        # Only consider the posts from this epoch for immediate impact
        epoch_sentiment = 0
        total_influence = 0
        
        for sentiment in self.sentiment_by_epoch.get(current_epoch, []):
            epoch_sentiment += sentiment
            total_influence += 1
        
        if total_influence > 0:
            epoch_sentiment = epoch_sentiment / total_influence
            
            # Update overall sentiment with smoothing
            prev_sentiment = self.overall_sentiment
            # 提高当前舆论的权重以反映更快的舆论反应
            self.overall_sentiment = prev_sentiment * 0.6 + epoch_sentiment * 0.4
        
        # Record history
        self.sentiment_history.append(self.overall_sentiment)
        self.post_volume_history.append(len(self.sentiment_by_epoch.get(current_epoch, [])))
        
        self.logger.info(f"Epoch {current_epoch}: Overall Sentiment={self.overall_sentiment:.2f}, "
                        f"Posts={self.post_count_by_epoch.get(current_epoch, 0)}")
        
        # Post key metrics to blackboard for other subsystems - add error handling
        try:
            self._post_to_blackboard("social_sentiment", self.overall_sentiment)
            self._post_to_blackboard("social_post_volume", self.post_count_by_epoch.get(current_epoch, 0))
        except Exception as e:
            # Log the error but continue execution
            self.logger.debug(f"Could not post social opinion data to blackboard: {e}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the effectiveness of policy communication and public opinion"""
        # Calculate key metrics
        sentiment_change = self.sentiment_history[-1] - self.sentiment_history[0]
        sentiment_volatility = np.std(self.sentiment_history)
        
        # Analyze post sentiment distribution
        sentiment_distribution = {
            "very_negative": sum(1 for p in self.posts if p["sentiment"] < -0.6),
            "negative": sum(1 for p in self.posts if -0.6 <= p["sentiment"] < -0.2),
            "neutral": sum(1 for p in self.posts if -0.2 <= p["sentiment"] < 0.2),
            "positive": sum(1 for p in self.posts if 0.2 <= p["sentiment"] < 0.6),
            "very_positive": sum(1 for p in self.posts if p["sentiment"] >= 0.6)
        }
        
        # Calculate opinion polarization
        opinions = list(self.agent_opinions.values())
        opinion_polarization = np.std(opinions) if opinions else 0
        
        # Calculate correlation between income level and opinion
        # (simplified - would need agent data here)
        
        # Calculate communication effectiveness
        communication_effectiveness = 0
        for epoch, announcement in self.policy_announcements.items():
            if epoch.isdigit() and int(epoch) < len(self.sentiment_history) - 1:
                epoch_num = int(epoch)
                sentiment_before = self.sentiment_history[epoch_num]
                sentiment_after = self.sentiment_history[epoch_num + 1]
                if sentiment_after > sentiment_before:
                    communication_effectiveness += 1
        
        communication_effectiveness /= max(len(self.policy_announcements), 1)
        
        evaluation_results = {
            "sentiment_metrics": {
                "final_sentiment": self.sentiment_history[-1],
                "sentiment_change": sentiment_change,
                "sentiment_volatility": sentiment_volatility
            },
            "engagement_metrics": {
                "total_posts": len(self.posts),
                "posts_per_epoch": {epoch: count for epoch, count in self.post_count_by_epoch.items()},
                "sentiment_distribution": sentiment_distribution
            },
            "opinion_metrics": {
                "opinion_polarization": opinion_polarization,
                "opinion_distribution": self._calculate_opinion_distribution()
            },
            "communication_metrics": {
                "communication_effectiveness": communication_effectiveness,
                "announcement_impact": self._calculate_announcement_impact()
            },
            "time_series": {
                "sentiment_history": self.sentiment_history,
                "post_volume_history": self.post_volume_history
            }
        }
        
        self.logger.info(f"Social Opinion Evaluation Results: {evaluation_results}")
        
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        return {
            "overall_sentiment": self.overall_sentiment,
            "post_count": self.post_count_by_epoch.get(current_epoch, 0),
            "total_posts": len(self.posts),
            "current_epoch": current_epoch
        }
        
    def _calculate_weighted_sentiment(self) -> float:
        """Calculate overall sentiment weighted by agent influence"""
        if not self.agent_opinions:
            return self.base_approval_rate
            
        total_weight = 0
        weighted_sum = 0
        
        for agent_id, opinion in self.agent_opinions.items():
            influence = self.agent_influence.get(agent_id, 0.1)
            weighted_sum += opinion * influence
            total_weight += influence
            
        return weighted_sum / total_weight if total_weight > 0 else self.base_approval_rate
        
    def _get_trending_topics(self) -> List[str]:
        """Generate trending topics based on current sentiment and posts"""
        topics = []
        
        if self.overall_sentiment > 0.5:
            topics.append("#TaxCutsWork")
            topics.append("#EconomicGrowth")
        elif self.overall_sentiment < -0.5:
            topics.append("#TaxScam")
            topics.append("#InequalityRising")
        else:
            topics.append("#TaxPolicy")
            topics.append("#EconomyWatch")
            
        if self.post_volume_history[-1] > 100:
            topics.append("#ViralTaxDebate")
            
        return topics
        
    def _generate_post_content(self, sentiment: float, announcement: Dict[str, Any]) -> str:
        """Generate a post content based on sentiment and current announcement"""
        if sentiment > 0.5:
            return "These tax incentives are exactly what our economy needs! #EconomicGrowth"
        elif sentiment > 0:
            return "The new tax policy seems promising, hoping it delivers the promised benefits."
        elif sentiment > -0.5:
            return "Not sure these tax cuts will help everyone. What about public services?"
        else:
            return "These tax cuts only benefit the wealthy. What about the rest of us? #Inequality"
            
    def _calculate_opinion_distribution(self) -> Dict[str, int]:
        """Calculate distribution of opinions across sentiment ranges"""
        distribution = {
            "very_negative": 0,
            "negative": 0,
            "neutral": 0,
            "positive": 0,
            "very_positive": 0
        }
        
        for opinion in self.agent_opinions.values():
            if opinion < -0.6:
                distribution["very_negative"] += 1
            elif opinion < -0.2:
                distribution["negative"] += 1
            elif opinion < 0.2:
                distribution["neutral"] += 1
            elif opinion < 0.6:
                distribution["positive"] += 1
            else:
                distribution["very_positive"] += 1
                
        return distribution
        
    def _calculate_announcement_impact(self) -> Dict[str, float]:
        """Calculate the impact of each announcement on sentiment"""
        impact = {}
        
        for epoch_str, announcement in self.policy_announcements.items():
            if epoch_str.isdigit():
                epoch = int(epoch_str)
                if epoch < len(self.sentiment_history) - 1:
                    before = self.sentiment_history[epoch]
                    after = self.sentiment_history[epoch + 1]
                    impact[f"announcement_{epoch}"] = after - before
                    
        return impact 