from typing import Dict, Any, List
import random
from collections import defaultdict, Counter

from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger


class PublicOpinionSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Any = None):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        
        # Policy announcements by epoch
        self.policy_announcements = config.get("policy_announcements", {})
        
        # Opinion tracking
        self.opinions_by_epoch = defaultdict(list)
        self.sentiment_distribution = defaultdict(lambda: Counter())
        self.opinion_topics = defaultdict(lambda: Counter())
        
        # Agent attributes for opinion formation
        self.agent_attributes = {}
        
        # Public sentiment metrics
        self.public_sentiment_score = 0.5  # 0-1, 0.5 is neutral
        self.sentiment_volatility = 0.0
        
        # Sentiment tracking by health code color
        self.sentiment_by_health_code = defaultdict(lambda: defaultdict(lambda: Counter()))
        
    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize with agent data"""
        self.logger.info(f"Initializing {self.name} with {len(all_agent_data)} agents")
        
        # Store relevant agent attributes
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            self.agent_attributes[agent_id] = {
                "age": agent_data.get("basic_info", {}).get("age", 30),
                "education_level": agent_data.get("basic_info", {}).get("education_level", "high_school"),
                "residence_type": agent_data.get("basic_info", {}).get("residence_type", "urban")
            }
        
        # Initialize system state
        self.system_state = {
            "total_agents": len(all_agent_data),
            "current_policy": "Normal period",
            "public_sentiment_score": self.public_sentiment_score
        }
        
    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent decisions about expressing opinions"""
        epoch = self.current_time.get_current_epoch() if self.current_time else 0
        self.logger.info(f"Processing opinions for epoch {epoch}")
        
        # Get current policy announcement
        current_policy = self.policy_announcements.get(str(epoch), "No announcement")
        self.system_state["current_policy"] = current_policy
        
        # Get current health code distribution and restriction level
        health_code_distribution = self._get_from_blackboard("health_code_distribution", {})
        agent_health_codes = self._get_from_blackboard("agent_health_codes", {})
        restriction_level = self._get_from_blackboard("restriction_level", 0)
        blocked_mobility_rate = self._get_from_blackboard("blocked_mobility_rate", 0)
        
        # Process each agent's opinion decision
        epoch_opinions = []
        epoch_sentiments = Counter()
        
        # Track sentiment by health code color
        sentiments_by_code = defaultdict(lambda: Counter())
        
        for agent_id, decisions in agent_decisions.items():
            opinion_decision = decisions.get(self.name, {})
            
            if opinion_decision.get("express_opinion") == "yes":
                opinion_content = opinion_decision.get("opinion_content", "")
                sentiment = opinion_decision.get("opinion_sentiment", "neutral")
                
                # Get agent's health code color
                health_code = agent_health_codes.get(agent_id, "green").lower()
                
                # Balance sentiment distribution - adjust extremely positive sentiments
                # People with red/yellow codes or those blocked from activities are more likely to have negative opinions
                if health_code in ["red", "yellow"] and sentiment == "positive" and random.random() < 0.7:
                    # 70% chance to downgrade sentiment for restricted individuals
                    sentiment = "neutral"
                    self.logger.debug(f"Adjusted sentiment for restricted agent {agent_id} from positive to neutral")
                
                # Track sentiment by health code
                sentiments_by_code[health_code][sentiment] += 1
                
                opinion_entry = {
                    "agent_id": agent_id,
                    "content": opinion_content,
                    "sentiment": sentiment,
                    "health_code": health_code,
                    "agent_attrs": self.agent_attributes.get(agent_id, {})
                }
                
                epoch_opinions.append(opinion_entry)
                epoch_sentiments[sentiment] += 1
                
                # Extract topics from opinion content
                self._extract_topics(opinion_content, epoch)
        
        # Store opinions
        self.opinions_by_epoch[epoch] = epoch_opinions
        self.sentiment_distribution[epoch] = epoch_sentiments
        self.sentiment_by_health_code[epoch] = dict(sentiments_by_code)
        
        # Calculate public sentiment metrics with policy and mobility impact
        self._update_public_sentiment(epoch_sentiments, restriction_level, blocked_mobility_rate)
        
        # Post to blackboard
        self._post_to_blackboard("public_sentiment_score", self.public_sentiment_score)
        self._post_to_blackboard("current_policy_announcement", current_policy)
        self._post_to_blackboard("sentiment_distribution", dict(epoch_sentiments))
        
        # Log statistics
        total_opinions = len(epoch_opinions)
        self.logger.info(f"Epoch {epoch}: {total_opinions} opinions expressed")
        self.logger.info(f"Sentiment distribution: {dict(epoch_sentiments)}")
        self.logger.info(f"Public sentiment score: {self.public_sentiment_score:.3f}")
        
    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Return environment information for a specific agent"""
        epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Get agent's health code color
        agent_health_codes = self._get_from_blackboard("agent_health_codes", {})
        agent_health_code = agent_health_codes.get(agent_id, "green").lower()
        
        # Get recent opinions from similar agents (same residence type and health code)
        agent_attrs = self.agent_attributes.get(agent_id, {})
        residence_type = agent_attrs.get("residence_type", "urban")
        
        recent_opinions = []
        if epoch > 0 and (epoch - 1) in self.opinions_by_epoch:
            for opinion in self.opinions_by_epoch[epoch - 1][-10:]:  # Last 10 opinions
                if opinion["agent_attrs"].get("residence_type") == residence_type:
                    recent_opinions.append({
                        "content": opinion["content"][:100] + "...",  # Truncate
                        "sentiment": opinion["sentiment"],
                        "health_code": opinion.get("health_code", "unknown")
                    })
        
        # Get infection data from blackboard to influence sentiment
        active_infections = self._get_from_blackboard("active_infections", 0)
        total_agents = len(self.agent_attributes)
        infection_rate = active_infections / max(total_agents, 1) if total_agents > 0 else 0
        
        # Get health code distribution
        health_code_distribution = self._get_from_blackboard("health_code_distribution", {})
        
        # Get restriction level
        restriction_level = self._get_from_blackboard("restriction_level", 0)
        
        # Adjust public sentiment based on infection rate and restrictions
        sentiment_description = self._get_sentiment_description(infection_rate, restriction_level, agent_health_code)
        
        # Generate media reports based on current situation
        media_report = self._generate_media_report(epoch, infection_rate, restriction_level)
        
        # Provide specific information based on agent's health code
        health_code_sentiment = {}
        if epoch in self.sentiment_by_health_code:
            for code, sentiments in self.sentiment_by_health_code[epoch].items():
                health_code_sentiment[code] = dict(sentiments)
        
        return {
            "policy_announcements": self.policy_announcements.get(str(epoch), "No announcement"),
            "public_sentiment": {
                "score": self.public_sentiment_score,
                "trend": "improving" if self.sentiment_volatility > 0 else "worsening" if self.sentiment_volatility < 0 else "stable",
                "recent_opinions": recent_opinions[:3],  # Show top 3
                "infection_concern": infection_rate,
                "situation_description": sentiment_description,
                "sentiment_by_health_code": health_code_sentiment
            },
            "media_reports": media_report,
            "your_health_code": agent_health_code,
            "health_code_distribution": health_code_distribution,
            "restriction_level": restriction_level
        }
    
    def _get_sentiment_description(self, infection_rate: float, restriction_level: float, health_code: str) -> str:
        """Generate sentiment description based on infection rate, policy epoch, and health code"""
        # Base description based on infection rate
        if infection_rate < 0.01:
            base_desc = "Public optimistic about health situation"
        elif infection_rate < 0.05:
            base_desc = "Public cautiously optimistic but concerned"
        elif infection_rate < 0.15:
            base_desc = "Growing public anxiety about infection spread"
        else:
            base_desc = "High public concern and anxiety about health crisis"
        
        # Adjust based on restriction level
        if restriction_level > 0.7:
            policy_impact = "Strong dissatisfaction with strict movement controls"
        elif restriction_level > 0.5:
            policy_impact = "Mixed feelings about health code restrictions"
        elif restriction_level > 0.3:
            policy_impact = "General acceptance of moderate health measures"
        else:
            policy_impact = "Satisfaction with minimal restrictions"
        
        # Specific sentiment for different health codes
        if health_code == "red":
            code_impact = "Red code holders express frustration about isolation"
        elif health_code == "yellow":
            code_impact = "Yellow code holders concerned about limited access"
        else:
            code_impact = "Green code holders generally supportive of system"
        
        return f"{base_desc}. {policy_impact}. {code_impact}."
    
    def _extract_topics(self, opinion_content: str, epoch: int):
        """Extract topics from opinion content"""
        # Simple keyword extraction
        keywords = ["freedom", "safety", "privacy", "economy", "health", "control", "quarantine", "work", "family"]
        
        content_lower = opinion_content.lower()
        for keyword in keywords:
            if keyword in content_lower:
                self.opinion_topics[epoch][keyword] += 1
    
    def _update_public_sentiment(self, sentiments: Counter, restriction_level: float, blocked_mobility_rate: float):
        """Update public sentiment score based on current opinions, infection situation, and policy impact"""
        total = sum(sentiments.values())
        if total == 0:
            return
        
        # Get infection data to influence sentiment
        active_infections = self._get_from_blackboard("active_infections", 0)
        total_agents = len(self.agent_attributes)
        infection_rate = active_infections / max(total_agents, 1) if total_agents > 0 else 0
        
        # Calculate weighted sentiment from opinions
        sentiment_weights = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
        weighted_sum = sum(sentiments[s] * sentiment_weights[s] for s in sentiment_weights)
        opinion_sentiment = weighted_sum / total
        
        # Adjust sentiment based on infection rate (more infections -> more negative)
        infection_penalty = min(infection_rate * 5, 0.5)  # Max 50% penalty
        
        # Adjust sentiment based on restriction level (more restrictions -> more negative)
        # This is a key change to make sentiment respond to policy strictness
        restriction_penalty = restriction_level * 0.4  # Max 40% penalty for strictest restrictions
        
        # Adjust sentiment based on blocked mobility (more people blocked -> more negative)
        mobility_penalty = blocked_mobility_rate * 0.3  # Max 30% penalty if everyone is blocked
        
        # Combine all factors
        total_penalty = min(infection_penalty + restriction_penalty + mobility_penalty, 0.8)  # Cap at 80% penalty
        adjusted_sentiment = opinion_sentiment * (1 - total_penalty)
        
        # Calculate volatility
        self.sentiment_volatility = adjusted_sentiment - self.public_sentiment_score
        
        # Update with smoothing
        self.public_sentiment_score = 0.7 * self.public_sentiment_score + 0.3 * adjusted_sentiment
        
    def _generate_media_report(self, epoch: int, infection_rate: float = 0, restriction_level: float = 0) -> str:
        """Generate a media report based on current situation"""
        # Base report on infection rate
        if infection_rate > 0.15:
            infection_report = "Media reports growing health crisis and calls for stronger measures."
        elif infection_rate > 0.05:
            infection_report = "Media coverage focuses on infection spread and policy effectiveness."
        else:
            infection_report = "Media reports stable health situation with occasional cases."
        
        # Add policy impact
        if restriction_level > 0.7:
            policy_report = "Headlines highlight economic impact of strict movement controls."
        elif restriction_level > 0.5:
            policy_report = "News discusses balance between health safety and normal activities."
        elif restriction_level > 0.3:
            policy_report = "Reports show general compliance with moderate health measures."
        else:
            policy_report = "Coverage shows normal daily life with minimal disruptions."
        
        # Add sentiment component
        if self.public_sentiment_score > 0.7:
            sentiment_report = "Public opinion polls show strong support for current approach."
        elif self.public_sentiment_score > 0.5:
            sentiment_report = "Surveys indicate cautious public optimism about health policies."
        elif self.public_sentiment_score > 0.3:
            sentiment_report = "Opinion columns reflect growing concerns about restrictions."
        else:
            sentiment_report = "Social media shows widespread frustration with health code system."
        
        return f"{infection_report} {policy_report} {sentiment_report}"
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the public opinion system"""
        self.logger.info("=== Public Opinion System Evaluation ===")
        
        # Calculate overall metrics
        total_opinions = sum(len(opinions) for opinions in self.opinions_by_epoch.values())
        
        # Sentiment trends over time
        sentiment_trends = defaultdict(dict)
        
        for epoch in sorted(self.opinions_by_epoch.keys()):
            sentiments = self.sentiment_distribution[epoch]
            sentiment_trends[epoch] = dict(sentiments)
        
        # Topic frequency analysis
        all_topics = Counter()
        for topics in self.opinion_topics.values():
            all_topics.update(topics)
        
        # Opinion participation by demographics
        participation_by_education = defaultdict(int)
        sentiment_by_age = defaultdict(lambda: Counter())
        
        for opinions in self.opinions_by_epoch.values():
            for opinion in opinions:
                attrs = opinion["agent_attrs"]
                participation_by_education[attrs.get("education_level", "unknown")] += 1
                
                age = attrs.get("age", 30)
                age_group = "young" if age < 30 else "middle" if age < 60 else "senior"
                sentiment_by_age[age_group][opinion["sentiment"]] += 1
        
        # Policy impact analysis
        policy_response = self._analyze_policy_response()
        
        evaluation_results = {
            "total_opinions_expressed": total_opinions,
            "average_opinions_per_epoch": total_opinions / max(len(self.opinions_by_epoch), 1),
            "final_public_sentiment": self.public_sentiment_score,
            "sentiment_trends": dict(sentiment_trends),
            "top_topics": dict(all_topics.most_common(5)),
            "participation_by_education": dict(participation_by_education),
            "sentiment_by_age_group": {k: dict(v) for k, v in sentiment_by_age.items()},
            "policy_response_analysis": policy_response,
            "sentiment_by_health_code": {epoch: {code: dict(sentiments) for code, sentiments in code_sentiments.items()} 
                                        for epoch, code_sentiments in self.sentiment_by_health_code.items()}
        }
        
        # Log key metrics
        self.logger.info(f"Total opinions expressed: {total_opinions}")
        self.logger.info(f"Final public sentiment score: {self.public_sentiment_score:.3f}")
        self.logger.info(f"Top topics: {dict(all_topics.most_common(3))}")
        
        # Log sentiment trends
        self.logger.info(f"Sentiment trends over epochs:")
        for epoch, sentiments in sentiment_trends.items():
            self.logger.info(f"  Epoch {epoch}: {sentiments}")
        
        return evaluation_results
    
    def _analyze_policy_response(self) -> Dict[str, Any]:
        """Analyze how public opinion responded to policy changes"""
        policy_response = {}
        
        # Track sentiment changes after policy announcements
        for epoch in range(1, self.current_time.get_current_epoch() + 1):
            policy = self.policy_announcements.get(str(epoch), "")
            if not policy:
                continue
                
            # Get sentiment before and after policy
            sentiment_before = self.public_sentiment_score if epoch == 0 else self._get_epoch_sentiment_score(epoch - 1)
            sentiment_after = self._get_epoch_sentiment_score(epoch)
            
            sentiment_change = sentiment_after - sentiment_before
            
            policy_response[f"epoch_{epoch}"] = {
                "policy": policy,
                "sentiment_before": sentiment_before,
                "sentiment_after": sentiment_after,
                "sentiment_change": sentiment_change,
                "reaction": "positive" if sentiment_change > 0.05 else 
                           "negative" if sentiment_change < -0.05 else "neutral"
            }
        
        return policy_response
    
    def _get_epoch_sentiment_score(self, epoch: int) -> float:
        """Calculate sentiment score for a specific epoch"""
        if epoch not in self.sentiment_distribution:
            return 0.5  # Default neutral
            
        sentiments = self.sentiment_distribution[epoch]
        total = sum(sentiments.values())
        
        if total == 0:
            return 0.5
            
        sentiment_weights = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
        weighted_sum = sum(sentiments[s] * sentiment_weights[s] for s in sentiment_weights)
        
        return weighted_sum / total 