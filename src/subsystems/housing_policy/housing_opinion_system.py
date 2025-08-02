import random
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class HousingOpinionSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)
        
        # Opinion tracking
        self.posts = []  # List of all posts
        self.post_id_counter = 0
        self.agent_posts = defaultdict(list)  # agent_id -> list of post_ids
        self.post_likes = defaultdict(int)  # post_id -> like count
        self.post_shares = defaultdict(int)  # post_id -> share count
        
        # Policy announcements
        self.policy_announcements = config.get("policy_announcements", {})
        
        # Sentiment tracking
        self.sentiment_history = []
        self.topic_frequency = defaultdict(int)
        self.topic_frequency_history = []
        
        # Market sentiment influence
        self.current_market_sentiment = "neutral"  # neutral, optimistic, pessimistic
        self.sentiment_influence_factor = config.get("sentiment_influence_factor", 0.5)
        
        self.logger.info("HousingOpinionSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize with some seed posts about housing"""
        seed_posts = [
            {"content": "Housing prices are getting out of control! Young people can't afford homes anymore.", 
             "sentiment": "negative", "topic": "affordability"},
            {"content": "The real estate market needs regulation. Current prices are unsustainable.", 
             "sentiment": "negative", "topic": "regulation"},
            {"content": "Just bought my first home! Dreams do come true with hard work.", 
             "sentiment": "positive", "topic": "success_story"},
            {"content": "Investment properties are still a good bet for long-term wealth building.", 
             "sentiment": "positive", "topic": "investment"},
            {"content": "Renting forever seems to be the only option for our generation.", 
             "sentiment": "negative", "topic": "affordability"}
        ]
        
        for post in seed_posts:
            self._create_post("system", post["content"], post["sentiment"], post["topic"])
        
        self.logger.info(f"Initialized with {len(seed_posts)} seed posts")

    def _create_post(self, author_id: str, content: str, sentiment: str = "neutral", topic: str = "general"):
        """Create a new post"""
        post = {
            "id": f"post_{self.post_id_counter}",
            "author": author_id,
            "content": content,
            "sentiment": sentiment,
            "topic": topic,
            "epoch": self.current_time.get_current_epoch() if self.current_time else 0,
            "likes": 0,
            "shares": 0
        }
        
        self.posts.append(post)
        self.agent_posts[author_id].append(post["id"])
        self.post_id_counter += 1
        
        # Track topic frequency (overall)
        self.topic_frequency[topic] += 1
        
        return post["id"], topic

    def _analyze_sentiment(self, content: str) -> str:
        """Simple sentiment analysis based on keywords"""
        negative_keywords = ["expensive", "unaffordable", "crisis", "bubble", "can't afford", 
                           "impossible", "unfair", "regulation", "restriction", "overpriced"]
        positive_keywords = ["opportunity", "investment", "bought", "happy", "affordable", 
                           "success", "dream", "stable", "growth", "potential"]
        
        content_lower = content.lower()
        neg_count = sum(1 for keyword in negative_keywords if keyword in content_lower)
        pos_count = sum(1 for keyword in positive_keywords if keyword in content_lower)
        
        if neg_count > pos_count:
            return "negative"
        elif pos_count > neg_count:
            return "positive"
        else:
            return "neutral"

    def _extract_topic(self, content: str) -> str:
        """Extract main topic from post content"""
        topic_keywords = {
            "policy": ["policy", "government", "regulation", "restriction", "limit"],
            "affordability": ["afford", "expensive", "price", "cost", "salary"],
            "investment": ["investment", "profit", "return", "wealth", "asset"],
            "market": ["market", "bubble", "trend", "supply", "demand"]
        }
        
        content_lower = content.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return topic
        
        return "general"

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide trending topics and posts to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Check for policy announcements
        policy_news = self.policy_announcements.get(str(current_epoch), "")
        
        # Get trending posts (most liked/shared)
        trending_posts = sorted(self.posts[-20:], 
                              key=lambda p: self.post_likes[p["id"]] + self.post_shares[p["id"]], 
                              reverse=True)[:5]
        
        # Calculate market sentiment
        recent_posts = self.posts[-50:] if len(self.posts) > 50 else self.posts
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for post in recent_posts:
            sentiment_counts[post["sentiment"]] += 1
        
        total_recent = len(recent_posts)
        if total_recent > 0:
            if sentiment_counts["negative"] / total_recent > 0.5:
                self.current_market_sentiment = "pessimistic"
            elif sentiment_counts["positive"] / total_recent > 0.4:
                self.current_market_sentiment = "optimistic"
            else:
                self.current_market_sentiment = "neutral"
        
        return {
            "trending_topics": list(self.topic_frequency.keys())[:3],
            "policy_news": policy_news,
            "market_sentiment": self.current_market_sentiment,
            "recommended_posts": [
                {
                    "id": post["id"],
                    "content": post["content"],
                    "likes": self.post_likes[post["id"]],
                    "shares": self.post_shares[post["id"]]
                }
                for post in trending_posts
            ]
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent social media actions"""
        epoch_sentiment = {"positive": 0, "negative": 0, "neutral": 0}
        epoch_topic_frequency = defaultdict(int)
        
        for agent_id, decisions in agent_decisions.items():
            if self.name not in decisions:
                continue
            
            housing_decision = decisions.get(self.name, {})
            actions = housing_decision.get("social_actions", [])
            
            for action in actions:
                # Handle case where action might be a string instead of dict
                if isinstance(action, str):
                    continue
                    
                # Ensure action is a dictionary before calling .get()
                if not isinstance(action, dict):
                    continue
                    
                action_type = action.get("action")
                
                if action_type == "post":
                    content = action.get("content", "")
                    if content:
                        sentiment = self._analyze_sentiment(content)
                        extracted_topic = self._extract_topic(content)
                        post_id, post_topic = self._create_post(agent_id, content, sentiment, extracted_topic)
                        epoch_sentiment[sentiment] += 1
                        epoch_topic_frequency[post_topic] += 1
                        
                        self.logger.debug(f"Agent {agent_id} posted: '{content[:50]}...' "
                                        f"(sentiment: {sentiment}, topic: {post_topic})")
                
                elif action_type == "like":
                    post_id = action.get("post_id")
                    if post_id and any(p["id"] == post_id for p in self.posts):
                        self.post_likes[post_id] += 1
                
                elif action_type == "share":
                    post_id = action.get("post_id")
                    if post_id and any(p["id"] == post_id for p in self.posts):
                        self.post_shares[post_id] += 1
        
        # Record epoch sentiment
        total_posts = sum(epoch_sentiment.values())
        if total_posts > 0:
            sentiment_ratio = {
                "positive": epoch_sentiment["positive"] / total_posts,
                "negative": epoch_sentiment["negative"] / total_posts,
                "neutral": epoch_sentiment["neutral"] / total_posts
            }
            self.sentiment_history.append(sentiment_ratio)
        
        self.topic_frequency_history.append(dict(epoch_topic_frequency))

        self.logger.info(f"Epoch {self.current_time.get_current_epoch() if self.current_time else 0}: "
                        f"New posts={total_posts}, Sentiment distribution={epoch_sentiment}, "
                        f"Topic Freq={dict(epoch_topic_frequency)}, Market Sentiment={self.current_market_sentiment}")

    def get_sentiment_influence(self) -> float:
        """Calculate sentiment influence factor for housing market"""
        if self.current_market_sentiment == "pessimistic":
            return -self.sentiment_influence_factor
        elif self.current_market_sentiment == "optimistic":
            return self.sentiment_influence_factor
        else:
            return 0.0

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate public opinion dynamics regarding housing policy"""
        # Analyze sentiment trends
        sentiment_trends = {
            "positive": [],
            "negative": [],
            "neutral": []
        }
        
        for sentiment_ratio in self.sentiment_history:
            for sentiment, ratio in sentiment_ratio.items():
                sentiment_trends[sentiment].append(ratio)
        
        # Find most engaged posts
        top_posts = sorted(self.posts, 
                          key=lambda p: self.post_likes[p["id"]] + self.post_shares[p["id"]], 
                          reverse=True)[:10]
        
        # Calculate engagement metrics
        total_likes = sum(self.post_likes.values())
        total_shares = sum(self.post_shares.values())
        
        evaluation_results = {
            "total_posts": len(self.posts),
            "total_engagement": {
                "likes": total_likes,
                "shares": total_shares,
                "average_likes_per_post": total_likes / len(self.posts) if self.posts else 0,
                "average_shares_per_post": total_shares / len(self.posts) if self.posts else 0
            },
            "sentiment_analysis": {
                "final_sentiment_distribution": self.sentiment_history[-1] if self.sentiment_history else {},
                "sentiment_trends": sentiment_trends,
                "dominant_sentiment": max(self.sentiment_history[-1].items(), key=lambda x: x[1])[0] if self.sentiment_history else "neutral"
            },
            "topic_analysis": {
                "overall_topic_frequency": dict(self.topic_frequency),
                "most_discussed_topic_overall": max(self.topic_frequency.items(), key=lambda x: x[1])[0] if self.topic_frequency else "general",
                "topic_frequency_over_time": self.topic_frequency_history
            },
            "top_posts": [
                {
                    "content": post["content"][:100] + "..." if len(post["content"]) > 100 else post["content"],
                    "sentiment": post["sentiment"],
                    "topic": post["topic"],
                    "engagement": self.post_likes[post["id"]] + self.post_shares[post["id"]]
                }
                for post in top_posts[:5]
            ],
            "policy_impact": {
                "sentiment_shift_after_policy": self._calculate_policy_impact() if len(self.sentiment_history) > 3 else "insufficient_data",
                "sentiment_influence_on_market": self.get_sentiment_influence()
            }
        }
        
        self.logger.info(f"evaluation_results={evaluation_results}")
  
        return evaluation_results

    def _calculate_policy_impact(self) -> str:
        """Calculate how policy announcements affected sentiment"""
        # Simple before/after comparison
        early_sentiment = self.sentiment_history[:2]
        late_sentiment = self.sentiment_history[-2:]
        
        if not early_sentiment or not late_sentiment:
            return "insufficient_data"
        
        early_negative = sum(s.get("negative", 0) for s in early_sentiment) / len(early_sentiment)
        late_negative = sum(s.get("negative", 0) for s in late_sentiment) / len(late_sentiment)
        
        if late_negative > early_negative + 0.1:
            return "increased_negative_sentiment"
        elif late_negative < early_negative - 0.1:
            return "decreased_negative_sentiment"
        else:
            return "stable_sentiment"

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        return {
            "total_posts": len(self.posts),
            "current_sentiment": self.sentiment_history[-1] if self.sentiment_history else {},
            "trending_topics": list(self.topic_frequency.keys())[:3],
            "market_sentiment": self.current_market_sentiment,
            "epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 