from typing import Dict, Any, List, Optional
import random
import uuid
from datetime import datetime,timedelta  # Use this import for the datetime class
import pandas as pd
import math
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger
from src.utils.data_loader import get_nested_value
from src.utils.embedding_clients import EmbeddingClient, cosine_similarity
from src.utils.llm_clients import OpenaiModel, LLMConfig # For sentiment analysis

class OpinionSystemExp1(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any],blackboard: Any = None,
                 embedding_client: Optional[EmbeddingClient] = None, 
                 llm_client_for_sentiment: Optional[OpenaiModel] = None):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        self.recommendation_algorithm = config.get("recommendation_algorithm", "similar")
        self.num_recommendations = config.get("num_recommendations", 5)
        self.use_initial_posts = config.get("use_initial_posts", True)
        self.news_headlines = config.get("news_headlines", {})
        self.embedding_client = embedding_client
        self.llm_client_for_sentiment = llm_client_for_sentiment

        self.posts: List[Dict[str, Any]] = [] # List of {id, agent_id, content, timestamp, likes, shares, embedding}
        self.agent_profiles: Dict[str, Dict[str, Any]] = {} # agent_id -> {attributes, embedding}
        self.overall_sentiment_history: List[Dict[str, Any]] = []
        self.wordcloud_data_history: List[Dict[str, int]] = [] # epoch -> word counts
        self.next_post_id: int = 0

        # Initial posts are generic and not from news_headlines configuration
        if self.use_initial_posts:
            self._generate_initial_posts() 
        self.logger.info(f"Initialized OpinionSystemExp1 with config: {config}")

    async def init_async_resources(self):
        # Create embeddings for initial posts if not already done and client available
        if self.use_initial_posts and self.embedding_client:
            post_contents = [post['content'] for post in self.posts if post.get('embedding') is None]
            if post_contents:
                embeddings = await self.embedding_client.get_embeddings(post_contents)
                if embeddings:
                    idx = 0
                    for post in self.posts:
                        if post.get('embedding') is None and idx < len(embeddings):
                            post['embedding'] = embeddings[idx]
                            idx += 1
    
    def _generate_initial_posts(self):
        # Generate 5 initial posts about Spring Festival (as per existing code)
        # These are not from the self.news_headlines config.
        spring_festival_posts = [

        ]
        initial_timestamp = self.current_time.current_time if self.current_time else datetime.now()

        for post_content in spring_festival_posts:
            self.posts.append({
                "id": str(self.next_post_id), # Use incrementing ID
                "agent_id": "system_initial_posts", # Differentiate from system_news
                "content": post_content,
                "timestamp": initial_timestamp,
                "likes": 0,
                "shares": 0,
                "embedding": None
            })
            self.next_post_id += 1 # Increment after use
        self.logger.info(f"Generated {len(spring_festival_posts)} initial posts about Spring Festival.")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        self.logger.info(f"Initializing OpinionSystemExp1 state for {len(all_agent_data)} agents.")
        self.agent_profiles = {}
        self.posts = [] # Reset posts if re-initializing
        self.next_post_id = 0 # Reset post ID counter on init
        if self.use_initial_posts:
             self._generate_initial_posts()
        self.overall_sentiment_history = []
        self.wordcloud_data_history = []

        for agent_data in all_agent_data:
            agent_id = get_nested_value(agent_data, "id")
            if agent_id is None: continue
            agent_id = str(agent_id)
            
            profile = {"attributes": {}}
            for attr_key in self.required_agent_attributes:
                profile["attributes"][attr_key] = get_nested_value(agent_data, attr_key, "Unknown")
            
            # Precompute agent profile embedding if possible (simplistic: join attributes)
            # A more robust approach would be to embed a descriptive string of the agent
            profile_text = ", ".join([f"{k}: {v}" for k, v in profile["attributes"].items()])
            profile["profile_text"] = profile_text # Store for later embedding if client available then
            profile["embedding"] = None # Will be populated later if embedding_client is set
            self.agent_profiles[agent_id] = profile
    
    async def precompute_embeddings_if_needed(self):
        # Called by scheduler before epoch starts if embeddings are used
        if not self.embedding_client: return

        # Embed agent profiles
        agent_ids_to_embed = [uid for uid, prof in self.agent_profiles.items() if prof.get('embedding') is None and prof.get("profile_text")]
        profile_texts_to_embed = [self.agent_profiles[uid]["profile_text"] for uid in agent_ids_to_embed]
        if profile_texts_to_embed:
            profile_embeddings = await self.embedding_client.get_embeddings(profile_texts_to_embed)
            if profile_embeddings:
                for i, agent_id in enumerate(agent_ids_to_embed):
                    self.agent_profiles[agent_id]["embedding"] = profile_embeddings[i]
        
        # Embed posts that don't have one
        post_indices_to_embed = [i for i, post in enumerate(self.posts) if post.get('embedding') is None]
        post_contents_to_embed = [self.posts[i]["content"] for i in post_indices_to_embed]
        if post_contents_to_embed:
            post_embeddings = await self.embedding_client.get_embeddings(post_contents_to_embed)
            if post_embeddings:
                for i, post_idx in enumerate(post_indices_to_embed):
                    if i < len(post_embeddings):
                        self.posts[post_idx]["embedding"] = post_embeddings[i]


    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        current_epoch = self.current_time.get_current_epoch()
        self.logger.info(f"OpinionSystemExp1 - Epoch {current_epoch} Step Start")

        new_posts_this_epoch = []
        for agent_id, decisions in agent_decisions.items():
            op_decision = decisions.get(self.name, {}) # Get decisions for this subsystem
            social_actions = op_decision.get("social_actions", [])
            if not isinstance(social_actions, list):
                social_actions = [social_actions] # Handle single action not in list

            for action_item in social_actions: # An agent can perform multiple actions
                if not isinstance(action_item, dict): continue
                action_type = action_item.get("action")
                
                if action_type == "post" and action_item.get("content"):
                    post_id_str = str(self.next_post_id)
                    new_post = {
                        "id": post_id_str,
                        "agent_id": agent_id,
                        "content": action_item["content"],
                        "timestamp": self.current_time.current_time,
                        "likes": 0,
                        "shares": 0,
                        "embedding": None # Will be embedded in next precompute_embeddings_if_needed call
                    }
                    self.next_post_id += 1 # Increment after use
                    new_posts_this_epoch.append(new_post)
                    self.logger.debug(f"Agent {agent_id} posted: {action_item['content'][:50]}...")
                
                elif action_type in ["like", "repost"] and action_item.get("post_id"):
                    post_id_acted_on = action_item["post_id"]
                    for post in self.posts:
                        if post["id"] == post_id_acted_on:
                            if action_type == "like":
                                post["likes"] += 1
                                self.logger.debug(f"Agent {agent_id} liked post {post_id_acted_on}")
                            elif action_type == "repost": # Repost could create a new post or just bump score
                                post["shares"] += 1
                                self.logger.debug(f"Agent {agent_id} reposted post {post_id_acted_on}")
                            break
        self.posts.extend(new_posts_this_epoch)
        # News headlines are NOT added to self.posts here.
        # They are provided via get_system_information.
        current_epoch_str = str(self.current_time.get_current_epoch())
        if current_epoch_str in self.news_headlines:
            news_content = self.news_headlines[current_epoch_str]
            self.logger.info(f"Current news headline for epoch {current_epoch_str}: {news_content[:50]}...")

        self.logger.info(f"OpinionSystemExp1 - Epoch {current_epoch} Step End. Total posts: {len(self.posts)}")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        # Prepare recommended posts part
        recommended_posts_details = []
        if self.posts: # Only recommend if there are posts
            agent_profile = self.agent_profiles.get(agent_id)
            if self.recommendation_algorithm == "similar" and self.embedding_client and agent_profile and agent_profile.get("embedding") and any(p.get("embedding") for p in self.posts):
                agent_embedding = agent_profile["embedding"]
                embeddable_posts = [p for p in self.posts if p.get("embedding") is not None]
                if not embeddable_posts:
                    top_posts = random.sample(self.posts, min(len(self.posts), self.num_recommendations)) # Fallback to random from all posts
                else:
                    scored_posts = []
                    for post in embeddable_posts:
                        if post["embedding"]:
                            similarity = cosine_similarity(agent_embedding, post["embedding"])
                            time_decay = 1.0 / ( (self.current_time.current_time - post["timestamp"]).total_seconds() / 3600 + 1) 
                            engagement_score = math.log(post["likes"] + post["shares"] * 2 + 1)  # Add 1 to avoid log(0)
                            score = 0.4 * similarity + 0.5 * time_decay + 0.1 * engagement_score
                            scored_posts.append((score, post))
                    scored_posts.sort(key=lambda x: x[0], reverse=True)
                    top_posts = [post for score, post in scored_posts[:self.num_recommendations]]
            else:
                if self.recommendation_algorithm == "similar":
                     self.logger.warning(f"Cannot use 'similar' recommendation for agent {agent_id}. Conditions not met. Falling back to random.")
                top_posts = random.sample(self.posts, min(len(self.posts), self.num_recommendations))
            
            for post in top_posts:
                recommended_posts_details.append({
                    "id": post["id"],
                    "content": post["content"],
                    "likes": post["likes"],
                    "shares": post["shares"],
                    "author_id": post["agent_id"]
                })
        else: # No posts to recommend
            recommended_posts_details = [] 

        # Prepare current news headline part
        current_epoch_str = str(self.current_time.get_current_epoch())
        current_time_str = self.current_time.get_current_time_str("%Y-%m-%d")
        current_news_headline = self.news_headlines.get(current_epoch_str, "No specific news headline this period.")

        # Format historical news with epoch information and time information
        historical_news_info = {}
        for epoch, news_text in self.news_headlines.items():
            # Calculate the date for this historical news based on epoch
            # Each epoch is typically a month as per the documentation
            current_epoch = self.current_time.get_current_epoch()
            epoch_diff = current_epoch - int(epoch)
            # Calculate approximate date
            if epoch_diff > 0:
                historical_date = self.current_time.current_time - timedelta(days=30 * epoch_diff)
                historical_time_str = historical_date.strftime("%Y-%m-%d")
                historical_news_info[f"date_{historical_time_str}"] = news_text

        # Construct the full system information
        system_info = {}
        if "recommended_posts" in self.environment_attributes:
            system_info["recommended_posts"] = recommended_posts_details
        
        # Add historical news, assuming it will be an expected environment_attribute
        system_info["historical_news_headlines"] = historical_news_info
        
        if "current_news_headline" in self.environment_attributes:
            system_info["current_news_headline"] = {
                "date": current_time_str,
                "headline_text": current_news_headline
            }
        
        return system_info

    def _get_random_recommendations(self) -> Dict[str, Any]:
        # This is a helper for the main get_system_information, so it only returns the posts part
        if not self.posts: return {"recommended_posts": []} # Should align with how it's used
        top_posts = random.sample(self.posts, min(len(self.posts), self.num_recommendations))
        recommended_posts_details = [
            {"id": p["id"], "content": p["content"], "likes": p["likes"], "shares": p["shares"], "author_id": p["agent_id"]}
            for p in top_posts
        ]
        return {"recommended_posts": recommended_posts_details} if "recommended_posts" in self.environment_attributes else {}

    async def evaluate(self) -> Dict[str, Any]:
        self.logger.info("Evaluating OpinionSystemExp1 results.")
        
        # Simplified evaluation that just returns all posts for further processing
        results = {
            "total_posts": len(self.posts),
            "all_posts": [],  # Will contain posts with datetime converted to string
            "posts_by_epoch": {}
        }
        
        # Process posts and convert datetime to string format
        for post in self.posts:
            # Create a copy of the post with timestamp converted to string
            post_copy = {
                "id": post["id"],
                "agent_id": post["agent_id"], 
                "content": post["content"],
                "likes": post["likes"],
                "shares": post["shares"]
            }
            
            if 'timestamp' in post and isinstance(post['timestamp'], datetime):
                post_copy['timestamp'] = post['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            
            # Add to all_posts
            results["all_posts"].append(post_copy)
            
            # Organize by epoch
            if isinstance(post['timestamp'], datetime):
                post_date = post['timestamp']
                epoch = self.current_time.get_epoch_from_date(post_date)
                if epoch not in results["posts_by_epoch"]:
                    results["posts_by_epoch"][epoch] = []
                # Add the string-converted post to posts_by_epoch
                results["posts_by_epoch"][epoch].append(post_copy)
            elif post.get("agent_id") == "system_initial_posts":
                if -1 not in results["posts_by_epoch"]:
                    results["posts_by_epoch"][-1] = []
                results["posts_by_epoch"][-1].append(post_copy)

        self.evaluation_results = results
        self.logger.info("OpinionSystemExp1 evaluation complete - returning raw post data for external processing.")
        return results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        return {
            "total_posts": len(self.posts),
            "latest_sentiment": self.overall_sentiment_history[-1] if self.overall_sentiment_history else None,
            "latest_wordcloud_data": self.wordcloud_data_history[-1]["counts"] if self.wordcloud_data_history and "counts" in self.wordcloud_data_history[-1] else None
        }


