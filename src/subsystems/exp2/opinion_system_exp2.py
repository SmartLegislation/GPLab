from typing import Dict, Any, List, Tuple
import random
import networkx as nx
import numpy as np
from collections import defaultdict
import time

from src.subsystems.base import SocialSystemBase
from src.simulation.time import SimulationTime
from src.utils.logger import get_logger

class OpinionSystemExp2(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Any = None):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(self.name)
        
        # --- Attributes specific to OpinionSystemExp2 ---
        # Social network configuration
        self.network_k_neighbors = config.get("network_k_neighbors", 4)
        self.network_rewire_prob = config.get("network_rewire_prob", 0.1)
        self.num_recommendations = config.get("num_recommendations", 5)
        self.recommendation_algorithm = config.get("recommendation_algorithm", "similar")
        
        # Store network graph
        self.network = None
        self.embeddings = {}  # agent_id -> embedding vector
        
        # Posts and interactions
        self.posts = []  # List of post dicts with id, author_id, content, timestamp, likes, etc.
        self.agent_posts = defaultdict(list)  # agent_id -> list of post_ids
        self.agent_likes = defaultdict(set)  # agent_id -> set of post_ids liked
        self.agent_reposts = defaultdict(set)  # agent_id -> set of post_ids reposted
        
        # Consumer type information (now managed here, not in environment system)
        self.agent_consumer_types = {}  # agent_id -> "green" or "traditional"
        self.initial_green_consumer_ratio = config.get("initial_green_consumer_ratio", 0.5)
        
        # Metrics
        self.system_state["green_consumer_ratio"] = 0.0
        self.system_state["traditional_consumer_ratio"] = 0.0
        self.system_state["total_posts"] = 0
        self.system_state["total_likes"] = 0
        self.system_state["total_reposts"] = 0
        self.system_state["post_sentiment_green"] = 0.0  # Average sentiment towards green products
        
        # Government messages by epoch
        self.government_promotion_headlines = config.get("government_promotion_headlines", {})
        self.gov_post_base_likes = config.get("gov_post_base_likes", 15)
        
        # Track metrics over time for evaluation
        self.epoch_metrics = {}
        
        self.logger.info(f"OpinionSystemExp2 initialized with config: {config}")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        self.logger.info("Initializing OpinionSystemExp2...")
        
        # Initialize consumer types
        self._initialize_consumer_types(all_agent_data)
        
        # Build social network (Watts-Strogatz small-world)
        self._build_social_network(all_agent_data)
        
        # --- BEGIN MODIFICATION: Store initial network state as epoch -1 ---
        if self.network: # Ensure network is built
            initial_nodes = []
            for node_idx in self.network.nodes():
                agent_id = str(self.node_to_agent[node_idx])
                consumer_type = self.agent_consumer_types.get(agent_id, "unknown")
                initial_nodes.append({"id": agent_id, "type": consumer_type})
            
            initial_edges = []
            for u_idx, v_idx in self.network.edges():
                initial_edges.append({
                    "source": str(self.node_to_agent[u_idx]),
                    "target": str(self.node_to_agent[v_idx])
                })
            
            self.epoch_metrics[-1] = {
                "green_ratio": self.system_state["green_consumer_ratio"],
                "traditional_ratio": self.system_state["traditional_consumer_ratio"],
                "total_posts": 0, # No posts at true initial state
                "network_snapshot": self.network.copy(), # Store the graph object
                "nodes": initial_nodes, # Store node list with types
                "edges": initial_edges,  # Store edge list
                "agent_types_at_epoch": dict(self.agent_consumer_types) # Store consumer types at this point
            }
            self.logger.info(f"Stored initial network state in epoch_metrics[-1] with {len(initial_nodes)} nodes and {len(initial_edges)} edges.")
        # --- END MODIFICATION ---
        
        # Initialize metrics tracking
        if self.current_time:
            current_epoch = self.current_time.get_current_epoch()
            
            # Create a proper network representation with nodes and edges
            nodes = []
            for node_idx in self.network.nodes(): # Corrected loop variable
                agent_id = str(self.node_to_agent[node_idx]) # Ensure agent_id is string for consistency
                consumer_type = self.agent_consumer_types.get(agent_id, "unknown")
                nodes.append({
                    "id": agent_id,
                    "type": consumer_type
                })
            
            # Extract edges
            edges = []
            for u_idx, v_idx in self.network.edges(): # Corrected loop variable
                edges.append({
                    "source": str(self.node_to_agent[u_idx]), # Ensure agent_id is string
                    "target": str(self.node_to_agent[v_idx])  # Ensure agent_id is string
                })
            
            self.epoch_metrics[current_epoch] = { # current_epoch is typically 0 here
                "green_ratio": self.system_state["green_consumer_ratio"],
                "traditional_ratio": self.system_state["traditional_consumer_ratio"],
                "total_posts": 0,
                "network_snapshot": self.network.copy(),
                "nodes": nodes,
                "edges": edges,
                "agent_types_at_epoch": dict(self.agent_consumer_types) # Store consumer types at epoch 0
            }
            
            self.logger.info(f"Initialized network snapshot for epoch {current_epoch} with {len(nodes)} nodes and {len(edges)} edges")
            
        # Post first government message if available
        self._post_government_message(epoch_str="0")
        
        self.logger.info("OpinionSystemExp2 initialized.")
    
    def _initialize_consumer_types(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize consumer types for all agents."""
        num_agents = len(all_agent_data)
        num_green_consumers = int(num_agents * self.initial_green_consumer_ratio)
        
        agent_ids = [str(agent["id"]) for agent in all_agent_data]  # Ensure agent_ids are strings
        random.shuffle(agent_ids)  # Shuffle to randomize assignment
        
        for i, agent_id in enumerate(agent_ids):
            if i < num_green_consumers:
                self.agent_consumer_types[agent_id] = "green"
            else:
                self.agent_consumer_types[agent_id] = "traditional"
        
        self._update_consumer_ratios()
        self.logger.info(f"Initialized consumer types: {self.system_state['green_consumer_ratio']:.2f} green, {self.system_state['traditional_consumer_ratio']:.2f} traditional.")
    
    def _update_consumer_ratios(self):
        """Update the ratios of green vs traditional consumers."""
        if not self.agent_consumer_types:
            self.system_state["green_consumer_ratio"] = 0.0
            self.system_state["traditional_consumer_ratio"] = 0.0
            return

        num_total_consumers = len(self.agent_consumer_types)
        num_green = sum(1 for type_val in self.agent_consumer_types.values() if type_val == "green")
        
        self.system_state["green_consumer_ratio"] = num_green / num_total_consumers if num_total_consumers > 0 else 0
        self.system_state["traditional_consumer_ratio"] = 1.0 - self.system_state["green_consumer_ratio"]
    
    def _build_social_network(self, all_agent_data: List[Dict[str, Any]]):
        """Build a small-world network using the Watts-Strogatz model."""
        agent_ids = [str(agent["id"]) for agent in all_agent_data] # Ensure agent_ids are strings
        n = len(agent_ids)
        
        # Ensure k is even and less than n
        k = min(self.network_k_neighbors, n-1)
        if k % 2 == 1:
            k -= 1
        if k < 2:
            k = 2
            
        # Create the small-world network
        self.network = nx.watts_strogatz_graph(n, k, self.network_rewire_prob)
        
        # Map node indices to agent_ids
        self.node_to_agent = {i: agent_ids[i] for i in range(n)}
        self.agent_to_node = {agent_id: i for i, agent_id in self.node_to_agent.items()}
        
        self.logger.info(f"Built small-world network with {n} nodes, {k} initial neighbors, and rewire probability {self.network_rewire_prob}")
    
    def _post_government_message(self, epoch_str: str):
        """Post a message from the government at the given epoch."""
        if epoch_str in self.government_promotion_headlines:
            message = self.government_promotion_headlines[epoch_str]
            
            # Generate unique ID with timestamp to avoid collisions
            post_id = f"{len(self.posts)}"
            post = {
                "id": post_id,
                "author_id": "government",
                "content": message,
                "timestamp": self.current_time.get_current_time_str() if self.current_time else "0",
                "likes": self.gov_post_base_likes,  # Government posts start with higher visibility
                "reposts": 0,
                "is_government": True,
                "epoch": epoch_str
            }
            
            self.posts.append(post)
            self.system_state["total_posts"] += 1
            self.logger.info(f"Government posted message for epoch {epoch_str}: {message[:50]}... (total posts: {self.system_state['total_posts']})")
    
    def set_time(self, simulation_time: SimulationTime):
        """Override to post government messages at appropriate epochs."""
        previous_epoch = self.current_time.get_current_epoch() if self.current_time else None
        super().set_time(simulation_time)
        current_epoch = simulation_time.get_current_epoch()
        
        # Post government message if epoch changed
        if previous_epoch != current_epoch:
            epoch_str = str(current_epoch)
            self._post_government_message(epoch_str)
    
    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        self.logger.debug(f"OpinionSystemExp2 stepping with agent decisions: {agent_decisions}")
        decisions_made = False
        
        # Track decisions that change consumer types for better reporting
        consumer_type_changes = 0
        posts_created = 0
        likes_added = 0
        reposts_made = 0
        
        # Process agent decisions
        for agent_id, decisions in agent_decisions.items():
            if not decisions:
                continue
            
            # Check for OpinionSystemExp2 specific decisions
            if "OpinionSystemExp2" in decisions and isinstance(decisions["OpinionSystemExp2"], dict):
                opinion_decisions = decisions["OpinionSystemExp2"]
                
                # Process consumer type decisions
                if "become_consumer_type" in opinion_decisions:
                    new_type = opinion_decisions["become_consumer_type"]
                    if new_type in ["green", "traditional"]:
                        old_type = self.agent_consumer_types.get(agent_id, "unknown") # Use string agent_id
                        if old_type != new_type:
                            self.agent_consumer_types[agent_id] = new_type # Use string agent_id
                            self.logger.info(f"Agent {agent_id} changed consumer type from {old_type} to {new_type}")
                            decisions_made = True
                            consumer_type_changes += 1
                
                # Process social actions (posts, likes, reposts)
                if "social_actions" in opinion_decisions:
                    actions = opinion_decisions["social_actions"]
                    
                    # The social_actions is already a JSON string, parse it
                    if isinstance(actions, str):
                        import json
                        try:
                            parsed_actions = json.loads(actions)
                            
                            # Convert to list if it's a single action
                            if not isinstance(parsed_actions, list):
                                parsed_actions = [parsed_actions]
                                
                            for action in parsed_actions:
                                if isinstance(action, dict):
                                    action_type = action.get("action", "")
                                    
                                    # Handle post action
                                    if action_type == "post" and "content" in action:
                                        # Generate unique post ID with timestamp
                                        post_id = f"{len(self.posts)}"
                                        post = {
                                            "id": post_id,
                                            "author_id": agent_id,
                                            "content": action["content"],
                                            "timestamp": self.current_time.get_current_time_str() if self.current_time else "0",
                                            "likes": 0,
                                            "reposts": 0,
                                            "is_government": False,
                                            "consumer_type": self.agent_consumer_types.get(agent_id, "unknown") # Use string agent_id (author)
                                        }
                                        
                                        self.posts.append(post)
                                        self.agent_posts[agent_id].append(post_id)
                                        self.system_state["total_posts"] += 1
                                        posts_created += 1
                                        decisions_made = True
                                        self.logger.debug(f"Agent {agent_id} posted: {action['content'][:50]}...")
                                        
                                    # Handle like action
                                    elif action_type == "like" and "post_id" in action:
                                        post_id = action["post_id"]
                                        for post in self.posts:
                                            if post["id"] == post_id and post_id not in self.agent_likes[agent_id]:
                                                post["likes"] += 1
                                                self.agent_likes[agent_id].add(post_id)
                                                self.system_state["total_likes"] += 1
                                                likes_added += 1
                                                decisions_made = True
                                                break
                                                
                                    # Handle repost action
                                    elif action_type == "repost" and "post_id" in action:
                                        post_id = action["post_id"]
                                        for post in self.posts:
                                            if post["id"] == post_id and post_id not in self.agent_reposts[agent_id]:
                                                post["reposts"] += 1
                                                self.agent_reposts[agent_id].add(post_id)
                                                self.system_state["total_reposts"] += 1
                                                reposts_made += 1
                                                decisions_made = True
                                                
                                                # Create a repost entry with unique ID
                                                repost_id = f"repost_{agent_id}_{int(time.time()*1000)}_{len(self.posts)}"
                                                repost = {
                                                    "id": repost_id,
                                                    "author_id": agent_id,
                                                    "original_post_id": post_id,
                                                    "original_author_id": post["author_id"],
                                                    "content": post["content"],
                                                    "timestamp": self.current_time.get_current_time_str() if self.current_time else "0",
                                                    "likes": 0,
                                                    "reposts": 0,
                                                    "is_repost": True,
                                                    "is_government": post.get("is_government", False),
                                                    "consumer_type": self.agent_consumer_types.get(agent_id, "unknown") # agent_id is reposter, str
                                                }
                                                self.posts.append(repost)
                                                break
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Could not parse social_actions JSON from agent {agent_id}: {e}")
            
            # Legacy format support - for backward compatibility
            # Process consumer type decisions directly
            elif "become_consumer_type" in decisions:
                new_type = decisions["become_consumer_type"]
                if new_type in ["green", "traditional"]:
                    old_type = self.agent_consumer_types.get(agent_id, "unknown") # agent_id is str
                    if old_type != new_type:
                        self.agent_consumer_types[agent_id] = new_type # agent_id is str
                        self.logger.info(f"Agent {agent_id} changed consumer type from {old_type} to {new_type} (legacy format)")
                        decisions_made = True
                        consumer_type_changes += 1
            
            # Legacy social actions handling
            elif "social_actions" in decisions:
                self.logger.warning(f"Using legacy social_actions format for agent {agent_id}")
                actions = decisions["social_actions"]
                if isinstance(actions, str):
                    # Try to parse JSON if it's a string
                    import json
                    try:
                        actions = json.loads(actions)
                    except:
                        self.logger.warning(f"Could not parse social_actions JSON from agent {agent_id}: {actions}")
                        actions = []
                
                if not isinstance(actions, list):
                    actions = [actions]
                    
                for action in actions:
                    if isinstance(action, dict):
                        action_type = action.get("action", "")
                        
                        # Handle post action
                        if action_type == "post" and "content" in action:
                            # Generate unique post ID with timestamp
                            post_id = f"{agent_id}_{int(time.time()*1000)}_{len(self.posts)}"
                            post = {
                                "id": post_id,
                                "author_id": agent_id,
                                "content": action["content"],
                                "timestamp": self.current_time.get_current_time_str() if self.current_time else "0",
                                "likes": 0,
                                "reposts": 0,
                                "is_government": False,
                                "consumer_type": self.agent_consumer_types.get(agent_id, "unknown") # agent_id is str (author)
                            }
                            
                            self.posts.append(post)
                            self.agent_posts[agent_id].append(post_id)
                            self.system_state["total_posts"] += 1
                            posts_created += 1
                            decisions_made = True
                            self.logger.debug(f"Agent {agent_id} posted (legacy): {action['content'][:50]}...")
        
        # Update consumer ratios if decisions were made
        if decisions_made:
            self._update_consumer_ratios()
            
        # Log decision statistics
        if consumer_type_changes > 0 or posts_created > 0 or likes_added > 0 or reposts_made > 0:
            self.logger.info(f"Agent activity summary: {consumer_type_changes} consumer type changes, {posts_created} new posts, {likes_added} likes, {reposts_made} reposts")
            
        # Record metrics for this epoch
        if self.current_time:
            epoch = self.current_time.get_current_epoch()
            self.epoch_metrics[epoch] = {
                "green_ratio": self.system_state["green_consumer_ratio"],
                "traditional_ratio": self.system_state["traditional_consumer_ratio"],
                "total_posts": self.system_state["total_posts"],
                "network_snapshot": self.network.copy(),  # Store network every epoch
                "agent_types_at_epoch": dict(self.agent_consumer_types) # Store types every epoch
            }
        
        self.logger.debug("OpinionSystemExp2 step completed with current stats: " +
                         f"green_ratio={self.system_state['green_consumer_ratio']:.2f}, " +
                         f"traditional_ratio={self.system_state['traditional_consumer_ratio']:.2f}, " +
                         f"total_posts={self.system_state['total_posts']}")
    
    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Returns social information relevant to a specific agent."""
        # Get recommendations for this agent - half from network neighbors, half from content similarity
        recommended_posts = self._get_recommendations_for_agent(agent_id)
        
        # Get information about current consumer type distribution
        current_consumer_type = self.agent_consumer_types.get(agent_id, "unknown")
        
        # Get information about what agent's neighbors (in the network) believe
        neighbor_types = self._get_neighbor_consumer_types(agent_id)
        
        agent_info = {
            "recommended_posts": recommended_posts,
            "your_current_consumer_type": current_consumer_type,
            "your_neighbor_consumer_types": neighbor_types
        }
        
        return agent_info
    
    def _get_neighbor_consumer_types(self, agent_id: str) -> Dict[str, float]:
        """Get information about the consumer types of an agent's neighbors."""
        if agent_id not in self.agent_to_node: # agent_id is str, agent_to_node keys are str
            return {"green_ratio": 0, "traditional_ratio": 0, "total_neighbors": 0}
        
        node_id = self.agent_to_node[agent_id]
        neighbors = list(self.network.neighbors(node_id))
        total_neighbors = len(neighbors)
        
        if total_neighbors == 0:
            return {"green_ratio": 0, "traditional_ratio": 0, "total_neighbors": 0}
        
        neighbor_agent_ids = [self.node_to_agent[n] for n in neighbors]
        green_count = sum(1 for n_id in neighbor_agent_ids if self.agent_consumer_types.get(n_id, "") == "green") # n_id is str
        
        return {
            "green_ratio": green_count / total_neighbors,
            "traditional_ratio": (total_neighbors - green_count) / total_neighbors,
            "total_neighbors": total_neighbors
        }
    
    def _get_recommendations_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get recommended posts for an agent based on network and content similarity."""
        if len(self.posts) == 0:
            return []
            
        # Calculate how many posts to get from neighbors vs algorithm
        neighbor_count = self.num_recommendations // 2
        algorithm_count = self.num_recommendations - neighbor_count
        
        # Get posts from neighbors
        neighbor_posts = self._get_neighbor_posts(agent_id, neighbor_count)
        algorithm_posts = self._get_recent_high_engagement_posts(algorithm_count)
        
        # Combine and return
        recommended_posts = neighbor_posts + algorithm_posts
        
        # Convert to format suitable for agent
        simplified_posts = []
        for post in recommended_posts:
            is_gov = post.get("is_government", False)
            simplified = {
                "post_id": post["id"],
                "author": "Government" if is_gov else f"Agent_{post['author_id']}",
                "content": post["content"],
                "likes": post["likes"],
                "consumer_type": "government" if is_gov else post.get("consumer_type", "unknown")
            }
            simplified_posts.append(simplified)
            
        return simplified_posts
    
    def _get_neighbor_posts(self, agent_id: str, count: int) -> List[Dict[str, Any]]:
        """Get recent posts from network neighbors."""
        if agent_id not in self.agent_to_node: # agent_id is str
            return []
            
        node_id = self.agent_to_node[agent_id]
        neighbors = list(self.network.neighbors(node_id))
        
        if not neighbors:
            return []
            
        # Get posts from neighbors
        neighbor_agent_ids = [self.node_to_agent[n] for n in neighbors]
        neighbor_posts = []
        
        for n_id in neighbor_agent_ids:
            for post_id in self.agent_posts.get(n_id, []):
                for post in self.posts:
                    if post["id"] == post_id:
                        neighbor_posts.append(post)
        
        # Sort by recency (assuming higher post ids are more recent)
        neighbor_posts.sort(key=lambda x: x["id"], reverse=True)
        
        return neighbor_posts[:count]
    
    def _get_similar_content_posts(self, agent_id: str, count: int) -> List[Dict[str, Any]]:
        """Get posts with similar content to agent's interests."""
        # For simplicity, just get posts by agents with the same consumer type
        agent_type = self.agent_consumer_types.get(agent_id, "unknown") # agent_id is str
        
        if agent_type == "unknown":
            return self._get_recent_high_engagement_posts(count)
            
        # Get posts from agents with same consumer type
        similar_posts = []
        for post in self.posts:
            post_type = post.get("-", "unknown")
            if post_type == agent_type:
                similar_posts.append(post)
                
        # Sort by engagement (likes + reposts)
        similar_posts.sort(key=lambda x: x.get("likes", 0) + x.get("reposts", 0), reverse=True)
        
        return similar_posts[:count]
    
    def _get_recent_high_engagement_posts(self, count: int) -> List[Dict[str, Any]]:
        """Get recent posts with high engagement."""
        # Copy posts and sort by engagement
        sorted_posts = sorted(self.posts, key=lambda x: x.get("likes", 0) + x.get("reposts", 0), reverse=True)
        
        return sorted_posts[:count]
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the state of the opinion system at the end of the simulation.
        Returns comprehensive metrics on social network and opinion dynamics.
        """
        self.logger.info("Evaluating OpinionSystemExp2...")
        self._update_consumer_ratios()  # Ensure ratios are current
        
        # Calculate network metrics
        degree_centrality = nx.degree_centrality(self.network)
        avg_clustering = nx.average_clustering(self.network)
        
        # Get temporal snapshots for visualization - beginning, middle, end
        temporal_snapshots = self._get_network_snapshots_for_visualization()
        
        # Analyze post sentiment and engagement by consumer type
        post_analysis = self._analyze_posts_by_consumer_type()
        
        # Count post statistics by type for verification
        gov_posts = sum(1 for p in self.posts if p.get("is_government", False))
        agent_posts = sum(1 for p in self.posts if not p.get("is_government", False) and not p.get("is_repost", False))
        reposts = sum(1 for p in self.posts if p.get("is_repost", False))
        
        self.evaluation_results = {
            "final_green_consumer_ratio": self.system_state["green_consumer_ratio"],
            "final_traditional_consumer_ratio": self.system_state["traditional_consumer_ratio"],
            "num_green_consumers": sum(1 for type_val in self.agent_consumer_types.values() if type_val == "green"),
            "num_traditional_consumers": sum(1 for type_val in self.agent_consumer_types.values() if type_val == "traditional"),
            "total_posts": self.system_state["total_posts"],
            "post_breakdown": {
                "government_posts": gov_posts,
                "agent_posts": agent_posts,
                "reposts": reposts
            },
            "total_likes": self.system_state["total_likes"],
            "total_reposts": self.system_state["total_reposts"],
            "network_metrics": {
                "avg_degree_centrality": sum(degree_centrality.values()) / len(degree_centrality) if degree_centrality else 0,
                "avg_clustering_coefficient": avg_clustering,
                "network_density": nx.density(self.network)
            },
            "post_engagement": post_analysis,
            "temporal_network_snapshots": temporal_snapshots,
            "agent_consumer_type_map": dict(self.agent_consumer_types),  # For network visualization
            "all_posts_for_analysis": [self._simplify_post(p) for p in self.posts]  # Add simplified posts for analysis
        }
        
        # self.logger.info(f"OpinionSystemExp2 evaluation results: {self.evaluation_results}")
        return self.evaluation_results
    
    def _get_network_snapshots_for_visualization(self) -> Dict[int, Any]:
        """Get network snapshots for all epochs that have valid snapshot data."""
        if not self.epoch_metrics:
            self.logger.warning("No epoch metrics available to create network snapshots.")
            return {}

        snapshots_to_return = {}
        epochs_with_valid_snapshots = sorted([
            epoch for epoch, metrics in self.epoch_metrics.items()
            if metrics.get("network_snapshot") is not None and metrics.get("agent_types_at_epoch") is not None
        ])

        if not epochs_with_valid_snapshots:
            self.logger.warning("No valid network snapshots (with network and agent types) found across all epochs.")
            return {}

        for epoch in epochs_with_valid_snapshots:
            metrics = self.epoch_metrics[epoch]
            network_graph_object = metrics.get("network_snapshot")
            agent_types_for_this_epoch = metrics.get("agent_types_at_epoch")

            # This check is redundant due to prior filtering but kept for safety
            if network_graph_object is None or agent_types_for_this_epoch is None:
                self.logger.warning(f"Unexpected None network_snapshot or agent_types_at_epoch for epoch {epoch} during final processing. Skipping.")
                continue

            current_nodes_list = []
            # Check if nodes and edges were pre-calculated (for epoch -1 and 0 in init)
            # For other epochs, generate them from the network_graph_object
            if "nodes" in metrics and "edges" in metrics and epoch in [-1, 0]:
                current_nodes_list = metrics["nodes"]
                current_edges_list = metrics["edges"]
            else:
                if not hasattr(self, 'node_to_agent') or not self.node_to_agent:
                    self.logger.error(f"node_to_agent mapping not initialized. Cannot create snapshot for epoch {epoch}.")
                    continue

                for node_idx in network_graph_object.nodes():
                    if node_idx not in self.node_to_agent:
                        self.logger.warning(f"Snapshot generation: Node index {node_idx} for epoch {epoch} not in node_to_agent map. Skipping node.")
                        continue
                    agent_id = str(self.node_to_agent[node_idx])
                    consumer_type = agent_types_for_this_epoch.get(agent_id, "unknown")
                    current_nodes_list.append({"id": agent_id, "type": consumer_type})
                
                current_edges_list = []
                for u_idx, v_idx in network_graph_object.edges():
                    if u_idx not in self.node_to_agent or v_idx not in self.node_to_agent:
                        self.logger.warning(f"Snapshot generation: Edge node index {u_idx} or {v_idx} for epoch {epoch} not in node_to_agent map. Skipping edge.")
                        continue
                    current_edges_list.append({
                        "source": str(self.node_to_agent[u_idx]),
                        "target": str(self.node_to_agent[v_idx])
                    })

            snapshots_to_return[epoch] = {
                "green_ratio": metrics["green_ratio"],
                "traditional_ratio": metrics["traditional_ratio"],
                "nodes": current_nodes_list,
                "edges": current_edges_list
            }
            self.logger.info(f"Created network snapshot for epoch {epoch} with {len(current_nodes_list)} nodes and {len(current_edges_list)} edges")
        
        self.logger.info(f"Returning {len(snapshots_to_return)} snapshots for visualization for epochs: {list(snapshots_to_return.keys())}")
        return snapshots_to_return
    
    def _analyze_posts_by_consumer_type(self) -> Dict[str, Any]:
        """Analyze post engagement by consumer type."""
        green_posts = []
        traditional_posts = []
        government_posts = []
        
        for post in self.posts:
            if post.get("is_government", False):
                government_posts.append(post)
            elif post.get("consumer_type") == "green":
                green_posts.append(post)
            elif post.get("consumer_type") == "traditional":
                traditional_posts.append(post)
        
        # Calculate average engagement
        def calc_avg_engagement(posts):
            if not posts:
                return {"avg_likes": 0, "avg_reposts": 0, "total_posts": 0}
            avg_likes = sum(p.get("likes", 0) for p in posts) / len(posts)
            avg_reposts = sum(p.get("reposts", 0) for p in posts) / len(posts)
            return {"avg_likes": avg_likes, "avg_reposts": avg_reposts, "total_posts": len(posts)}
        
        return {
            "green_consumer_posts": calc_avg_engagement(green_posts),
            "traditional_consumer_posts": calc_avg_engagement(traditional_posts),
            "government_posts": calc_avg_engagement(government_posts)
        }
    
    def _simplify_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simplified version of a post for analysis and visualization"""
        return {
            "id": post.get("id", ""),
            "content": post.get("content", ""),
            "author_id": post.get("author_id", ""),
            "consumer_type": post.get("consumer_type", "unknown"),
            "is_government": post.get("is_government", False),
            "likes": post.get("likes", 0),
            "reposts": post.get("reposts", 0),
            "timestamp": post.get("timestamp", "")
        }
    
    def get_state_for_persistence(self) -> Dict[str, Any]:
        state = super().get_state_for_persistence()
        state.update({
            "agent_consumer_types": self.agent_consumer_types,
            "posts": self.posts,
            "agent_posts": dict(self.agent_posts),
            "agent_likes": {k: list(v) for k, v in self.agent_likes.items()},
            "agent_reposts": {k: list(v) for k, v in self.agent_reposts.items()},
        })
        return state 