import datetime
import math
from typing import List, Dict, Any, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.embedding_clients import EmbeddingClient, cosine_similarity
from src.utils.llm_clients import OpenaiModel

logger = get_logger(__name__)

class Memory:
    def __init__(self, 
                 content: str, 
                 timestamp: datetime.datetime, 
                 type: str = "observation", # observation, reflection
                 importance: float = 1.0, # 0-10, potentially LLM assigned later
                 embedding: Optional[List[float]] = None):
        self.content = content
        self.timestamp = timestamp
        self.type = type
        self.importance = importance
        self.embedding = embedding
        self.last_access_time = timestamp

    def update_last_access(self, time: datetime.datetime):
        self.last_access_time = time

    def __repr__(self):
        return f"Memory(type='{self.type}', time='{self.timestamp}', importance={self.importance}, content='{self.content[:50]}...')"

class MemorySystem:
    def __init__(self, 
                 agent_id: str, 
                 embedding_client: Optional[EmbeddingClient],
                 reflection_llm_client: Optional[OpenaiModel],
                 reflection_threshold: int = 20,
                 reflection_count: int = 2,
                 time_weight_lambda: float = 0.5, # Balance between time and similarity
                 retrieval_k: int = 5):
        self.agent_id = agent_id
        self.memories: List[Memory] = []
        self.embedding_client = embedding_client
        self.reflection_llm_client = reflection_llm_client
        self.reflection_threshold = reflection_threshold
        self.reflection_count = reflection_count
        self.time_weight_lambda = time_weight_lambda
        self.retrieval_k = retrieval_k
        self.memories_since_last_reflection = 0

    async def add_memory(self, content: str, timestamp: datetime.datetime, type: str = "observation", importance: float = 1.0):
        embedding = None
        if self.embedding_client:
            embedding = await self.embedding_client.get_embedding(content)
            if embedding is None:
                logger.warning(f"Agent {self.agent_id}: Failed to get embedding for memory: {content[:50]}...")
        
        new_memory = Memory(content, timestamp, type, importance, embedding)
        self.memories.append(new_memory)
        logger.debug(f"Agent {self.agent_id}: Added memory: {new_memory}")

        self.memories_since_last_reflection += 1
        if self.memories_since_last_reflection >= self.reflection_threshold and self.reflection_llm_client:
            await self._reflect(timestamp)

    async def _reflect(self, current_time: datetime.datetime):
        logger.info(f"Agent {self.agent_id}: Triggering reflection.")
        self.memories_since_last_reflection = 0
        
        # Get recent memories to reflect upon (e.g., last N or based on importance)
        recent_memories = sorted(self.memories, key=lambda m: m.timestamp, reverse=True)[:self.reflection_threshold * 2] # Look back a bit further
        if not recent_memories:
            logger.warning(f"Agent {self.agent_id}: No memories found to reflect upon.")
            return
        
        memory_contents = [m.content for m in recent_memories]
        try:
            reflections = await self.reflection_llm_client.generate_reflection(memory_contents)
            for reflection_content in reflections[:self.reflection_count]:
                logger.info(f"Agent {self.agent_id}: Generated reflection: {reflection_content}")
                await self.add_memory(f"My reflection: {reflection_content}", current_time, type="reflection", importance=5.0) # Reflections are more important
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Error during reflection generation: {e}")

    async def retrieve_memories(self, query: str, current_time: datetime.datetime) -> List[str]:
        if not self.memories:
            return []

        query_embedding = None
        if self.embedding_client:
            query_embedding = await self.embedding_client.get_embedding(query)
        
        if query_embedding is None or not self.embedding_client:
            # Fallback to recency if no embeddings
            logger.warning(f"Agent {self.agent_id}: Cannot perform embedding-based retrieval. Falling back to recency.")
            scored_memories = []
            for mem in self.memories:
                 recency_score = self._calculate_recency_score(mem, current_time)
                 score = mem.importance * recency_score # Importance + Recency
                 scored_memories.append((score, mem))
        else:
            # Score based on similarity, recency, and importance
            scored_memories = []
            for mem in self.memories:
                if mem.embedding:
                    similarity_score = cosine_similarity(query_embedding, mem.embedding)
                else:
                    similarity_score = 0.0 # Cannot compare if memory has no embedding
                
                recency_score = self._calculate_recency_score(mem, current_time)
                
                # Combine scores: λ * Recency + (1-λ) * Similarity + small Importance bonus
                # Adjust weights as needed
                combined_score = (self.time_weight_lambda * recency_score + 
                                  (1 - self.time_weight_lambda) * similarity_score + 
                                  mem.importance * 0.1) # Importance as a smaller factor
                
                scored_memories.append((combined_score, mem))

        # Sort by score (descending) and get top K
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        retrieved_memories = scored_memories[:self.retrieval_k]

        # Update last access time for retrieved memories
        retrieved_contents = []
        for score, mem in retrieved_memories:
            mem.update_last_access(current_time)
            retrieved_contents.append(mem.content)
            logger.debug(f"Agent {self.agent_id}: Retrieved memory (score={score:.3f}): {mem}")
            
        return retrieved_contents

    def _calculate_recency_score(self, memory: Memory, current_time: datetime.datetime) -> float:
        # Exponential decay based on last access time
        time_diff_hours = (current_time - memory.last_access_time).total_seconds() / 3600.0
        decay_factor = 0.99 # How much score remains after 1 hour
        return math.pow(decay_factor, time_diff_hours)

    def get_status_for_persistence(self) -> Dict[str, Any]:
        # Return a serializable representation of the memory state
        return {
            "memories": [
                {
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "type": m.type,
                    "importance": m.importance,
                    "last_access_time": m.last_access_time.isoformat()
                    # Embedding is not saved to keep DB size manageable
                } for m in self.memories
            ],
            "memories_since_last_reflection": self.memories_since_last_reflection
        }


