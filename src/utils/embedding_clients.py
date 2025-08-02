from typing import Any, List, Optional, Tuple, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from openai import AsyncOpenAI # For potential remote embedding API
import numpy as np

from .logger import get_logger
from .llm_clients import LLMConfig # Re-use for API config structure

logger = get_logger(__name__)

class EmbeddingClient:
    def __init__(self, 
                 use_local: bool = True, 
                 local_model_path: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
                 local_model_device: Optional[str] = "cpu",
                 remote_api_config: Optional[LLMConfig] = None,
                 dimension: int = 384):
        self.use_local = use_local
        self.dimension = dimension
        self.local_model = None
        self.remote_client = None
        self.remote_api_config = remote_api_config

        if self.use_local:
            try:
                logger.info(f"Initializing local embedding model: {local_model_path} on device: {local_model_device}")
                self.local_model = HuggingFaceEmbeddings(
                    model_name=local_model_path,
                    model_kwargs={'device': local_model_device}
                )
                # Test query to confirm dimension, or trust config
                # test_embedding = self.local_model.embed_query("test")
                # self.dimension = len(test_embedding)
                logger.info(f"Local embedding model initialized. Dimension: {self.dimension}")
            except Exception as e:
                logger.error(f"Failed to initialize local HuggingFace embedding model: {e}")
                logger.warning("Falling back to no embeddings or remote if configured.")
                self.use_local = False # Fallback if local fails
        
        if not self.use_local and self.remote_api_config:
            if not self.remote_api_config.api_key or not self.remote_api_config.base_url:
                logger.warning(f"API key or base URL missing for remote embedding model {self.remote_api_config.model}. Remote client not initialized.")
            else:
                try:
                    logger.info(f"Initializing remote embedding client for: {self.remote_api_config.description or self.remote_api_config.model}")
                    self.remote_client = AsyncOpenAI(
                        api_key=self.remote_api_config.api_key,
                        base_url=self.remote_api_config.base_url
                    )
                    logger.info(f"Remote embedding client initialized. Assumed Dimension: {self.dimension}")
                except Exception as e:
                    logger.error(f"Failed to initialize remote embedding client: {e}")
                    self.remote_client = None
        elif not self.use_local and not self.remote_api_config:
            logger.warning("No local embedding model and no remote embedding API configured.")

    async def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        if self.use_local and self.local_model:
            try:
                return self.local_model.embed_documents(texts)
            except Exception as e:
                logger.error(f"Error generating embeddings with local model: {e}")
                return None
        elif not self.use_local and self.remote_client and self.remote_api_config:
            try:
                response = await self.remote_client.embeddings.create(
                    input=texts,
                    model=self.remote_api_config.model
                )
                embeddings = [item.embedding for item in response.data]
                # Ensure all embeddings have the expected dimension by padding or truncating
                processed_embeddings = []
                for emb in embeddings:
                    if len(emb) < self.dimension:
                        processed_embeddings.append(emb + [0.0] * (self.dimension - len(emb)))
                    else:
                        processed_embeddings.append(emb[:self.dimension])
                return processed_embeddings
            except Exception as e:
                logger.error(f"Error generating embeddings with remote API ({self.remote_api_config.model}): {e}")
                return None
        else:
            logger.warning("Embedding model not available.")
            return None

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings and embeddings[0] else None

def get_embedding_client_from_config(config: Dict[str, Any]) -> EmbeddingClient:
    llm_api_configs = config.get('llm_api_configs', [])
    use_local = config.get('use_local_embedding_model', True)
    dimension = config.get('embedding_dimension', 384)
    remote_api_config = None

    if not use_local:
        embedding_api_index = config.get('embedding_api_index')
        if embedding_api_index is not None and 0 <= embedding_api_index < len(llm_api_configs):
            remote_conf_dict = llm_api_configs[embedding_api_index]
            remote_api_config = LLMConfig(**remote_conf_dict) # Validate with Pydantic
        else:
            logger.warning(f"Invalid embedding_api_index: {embedding_api_index}. Cannot configure remote embedding client.")
            # Potentially fall back to local or disable embeddings if no valid remote config
            # For now, if use_local is False and remote is misconfigured, it will warn in EmbeddingClient init.

    return EmbeddingClient(
        use_local=use_local,
        local_model_path=config.get('local_embedding_model_path'),
        local_model_device=config.get('embedding_device'),
        remote_api_config=remote_api_config,
        dimension=dimension
    )

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if v1.shape != v2.shape or v1.size == 0:
        logger.warning(f"Cannot compute cosine similarity for vectors of different shapes or zero size. v1: {v1.shape}, v2: {v2.shape}")
        return 0.0
    
    # Handle potential all-zero vectors that might lead to nan
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0 # Or handle as perfectly dissimilar, or identical if both are zero

    similarity = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return float(similarity)





