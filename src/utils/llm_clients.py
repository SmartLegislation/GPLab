from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import json

from .logger import get_logger

logger = get_logger(__name__)

class LLMConfig(BaseModel):
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    description: Optional[str] = None
    weight: Optional[float] = 1.0                                                 # Load balancing weight for this API

class OpenaiModel:
    def __init__(self, config: LLMConfig):
        self.config = config
        # Initialize token counters
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        
        if not self.config.api_key or not self.config.base_url:
            logger.warning(f"API key or base URL missing for model {self.config.model}. Client not initialized.")
            self.aclient = None
        else:
            try:
                self.aclient = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client for {self.config.model}: {e}")
                self.aclient = None
        logger.info(f"Initialized LLM client for: {self.config.description or self.config.model}")

    async def async_text_query(self,
                               query: str,
                               system_prompt: Optional[str] = None,
                               hyper_params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if not self.aclient:
            logger.error(f"LLM client for {self.config.model} not initialized. Cannot perform query.")
            return None

        if hyper_params is None:
            hyper_params = {}

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        query_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": hyper_params.get("temperature", self.config.temperature),
            "top_p": hyper_params.get("top_p", self.config.top_p),
            **{k: v for k, v in hyper_params.items() if k not in ["temperature", "top_p"]}
        }
        query_params = {k: v for k, v in query_params.items() if v is not None}

        try:
            logger.debug(f"Sending query to {self.config.model}. Params: {query_params}")
            response = await self.aclient.chat.completions.create(**query_params)
            result = response.choices[0].message.content
            
            # Update token counters
            if hasattr(response, 'usage'):
                self.prompt_tokens += response.usage.prompt_tokens
                self.completion_tokens += response.usage.completion_tokens
                self.total_tokens += response.usage.total_tokens
                logger.debug(f"Updated token count: +{response.usage.total_tokens} tokens")
            
            logger.debug(f"Received response from {self.config.model}: {result[:100]}...")
            return result
        except Exception as e:
            logger.error(f"Error during LLM query to {self.config.model}: {e}")
            return None

    async def evaluate_emotion(self, history: str) -> str:
        prompt = f"Based on the following recent memory and actions, describe the current emotional state in one sentence:\n\n{history}\n\nEmotional State:"
        emotion = await self.async_text_query(prompt)
        return emotion if emotion else "Neutral"

    async def generate_reflection(self, memories: List[str]) -> List[str]:
        memory_str = "\n".join([f"- {m}" for m in memories])
        prompt = f"Given the following observations:\n{memory_str}\n\nGenerate 2 more general, high-level insights or reflections based on these observations."
        reflection_text = await self.async_text_query(prompt)
        if not reflection_text:
            return []
        # Simple split, assuming LLM returns reflections separated by newlines
        reflections = [r.strip() for r in reflection_text.split('\n') if r.strip() and not r.strip().startswith("Here are") and not r.strip().startswith("1.") and not r.strip().startswith("2.")]
        return reflections[:2] # Return max 2
    
    def get_token_stats(self) -> Dict[str, int]:
        """Return current token usage statistics"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }
    
    def reset_token_counters(self):
        """Reset all token counters to zero"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

async def parse_json_response(llm_client: OpenaiModel, raw_response: str, max_retries: int = 2) -> Optional[Dict]:
    current_try = 0
    while current_try <= max_retries:
        try:
            # Try to find JSON block
            start_index = raw_response.find('```json')
            end_index = raw_response.find('```', start_index + 7)
            if start_index != -1 and end_index != -1:
                json_str = raw_response[start_index + 7:end_index].strip()
            else:
                 # If no markdown block, assume the whole response might be JSON
                 json_str = raw_response.strip()

            # Remove potential comments or leading/trailing text if not in a block
            if not (start_index != -1 and end_index != -1):
                first_brace = json_str.find('{')
                last_brace = json_str.rfind('}')
                if first_brace != -1 and last_brace != -1:
                    json_str = json_str[first_brace:last_brace+1]

            parsed_json = json.loads(json_str)
            logger.debug(f"Successfully parsed JSON response.")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {current_try + 1}/{max_retries + 1}: Failed to parse JSON response: {e}. Raw response: {raw_response[:200]}...")
            if current_try < max_retries:
                # Ask LLM to fix the JSON
                logger.info(f"Asking LLM to fix JSON format.")
                fix_prompt = f"The following text is not valid JSON. Please correct it and return only the valid JSON object:\n\n{raw_response}"
                fixed_response = await llm_client.async_text_query(fix_prompt)
                if fixed_response:
                    raw_response = fixed_response
                else:
                    logger.error("LLM failed to provide a fixed JSON response.")
                    return None # Could not get a fix
            current_try += 1
        except Exception as e:
            logger.error(f"An unexpected error occurred during JSON parsing: {e}")
            return None

    logger.error(f"Failed to parse JSON response after {max_retries + 1} attempts.")
    return None





