"""LLM provider module for API interactions"""
from typing import List, Dict, Optional, Iterator
from openai import OpenAI
import hashlib
import json


class LLMProvider:
    """Manages LLM API interactions with performance tracking and caching"""

    def __init__(self, provider: str = "deepseek", api_key: str = None, temperature: float = 0.5, 
                 max_tokens: int = 10000, enable_cache: bool = True):
        self.provider = provider.lower()
        self.temperature = temperature  # Lower temperature (0.5) for faster, more deterministic responses
        self.max_tokens = max_tokens  # Allow longer responses
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com"
        self.client = None
        self.enable_cache = enable_cache
        self._cache = {}  # Simple in-memory cache
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize the LLM provider client"""
        if self.provider == "deepseek":
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.model_name = "deepseek-chat"
            print(f"Initialized DeepSeek: {self.model_name}")

    def _get_cache_key(self, messages: List[Dict]) -> str:
        """Generate cache key from messages"""
        cache_data = json.dumps(messages, sort_keys=True)
        return hashlib.md5(cache_data.encode()).hexdigest()

    def generate_response(self, prompt: str, system_message: Optional[str] = None,
                         conversation_history: List[Dict] = None, use_cache: bool = True) -> str:
        """Generate response with optional conversation history and caching"""
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        if conversation_history:
            for exchange in conversation_history:
                messages.append({"role": "user", "content": exchange.get('user_query', '')})
                messages.append({"role": "assistant", "content": exchange.get('bot_response', '')})

        messages.append({"role": "user", "content": prompt})

        # Check cache (only for queries without conversation history for accuracy)
        if self.enable_cache and use_cache and not conversation_history:
            cache_key = self._get_cache_key(messages)
            if cache_key in self._cache:
                return self._cache[cache_key]

        request_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = self.client.chat.completions.create(**request_params)
        result = response.choices[0].message.content or ""
        
        # Cache result (only for queries without conversation history)
        if self.enable_cache and use_cache and not conversation_history:
            cache_key = self._get_cache_key(messages)
            self._cache[cache_key] = result
            # Limit cache size to prevent memory issues
            if len(self._cache) > 100:
                # Remove oldest entry (simple FIFO)
                first_key = next(iter(self._cache))
                del self._cache[first_key]
        
        return result

    def generate_response_stream(self, prompt: str, system_message: Optional[str] = None,
                                 conversation_history: List[Dict] = None) -> Iterator[str]:
        """Generate streaming response for faster perceived performance"""
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        if conversation_history:
            for exchange in conversation_history:
                messages.append({"role": "user", "content": exchange.get('user_query', '')})
                messages.append({"role": "assistant", "content": exchange.get('bot_response', '')})

        messages.append({"role": "user", "content": prompt})

        request_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }

        stream = self.client.chat.completions.create(**request_params)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

