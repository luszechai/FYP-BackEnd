"""LLM provider module for API interactions"""
from typing import List, Dict, Optional, Iterator, Tuple, Union
from openai import OpenAI
import base64
import hashlib
import json
import mimetypes


class LLMProvider:
    """Manages LLM API interactions with performance tracking and caching"""

    PROVIDER_DEFAULTS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
        },
        "kimi": {
            "base_url": "https://api.moonshot.cn/v1",
            "model": "moonshot-v1-8k",
        },
    }

    def __init__(self, provider: str = "deepseek", api_key: str = None, temperature: float = 0.5, 
                 max_tokens: int = 10000, enable_cache: bool = True,
                 base_url: str = None, model: str = None, kimi_disable_thinking: bool = False,
                 request_timeout: float = 300.0):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.kimi_disable_thinking = kimi_disable_thinking
        self.request_timeout = request_timeout
        defaults = self.PROVIDER_DEFAULTS.get(self.provider, {})
        self.base_url = base_url or defaults.get("base_url", "https://api.deepseek.com")
        self.model_name = model or defaults.get("model", "deepseek-chat")
        self.client = None
        self.enable_cache = enable_cache
        self._cache = {}
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize the LLM provider client"""
        if self.provider in ("deepseek", "kimi"):
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.request_timeout)
            label = "DeepSeek" if self.provider == "deepseek" else "Kimi"
            print(f"Initialized {label}: {self.model_name} @ {self.base_url}")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

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
        if self.provider == "kimi" and self.kimi_disable_thinking:
            request_params["extra_body"] = {"thinking": {"type": "disabled"}}

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

    def generate_response_with_images(
        self,
        prompt: str,
        images: List[Tuple[bytes, str]],
        system_message: Optional[str] = None,
    ) -> str:
        """Generate a response using multimodal content (text + images).

        Args:
            prompt: The text prompt.
            images: List of (image_bytes, mime_type) tuples, e.g. (b'...', 'image/png').
            system_message: Optional system message.

        Returns:
            The model's text response.
        """
        messages: List[Dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        content_parts: List[Dict] = [{"type": "text", "text": prompt}]
        for img_bytes, mime in images:
            b64 = base64.b64encode(img_bytes).decode("ascii")
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

        messages.append({"role": "user", "content": content_parts})

        request_params: Dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.provider == "kimi" and self.kimi_disable_thinking:
            request_params["extra_body"] = {"thinking": {"type": "disabled"}}

        response = self.client.chat.completions.create(**request_params)
        return response.choices[0].message.content or ""

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
        if self.provider == "kimi" and self.kimi_disable_thinking:
            request_params["extra_body"] = {"thinking": {"type": "disabled"}}

        stream = self.client.chat.completions.create(**request_params)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

