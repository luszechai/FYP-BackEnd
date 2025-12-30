"""LLM provider module for API interactions"""
from typing import List, Dict, Optional
from openai import OpenAI


class LLMProvider:
    """Manages LLM API interactions with performance tracking"""

    def __init__(self, provider: str = "deepseek", api_key: str = None, temperature: float = 0.7):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = 1000
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com"
        self.client = None
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize the LLM provider client"""
        if self.provider == "deepseek":
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.model_name = "deepseek-chat"
            print(f"Initialized DeepSeek: {self.model_name}")

    def generate_response(self, prompt: str, system_message: Optional[str] = None,
                         conversation_history: List[Dict] = None) -> str:
        """Generate response with optional conversation history"""
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
            "max_tokens": self.max_tokens
        }

        response = self.client.chat.completions.create(**request_params)
        return response.choices[0].message.content or ""

