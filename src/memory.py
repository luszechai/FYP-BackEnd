"""Conversation memory management module"""
from collections import deque
from typing import List, Dict
from datetime import datetime


class ConversationMemory:
    """Manages short-term conversation memory for the chatbot"""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = deque(maxlen=max_history)

    def add_exchange(self, user_query: str, bot_response: str, context_used: List[str] = None):
        """Add a conversation exchange to memory"""
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'bot_response': bot_response,
            'context_ids': context_used or []
        })

    def get_recent_history(self, n: int = 5) -> List[Dict]:
        """Get the n most recent conversation exchanges"""
        return list(self.history)[-n:]

    def format_for_context(self, n: int = 3) -> str:
        """Format recent history for inclusion in prompts"""
        recent = self.get_recent_history(n)
        if not recent:
            return ""

        formatted = "Recent conversation history:\n"
        for exchange in recent:
            formatted += f"User: {exchange['user_query']}\n"
            # Removed truncation to preserve full context for anaphora resolution
            formatted += f"Assistant: {exchange['bot_response']}\n\n"
        return formatted

    def clear(self):
        """Clear conversation history"""
        self.history.clear()

