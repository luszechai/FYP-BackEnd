"""Utility functions for the chatbot"""
from datetime import datetime
from typing import Dict
import re


def get_current_datetime_info() -> Dict[str, str]:
    """Get current date and time information in various formats"""
    now = datetime.now()
    return {
        'full_datetime': now.strftime("%A, %B %d, %Y at %I:%M %p"),
        'date': now.strftime("%Y-%m-%d"),
        'time_24h': now.strftime("%H:%M:%S"),
        'time_12h': now.strftime("%I:%M %p"),
        'day_of_week': now.strftime("%A"),
        'month_name': now.strftime("%B"),
        'year': now.strftime("%Y"),
        'iso_format': now.isoformat()
    }


def is_deadline_query(query: str) -> bool:
    """Check if query is about deadlines or time-sensitive information"""
    query_lower = query.lower()
    deadline_keywords = ['deadline', 'due date', 'when', 'application date', 'scholarship']
    return any(keyword in query_lower for keyword in deadline_keywords)


def is_scholarship_query(query: str) -> bool:
    """Check if query is about scholarships"""
    query_lower = query.lower()
    scholarship_keywords = ['scholarship', 'deadline', 'due date', 'application date', 'when']
    return any(keyword in query_lower for keyword in scholarship_keywords)


def should_skip_retrieval(query: str) -> bool:
    """Determine if retrieval should be skipped for simple/non-informative queries"""
    query_lower = query.strip().lower()
    
    # Very short queries (1-2 words) that are likely greetings or acknowledgments
    simple_responses = {
        'ok', 'okay', 'yes', 'no', 'thanks', 'thank you', 'thx', 
        'sure', 'alright', 'fine', 'good', 'nice', 'cool', 'great',
        'hi', 'hello', 'hey', 'bye', 'goodbye', 'see you',
        'yep', 'nope', 'yeah', 'nah', 'uh huh', 'hmm', 'hmmm'
    }
    
    # Check if query is just a simple response
    if query_lower in simple_responses:
        return True
    
    # Check if query is too short (less than 3 characters) and not a question
    if len(query_lower) < 3:
        return True
    
    # Check if query is just punctuation or whitespace
    if not query_lower or query_lower.strip() == '':
        return True
    
    # Check if query is just numbers or special characters
    if re.match(r'^[\d\s\W]+$', query_lower):
        return True
    
    return False

