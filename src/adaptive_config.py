"""Adaptive configuration system that automatically adjusts parameters"""
from typing import Dict, List
from src.utils import is_deadline_query, is_scholarship_query


class AdaptiveConfig:
    """Automatically adjusts retrieval and generation parameters based on query characteristics"""
    
    # Base configuration values (can be overridden)
    BASE_RETRIEVAL_K = 5
    BASE_MAX_TOKENS = 10000
    BASE_SIMILARITY_THRESHOLD = 0.1
    BASE_MEMORY_HISTORY = 3
    
    @staticmethod
    def calculate_retrieval_k(query: str, enhanced_query: Dict) -> int:
        """Automatically determine how many documents to retrieve"""
        base_k = AdaptiveConfig.BASE_RETRIEVAL_K
        
        # Increase for complex queries
        complexity_score = 0
        
        # Scholarship/deadline queries need more documents
        if is_scholarship_query(query) or is_deadline_query(query):
            complexity_score += 2
        
        # Person queries with multiple name variations
        if enhanced_query.get('is_person_query', False):
            expanded_count = len(enhanced_query.get('expanded_queries', []))
            if expanded_count > 5:
                complexity_score += 1
        
        # Program queries with course codes
        if enhanced_query.get('is_program_query', False):
            expanded_count = len(enhanced_query.get('expanded_queries', []))
            if expanded_count > 3:
                complexity_score += 1
        
        # Long queries might need more context
        if len(query.split()) > 15:
            complexity_score += 1
        
        # Calculate adaptive retrieval_k
        adaptive_k = base_k + complexity_score
        
        # Cap at reasonable maximum
        return min(adaptive_k, 15)
    
    @staticmethod
    def calculate_similarity_threshold(query: str, retrieved_docs: List[Dict]) -> float:
        """Automatically adjust similarity threshold based on retrieved document quality"""
        base_threshold = AdaptiveConfig.BASE_SIMILARITY_THRESHOLD
        
        if not retrieved_docs:
            return base_threshold
        
        # Calculate average similarity of top documents
        top_similarities = [doc.get('similarity', 0) for doc in retrieved_docs[:5]]
        avg_similarity = sum(top_similarities) / len(top_similarities) if top_similarities else 0
        
        # If documents have low similarity, lower threshold to get more results
        if avg_similarity < 0.3:
            return 0.05  # Very permissive
        elif avg_similarity < 0.5:
            return 0.08  # Permissive
        elif avg_similarity < 0.7:
            return base_threshold  # Standard
        else:
            return 0.15  # Stricter for high-quality matches
    
    @staticmethod
    def calculate_max_tokens(context: str, query: str) -> int:
        """Automatically adjust max_tokens based on context size and query complexity"""
        base_tokens = AdaptiveConfig.BASE_MAX_TOKENS
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        context_tokens = len(context) // 4
        query_tokens = len(query) // 4
        
        # Base response tokens
        base_response_tokens = 200
        
        # Complex queries need longer responses
        complexity_multiplier = 1.0
        if len(query.split()) > 10:
            complexity_multiplier = 1.3
        if is_scholarship_query(query) or is_deadline_query(query):
            complexity_multiplier = 1.5  # Scholarship info needs more detail
        
        # Calculate needed tokens
        needed_tokens = int((base_response_tokens + query_tokens) * complexity_multiplier)
        
        # Ensure minimum response length
        min_tokens = 500
        max_tokens = min(needed_tokens, 2000)  # Cap at 2000
        
        return max(min_tokens, max_tokens)
    
    @staticmethod
    def calculate_memory_history_length(conversation_length: int, query_complexity: int) -> int:
        """Automatically adjust how much conversation history to include"""
        base_history = AdaptiveConfig.BASE_MEMORY_HISTORY
        
        # For longer conversations, include more history
        if conversation_length > 10:
            return min(base_history + 2, 7)
        elif conversation_length > 5:
            return min(base_history + 1, 5)
        else:
            return base_history
    
    @staticmethod
    def calculate_documents_to_use(retrieved_docs: List[Dict], query: str) -> int:
        """Automatically determine how many retrieved documents to use in context"""
        base_k = AdaptiveConfig.BASE_RETRIEVAL_K
        
        # Count high-quality documents (similarity > 0.5)
        high_quality_count = sum(1 for doc in retrieved_docs if doc.get('similarity', 0) > 0.5)
        
        # For scholarship/deadline queries, use more documents
        if is_scholarship_query(query) or is_deadline_query(query):
            # Use more if we have high-quality matches
            if high_quality_count >= 3:
                return min(high_quality_count + 2, 12)
            else:
                return min(base_k * 2, 10)
        
        # For normal queries, use fewer but higher quality
        if high_quality_count >= 3:
            return min(high_quality_count, base_k + 2)
        else:
            return base_k
    
    @staticmethod
    def should_expand_query(enhanced_query: Dict) -> bool:
        """Determine if query expansion is needed"""
        # Always expand person/program queries
        if enhanced_query.get('is_person_query', False) or enhanced_query.get('is_program_query', False):
            return True
        
        # Expand if query is very short (might need more context)
        original = enhanced_query.get('original', '')
        if len(original.split()) < 3:
            return True
        
        return False
    
    @staticmethod
    def get_adaptive_config(query: str, enhanced_query: Dict, retrieved_docs: List[Dict], 
                           context: str, conversation_length: int = 0) -> Dict:
        """Get all adaptive configuration values for a query"""
        return {
            'retrieval_k': AdaptiveConfig.calculate_retrieval_k(query, enhanced_query),
            'similarity_threshold': AdaptiveConfig.calculate_similarity_threshold(query, retrieved_docs),
            'max_tokens': AdaptiveConfig.calculate_max_tokens(context, query),
            'memory_history': AdaptiveConfig.calculate_memory_history_length(conversation_length, 0),
            'documents_to_use': AdaptiveConfig.calculate_documents_to_use(retrieved_docs, query),
            'should_expand': AdaptiveConfig.should_expand_query(enhanced_query)
        }

