"""RAG chatbot module"""
import time
import numpy as np
from typing import List, Dict, Tuple
from src.query_enhancer import QueryEnhancer
from src.memory import ConversationMemory
from src.vector_db import ChromaDBManager
from src.llm_provider import LLMProvider
from src.retrieval import HybridRetriever
from src.prompts import build_system_message, build_user_prompt
from src.utils import get_current_datetime_info, is_deadline_query
from src.adaptive_config import AdaptiveConfig


class RAGChatbot:
    """RAG-based chatbot with performance tracking"""

    def __init__(self, chroma_db: ChromaDBManager, llm_provider: LLMProvider, 
                 use_adaptive_config: bool = True):
        self.db = chroma_db
        self.llm = llm_provider
        self.memory = ConversationMemory(max_history=10)
        self.query_enhancer = QueryEnhancer()
        self.use_adaptive_config = use_adaptive_config
        
        # Base retrieval_k (will be adjusted adaptively if enabled)
        base_retrieval_k = AdaptiveConfig.BASE_RETRIEVAL_K if use_adaptive_config else 5
        self.retriever = HybridRetriever(chroma_db=chroma_db, retrieval_k=base_retrieval_k)
        self.retrieval_k = base_retrieval_k


        # Initialize metrics tracking
        self.session_metrics = []

        print(f"RAG Chatbot initialized with {self.db.collection.count()} documents")
        if use_adaptive_config:
            print("✅ Adaptive configuration enabled - parameters will adjust automatically")

    def retrieve_context(self, query: str, use_memory: bool = True) -> Tuple[List[Dict], str, Dict]:
        """Enhanced retrieval with query preprocessing"""
        enhanced_query = self.query_enhancer.enhance_query(query)

        # Get adaptive configuration
        if self.use_adaptive_config:
            adaptive_config = AdaptiveConfig.get_adaptive_config(
                query=query,
                enhanced_query=enhanced_query,
                retrieved_docs=[],  # Will be updated after retrieval
                context="",
                conversation_length=len(self.memory.history)
            )
            # Update retriever's retrieval_k if needed
            if adaptive_config['retrieval_k'] != self.retriever.retrieval_k:
                self.retriever.retrieval_k = adaptive_config['retrieval_k']
        else:
            adaptive_config = None

        if use_memory and len(self.memory.history) > 0:
            memory_history_n = adaptive_config['memory_history'] if adaptive_config else 2
            memory_context = self.memory.format_for_context(n=memory_history_n)
            enhanced_query['original'] = f"{memory_context}\nCurrent question: {enhanced_query['original']}"

        retrieved_docs = self.retriever.hybrid_retrieval(enhanced_query, use_memory)
        
        # Adaptive filtering based on document quality
        if self.use_adaptive_config and retrieved_docs:
            # Recalculate with actual retrieved docs
            adaptive_config = AdaptiveConfig.get_adaptive_config(
                query=query,
                enhanced_query=enhanced_query,
                retrieved_docs=retrieved_docs,
                context="",
                conversation_length=len(self.memory.history)
            )
            threshold = adaptive_config['similarity_threshold']
            k = adaptive_config['documents_to_use']
        else:
            # Fallback to fixed values
            is_deadline = is_deadline_query(query)
            threshold = 0.05 if is_deadline else 0.1
            k = self.retrieval_k * 2 if is_deadline else self.retrieval_k
        
        filtered_docs = [d for d in retrieved_docs if d.get('retrieval_score', 0) >= threshold]
        top_results = filtered_docs[:k]

        context_parts = []
        # Limit individual document length to prevent excessive context
        max_doc_length = 2000  # characters per document
        
        for result in top_results:
            section = result['metadata'].get('section', 'Unknown Section')
            content = result['document']
            rank = result['rank']
            
            # Truncate very long documents (keep first part which is usually most relevant)
            if len(content) > max_doc_length:
                content = content[:max_doc_length] + "... [truncated]"
            
            context_parts.append(f"[Document {rank} - {section}] (Score: {result['retrieval_score']:.3f})\n{content}")

        context_string = "\n\n---\n\n".join(context_parts)
        
        # Limit total context size (approximately 8000 chars = ~2000 tokens)
        max_total_context = 8000
        if len(context_string) > max_total_context:
            # Keep the highest scoring documents
            context_parts = context_parts[:min(len(context_parts), 5)]
            context_string = "\n\n---\n\n".join(context_parts)
        
        return top_results, context_string, enhanced_query

    def generate_response(self, query: str, context: str, use_memory: bool = True) -> str:
        """Generate response with memory context"""
        # Get current date and time information
        dt_info = get_current_datetime_info()
        
        # Build prompts using prompt templates
        system_message = build_system_message(dt_info)
        user_prompt = build_user_prompt(query, context, dt_info)

        # Adaptive memory history length
        if self.use_adaptive_config and use_memory:
            adaptive_config = AdaptiveConfig.get_adaptive_config(
                query=query,
                enhanced_query={},
                retrieved_docs=[],
                context=context,
                conversation_length=len(self.memory.history)
            )
            memory_n = adaptive_config['memory_history']
        else:
            memory_n = 3
        
        conversation_history = self.memory.get_recent_history(n=memory_n) if use_memory else None

        # Adaptive max_tokens
        if self.use_adaptive_config:
            adaptive_config = AdaptiveConfig.get_adaptive_config(
                query=query,
                enhanced_query={},
                retrieved_docs=[],
                context=context,
                conversation_length=len(self.memory.history)
            )
            original_max_tokens = self.llm.max_tokens
            self.llm.max_tokens = adaptive_config['max_tokens']
        
        try:
            response_text = self.llm.generate_response(
                prompt=user_prompt,
                system_message=system_message,
                conversation_history=conversation_history
            )
        finally:
            # Restore original max_tokens if it was changed
            if self.use_adaptive_config:
                self.llm.max_tokens = original_max_tokens

        return response_text

    def chat(self, query: str, use_memory: bool = True) -> Dict:
        """Process a chat query with performance tracking"""
        print(f"\n🤔 Processing query: '{query}'")

        # Start timing
        start_time = time.time()

        # Retrieve context
        retrieval_start = time.time()
        retrieved_docs, context, enhanced_query = self.retrieve_context(query, use_memory=use_memory)
        retrieval_time = time.time() - retrieval_start

        # Generate response
        generation_start = time.time()
        if not context:
            response_text = "I couldn't find any relevant information in the admission documents."
        else:
            response_text = self.generate_response(query, context, use_memory=use_memory)
        generation_time = time.time() - generation_start

        # Calculate total response time
        total_time = time.time() - start_time

        # Track metrics
        query_category = self.query_enhancer.categorize_query(query)
        similarities = [doc.get('similarity', 0) for doc in retrieved_docs]

        metric = {
            'query': query,
            'category': query_category,
            'hit': len(retrieved_docs) > 0,
            'avg_similarity': np.mean(similarities) if similarities else 0,
            'max_similarity': max(similarities) if similarities else 0,
            'min_similarity': min(similarities) if similarities else 0,
            'num_docs': len(retrieved_docs),
            'response_time': total_time,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time
        }
        self.session_metrics.append(metric)

        print(f"⏱️ Response time: {total_time:.3f}s (Retrieval: {retrieval_time:.3f}s, Generation: {generation_time:.3f}s)")

        # Update memory
        context_ids = [doc['id'] for doc in retrieved_docs]
        self.memory.add_exchange(query, response_text, context_ids)

        return {
            'query': query,
            'answer': response_text,
            'sources': retrieved_docs,
            'memory_used': use_memory,
            'enhanced_query': enhanced_query,
            'performance': {
                'total_time': total_time,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time
            }
        }

