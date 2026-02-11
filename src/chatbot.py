"""RAG chatbot module with query intelligence integration"""
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from src.query_enhancer import QueryEnhancer
from src.memory import ConversationMemory
from src.vector_db import ChromaDBManager
from src.llm_provider import LLMProvider
from src.retrieval import HybridRetriever
from src.prompts import build_system_message, build_user_prompt, build_context_aware_prompts, get_follow_up_suggestion
from src.utils import get_current_datetime_info, is_deadline_query, should_skip_retrieval
from src.adaptive_config import AdaptiveConfig
from src.query_intelligence import QueryIntelligence, QueryClassification, QueryType, UserProfile


class RAGChatbot:
    """RAG-based chatbot with performance tracking and query intelligence"""

    def __init__(self, chroma_db: ChromaDBManager, llm_provider: LLMProvider, 
                 use_adaptive_config: bool = True, use_query_intelligence: bool = True):
        self.db = chroma_db
        self.llm = llm_provider
        self.memory = ConversationMemory(max_history=10)
        self.query_enhancer = QueryEnhancer()
        self.use_adaptive_config = use_adaptive_config
        self.use_query_intelligence = use_query_intelligence
        
        # Initialize query intelligence
        self.query_intelligence = QueryIntelligence()
        
        # Base retrieval_k (will be adjusted adaptively if enabled)
        base_retrieval_k = AdaptiveConfig.BASE_RETRIEVAL_K if use_adaptive_config else 5
        self.retriever = HybridRetriever(chroma_db=chroma_db, retrieval_k=base_retrieval_k)
        self.retrieval_k = base_retrieval_k

        # Initialize metrics tracking
        self.session_metrics = []
        
        # Session file storage for user-uploaded documents
        self.session_files: List[Dict] = []
        self.MAX_SESSION_FILES = 5
        self.MAX_FILE_CHARS = 15000
        self.MAX_TOTAL_FILE_CONTEXT = 30000
        
        # Store last classification for use in streaming
        self.last_classification: Optional[QueryClassification] = None

        print(f"RAG Chatbot initialized with {self.db.collection.count()} documents")
        if use_adaptive_config:
            print("✅ Adaptive configuration enabled - parameters will adjust automatically")
        if use_query_intelligence:
            print("✅ Query intelligence enabled - context-aware responses active")

    # ---- Session file management ----

    def add_session_file(self, file_id: str, filename: str, content: str) -> None:
        """Add an uploaded file to the session file store"""
        if len(self.session_files) >= self.MAX_SESSION_FILES:
            raise ValueError(f"Maximum of {self.MAX_SESSION_FILES} session files reached. Remove a file before uploading a new one.")
        
        # Truncate content if needed
        truncated = False
        if len(content) > self.MAX_FILE_CHARS:
            content = content[:self.MAX_FILE_CHARS]
            truncated = True
        
        self.session_files.append({
            'id': file_id,
            'filename': filename,
            'content': content,
            'uploaded_at': datetime.now().isoformat(),
            'truncated': truncated
        })
        print(f"📎 Session file added: {filename} ({len(content)} chars, truncated={truncated})")

    def remove_session_file(self, file_id: str) -> bool:
        """Remove a session file by ID. Returns True if found and removed."""
        for i, f in enumerate(self.session_files):
            if f['id'] == file_id:
                removed = self.session_files.pop(i)
                print(f"🗑️ Session file removed: {removed['filename']}")
                return True
        return False

    def get_session_files(self) -> List[Dict]:
        """Return metadata-only list of uploaded session files"""
        return [
            {
                'id': f['id'],
                'filename': f['filename'],
                'uploaded_at': f['uploaded_at'],
                'char_count': len(f['content']),
                'truncated': f['truncated']
            }
            for f in self.session_files
        ]

    def clear_session_files(self) -> None:
        """Remove all session files"""
        count = len(self.session_files)
        self.session_files.clear()
        print(f"🧹 Cleared {count} session file(s)")

    def format_session_file_context(self) -> str:
        """
        Format all uploaded session file text into a context string.
        Enforces per-file and total character limits.
        """
        if not self.session_files:
            return ""
        
        parts = []
        total_chars = 0
        
        for f in self.session_files:
            content = f['content']
            remaining = self.MAX_TOTAL_FILE_CONTEXT - total_chars
            if remaining <= 0:
                parts.append(f"[File: {f['filename']}] (omitted — total context limit reached)")
                break
            if len(content) > remaining:
                content = content[:remaining] + "... [truncated due to total limit]"
            
            parts.append(f"[File: {f['filename']}]\n{content}")
            total_chars += len(content)
        
        return "\n\n---\n\n".join(parts)

    # ---- Retrieval ----

    def retrieve_context(self, query: str, use_memory: bool = True, 
                         query_classification: Optional[QueryClassification] = None) -> Tuple[List[Dict], str, Dict, Optional[QueryClassification]]:
        """Enhanced retrieval with query preprocessing and query intelligence"""
        enhanced_query = self.query_enhancer.enhance_query(query)
        
        # Get query classification if not provided and intelligence is enabled
        if self.use_query_intelligence and query_classification is None:
            # Get conversation history for context inference
            conversation_history = None
            if use_memory and len(self.memory.history) > 0:
                conversation_history = self.memory.get_recent_history(n=5)
            query_classification = self.query_intelligence.analyze(query, conversation_history)
            print(f"📊 Query Intelligence: Type={query_classification.query_type.value}, Profile={query_classification.user_profile.value}, Confidence={query_classification.confidence:.2f}")

        # Skip retrieval for simple/non-informative queries
        if should_skip_retrieval(query):
            print("⏭️ Skipping retrieval for simple query")
            return [], "", enhanced_query, query_classification

        # Get adaptive configuration
        if self.use_adaptive_config:
            has_anaphora = enhanced_query.get('is_anaphora_query', False)
            adaptive_config = AdaptiveConfig.get_adaptive_config(
                query=query,
                enhanced_query=enhanced_query,
                retrieved_docs=[],  # Will be updated after retrieval
                context="",
                conversation_length=len(self.memory.history),
                has_anaphora=has_anaphora
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

        # Merge expanded queries from query intelligence
        if query_classification and query_classification.expanded_queries:
            existing_expanded = set(enhanced_query.get('expanded_queries', []))
            for eq in query_classification.expanded_queries:
                existing_expanded.add(eq)
            enhanced_query['expanded_queries'] = list(existing_expanded)

        # Perform retrieval with query classification context
        retrieved_docs = self.retriever.hybrid_retrieval(enhanced_query, use_memory, query_classification)
        
        # Adaptive filtering based on document quality
        if self.use_adaptive_config and retrieved_docs:
            # Recalculate with actual retrieved docs
            has_anaphora = enhanced_query.get('is_anaphora_query', False)
            adaptive_config = AdaptiveConfig.get_adaptive_config(
                query=query,
                enhanced_query=enhanced_query,
                retrieved_docs=retrieved_docs,
                context="",
                conversation_length=len(self.memory.history),
                has_anaphora=has_anaphora
            )
            threshold = adaptive_config['similarity_threshold']
            k = adaptive_config['documents_to_use']
        else:
            # Fallback to fixed values, adjusted by query type
            is_deadline = is_deadline_query(query)
            if query_classification:
                # Adjust based on query type
                if query_classification.query_type in [QueryType.EXPLORATORY, QueryType.TEMPORAL]:
                    threshold = 0.05
                    k = self.retrieval_k * 3
                elif query_classification.query_type == QueryType.FACTUAL_LOOKUP:
                    threshold = 0.15
                    k = self.retrieval_k
                else:
                    threshold = 0.08 if is_deadline else 0.1
                    k = self.retrieval_k * 2 if is_deadline else self.retrieval_k
            else:
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
        # Allow more context for exploratory queries
        max_total_context = 10000 if (query_classification and query_classification.query_type == QueryType.EXPLORATORY) else 8000
        if len(context_string) > max_total_context:
            # Keep the highest scoring documents
            context_parts = context_parts[:min(len(context_parts), 6)]
            context_string = "\n\n---\n\n".join(context_parts)
        
        # Append session file context if any files are uploaded
        session_file_context = self.format_session_file_context()
        if session_file_context:
            if context_string:
                context_string += "\n\n--- USER-UPLOADED DOCUMENTS ---\n\n" + session_file_context
            else:
                context_string = session_file_context
        
        return top_results, context_string, enhanced_query, query_classification

    def generate_response(self, query: str, context: str, use_memory: bool = True,
                         query_classification: Optional[QueryClassification] = None) -> str:
        """Generate response with memory context and query intelligence"""
        # Get current date and time information
        dt_info = get_current_datetime_info()
        
        # Get user file context for prompt injection
        user_file_context = self.format_session_file_context() if self.session_files else None

        # Build prompts using prompt templates with classification context
        if self.use_query_intelligence and query_classification:
            system_message, user_prompt = build_context_aware_prompts(
                query=query,
                context=context,
                dt_info=dt_info,
                classification=query_classification,
                user_file_context=user_file_context
            )
            print(f"📝 Using context-aware prompts for {query_classification.query_type.value} query")
        else:
            system_message = build_system_message(dt_info)
            user_prompt = build_user_prompt(query, context, dt_info, user_file_context=user_file_context)

        # Adaptive memory history length
        if self.use_adaptive_config and use_memory:
            # Check if query contains anaphora/references
            enhanced_query_check = self.query_enhancer.enhance_query(query)
            has_anaphora = enhanced_query_check.get('is_anaphora_query', False)
            adaptive_config = AdaptiveConfig.get_adaptive_config(
                query=query,
                enhanced_query=enhanced_query_check,
                retrieved_docs=[],
                context=context,
                conversation_length=len(self.memory.history),
                has_anaphora=has_anaphora
            )
            memory_n = adaptive_config['memory_history']
        else:
            memory_n = 3
        
        conversation_history = self.memory.get_recent_history(n=memory_n) if use_memory else None

        # Adaptive max_tokens - adjust based on query type
        if self.use_adaptive_config:
            # Check if query contains anaphora/references
            enhanced_query_check = self.query_enhancer.enhance_query(query)
            has_anaphora = enhanced_query_check.get('is_anaphora_query', False)
            adaptive_config = AdaptiveConfig.get_adaptive_config(
                query=query,
                enhanced_query=enhanced_query_check,
                retrieved_docs=[],
                context=context,
                conversation_length=len(self.memory.history),
                has_anaphora=has_anaphora
            )
            original_max_tokens = self.llm.max_tokens
            
            # Adjust max_tokens based on query type
            base_max_tokens = adaptive_config['max_tokens']
            if query_classification:
                if query_classification.query_type == QueryType.EXPLORATORY:
                    base_max_tokens = int(base_max_tokens * 1.5)  # More tokens for comprehensive answers
                elif query_classification.query_type == QueryType.FACTUAL_LOOKUP:
                    base_max_tokens = int(base_max_tokens * 0.7)  # Fewer tokens for concise answers
                elif query_classification.query_type == QueryType.PROCEDURAL:
                    base_max_tokens = int(base_max_tokens * 1.3)  # More tokens for step-by-step
            
            self.llm.max_tokens = base_max_tokens
        
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
        """Process a chat query with performance tracking and query intelligence"""
        print(f"\n🤔 Processing query: '{query}'")

        # Start timing
        start_time = time.time()

        # Retrieve context with query intelligence
        retrieval_start = time.time()
        retrieved_docs, context, enhanced_query, query_classification = self.retrieve_context(query, use_memory=use_memory)
        retrieval_time = time.time() - retrieval_start
        
        # Store classification for potential streaming use
        self.last_classification = query_classification

        # Generate response
        generation_start = time.time()
        if not context:
            # For simple queries, provide a friendly response without retrieval
            if should_skip_retrieval(query):
                response_text = "How can I help you with SFU admission information?"
            else:
                response_text = "I couldn't find any relevant information in the admission documents."
        else:
            response_text = self.generate_response(query, context, use_memory=use_memory, 
                                                   query_classification=query_classification)
        generation_time = time.time() - generation_start

        # Calculate total response time
        total_time = time.time() - start_time

        # Track metrics with query intelligence info
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
            'generation_time': generation_time,
            # Add query intelligence metrics
            'query_type': query_classification.query_type.value if query_classification else None,
            'user_profile': query_classification.user_profile.value if query_classification else None,
            'classification_confidence': query_classification.confidence if query_classification else None
        }
        self.session_metrics.append(metric)

        print(f"⏱️ Response time: {total_time:.3f}s (Retrieval: {retrieval_time:.3f}s, Generation: {generation_time:.3f}s)")

        # Update memory
        context_ids = [doc['id'] for doc in retrieved_docs]
        self.memory.add_exchange(query, response_text, context_ids)

        # Build response with query intelligence info
        response = {
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
        
        # Add query intelligence data if available
        if query_classification:
            response['query_intelligence'] = self.query_intelligence.to_dict(query_classification)
            
            # Add follow-up suggestion if appropriate
            follow_up = get_follow_up_suggestion(query_classification)
            if follow_up:
                response['suggested_follow_up'] = follow_up

        return response
    
    def get_classification_for_query(self, query: str, use_memory: bool = True) -> Optional[QueryClassification]:
        """Get query classification without performing full retrieval (for streaming)"""
        if not self.use_query_intelligence:
            return None
        
        conversation_history = None
        if use_memory and len(self.memory.history) > 0:
            conversation_history = self.memory.get_recent_history(n=5)
        
        return self.query_intelligence.analyze(query, conversation_history)

