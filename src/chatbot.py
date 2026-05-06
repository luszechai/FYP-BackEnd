"""RAG chatbot module"""
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from src.query_enhancer import QueryEnhancer
from src.memory import ConversationMemory
from src.vector_db import ChromaDBManager
from src.llm_provider import LLMProvider
from src.retrieval import HybridRetriever
from src.reranker import Reranker
from src.bm25_search import BM25Search
from src.prompts import build_system_message, build_user_prompt
from src.utils import get_current_datetime_info, is_deadline_query, should_skip_retrieval
from src.adaptive_config import AdaptiveConfig
from src.query_rewriter import rewrite_query
from src.program_catalog import build_programme_code_reference_doc, is_programme_code_query
from src.scholarship_catalog import build_scholarship_reference_doc, is_scholarship_reference_query


class RAGChatbot:
    """RAG-based chatbot with performance tracking"""

    def __init__(self, chroma_db: ChromaDBManager, llm_provider: LLMProvider,
                 use_adaptive_config: bool = True, use_reranker: bool = True,
                 use_dedup: bool = True, use_bm25: bool = True,
                 use_hybrid: bool = False,
                 use_person_boost: bool = False,
                 reranker: Optional["Reranker"] = None):
        """Initialise a RAG chatbot.

        The strategy flags (use_adaptive_config, use_reranker, use_dedup,
        use_bm25, use_hybrid, use_person_boost) can be toggled per instance so
        the evaluation dashboard can build transient chatbots for one-click
        A/B runs without mutating the shared ``/api/chat`` singleton.

        ``reranker`` may be passed in so evaluation runs reuse an already
        loaded cross-encoder instead of paying the ~1 GB model-load cost
        every time. When ``use_reranker`` is False the reranker is ignored.
        """
        self.db = chroma_db
        self.llm = llm_provider
        self.memory = ConversationMemory(max_history=10)
        self.query_enhancer = QueryEnhancer()
        self.use_adaptive_config = use_adaptive_config
        self.use_dedup = use_dedup
        self.use_bm25 = use_bm25
        self.use_hybrid = use_hybrid
        self.use_person_boost = use_person_boost

        # Build BM25 sparse keyword index for hybrid search
        self.bm25 = BM25Search()
        self.bm25.build_index(chroma_db.collection)

        # Base retrieval_k (will be adjusted adaptively if enabled)
        base_retrieval_k = AdaptiveConfig.BASE_RETRIEVAL_K if use_adaptive_config else 5
        self.retriever = HybridRetriever(
            chroma_db=chroma_db,
            retrieval_k=base_retrieval_k,
            bm25=self.bm25,
            use_bm25=use_bm25,
            use_hybrid=use_hybrid,
            use_person_boost=use_person_boost,
        )
        self.retrieval_k = base_retrieval_k

        # Initialize reranker (cross-encoder for more precise relevance scoring)
        self.use_reranker = use_reranker
        if use_reranker:
            if reranker is not None:
                self.reranker = reranker
            else:
                self.reranker = Reranker(model_name="BAAI/bge-reranker-base", use_fp16=True)
            if not self.reranker.is_available:
                print("[chatbot][warn] reranker failed to load — falling back to bi-encoder scoring only")
                self.use_reranker = False
        else:
            self.reranker = None

        # Initialize metrics tracking
        self.session_metrics = []
        
        # Session file storage for user-uploaded documents
        self.session_files: List[Dict] = []
        self.MAX_SESSION_FILES = 5
        self.MAX_FILE_CHARS = 15000
        self.MAX_TOTAL_FILE_CONTEXT = 30000

        print(f"[chatbot] initialized with {self.db.collection.count()} documents")
        if use_adaptive_config:
            print("[chatbot] adaptive config enabled - parameters will adjust automatically")
        if self.use_reranker:
            print("[chatbot] reranker enabled — documents will be re-scored with cross-encoder")

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
        print(f"[chatbot] session file added: {filename} ({len(content)} chars, truncated={truncated})")

    def remove_session_file(self, file_id: str) -> bool:
        """Remove a session file by ID. Returns True if found and removed."""
        for i, f in enumerate(self.session_files):
            if f['id'] == file_id:
                removed = self.session_files.pop(i)
                print(f"[chatbot] session file removed: {removed['filename']}")
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
        print(f"[chatbot] cleared {count} session file(s)")

    def format_session_file_context(self) -> str:
        """
        Format all uploaded session file text into a context string.
        Enforces per-file and total character limits.
        """
        if not self.session_files:
            return ""
        
        parts = []
        total_chars = 0
        reversed_files = list(reversed(self.session_files))
        
        for i, f in enumerate(reversed_files):
            content = f['content']
            remaining = self.MAX_TOTAL_FILE_CONTEXT - total_chars
            if remaining <= 0:
                parts.append(f"[File: {f['filename']}] (omitted — total context limit reached)")
                break
            if len(content) > remaining:
                content = content[:remaining] + "... [truncated due to total limit]"
            
            if i == 0:
                label = f"[MOST RECENT UPLOAD: {f['filename']}]"
            else:
                label = f"[Earlier Upload: {f['filename']}]"
            
            parts.append(f"{label}\n{content}")
            total_chars += len(content)
        
        return "\n\n---\n\n".join(parts)

    # ---- Retrieval ----

    def retrieve_context(self, query: str, use_memory: bool = True) -> Tuple[List[Dict], str, Dict]:
        """Enhanced retrieval with query preprocessing"""
        # Skip retrieval for simple/non-informative queries (before rewriting)
        if (
            should_skip_retrieval(query)
            and not is_programme_code_query(query)
            and not is_scholarship_reference_query(query)
        ):
            enhanced_query = self.query_enhancer.enhance_query(query)
            print("[chatbot] skipping retrieval for simple query")
            return [], "", enhanced_query

        # Fetch memory context BEFORE rewriting so the rewriter can resolve
        # follow-up references (e.g. "那4年加起來是多少" -> "total 4-year tuition for BSAI")
        memory_context = ""
        if use_memory and len(self.memory.history) > 0:
            # Use a small window for the rewriter prompt to keep it concise
            memory_context = self.memory.format_for_context(n=3)

        # LLM-based query rewriting (always runs: handles non-English, abbreviations, follow-ups)
        rewritten = rewrite_query(self.llm, query, conversation_context=memory_context)
        enhanced_query = self.query_enhancer.enhance_query(rewritten)
        # Keep the original wording available for display
        if rewritten != query:
            enhanced_query['original_raw'] = query
        # Store the rewritten query for downstream use (compressor, reranker)
        enhanced_query['rewritten'] = rewritten

        # Get adaptive configuration
        if self.use_adaptive_config:
            has_anaphora = enhanced_query.get('is_anaphora_query', False)
            adaptive_config = AdaptiveConfig.get_adaptive_config(
                query=query,
                enhanced_query=enhanced_query,
                retrieved_docs=[],
                context="",
                conversation_length=len(self.memory.history),
                has_anaphora=has_anaphora
            )
            if adaptive_config['retrieval_k'] != self.retriever.retrieval_k:
                self.retriever.retrieval_k = adaptive_config['retrieval_k']
        else:
            adaptive_config = None

        # Prepend full memory context to retrieval query
        if use_memory and len(self.memory.history) > 0:
            memory_history_n = adaptive_config['memory_history'] if adaptive_config else 2
            full_memory = self.memory.format_for_context(n=memory_history_n)
            enhanced_query['original'] = f"{full_memory}\nCurrent question: {enhanced_query['original']}"

        retrieved_docs = self.retriever.hybrid_retrieval(
            enhanced_query, use_memory, reranker_mode=self.use_reranker
        )
        
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
            # Fallback to fixed values
            is_deadline = is_deadline_query(query)
            threshold = 0.05 if is_deadline else 0.1
            k = self.retrieval_k * 2 if is_deadline else self.retrieval_k

        # Deduplicate retrieved documents before reranking (keep highest scoring copy)
        if self.use_dedup and retrieved_docs:
            seen = {}
            for doc in retrieved_docs:
                doc_id = doc['id']
                if doc_id not in seen or doc.get('retrieval_score', 0) > seen[doc_id].get('retrieval_score', 0):
                    seen[doc_id] = doc
            retrieved_docs = list(seen.values())

        # Rerank candidates with cross-encoder for more precise relevance scoring
        RERANKER_TOP_K = 10  # Keep the top 10 documents after reranking
        if self.use_reranker and retrieved_docs:
            # Use the raw query (without memory prepended) for reranking
            retrieved_docs = self.reranker.rerank(query, retrieved_docs, top_k=RERANKER_TOP_K)
            # After reranking, use a lower threshold since scores are normalized [0, 1]
            threshold = max(threshold, 0.05)
            k = RERANKER_TOP_K

        filtered_docs = [d for d in retrieved_docs if d.get('retrieval_score', 0) >= threshold]
        top_results = filtered_docs[:k]

        programme_reference_query = "\n".join(
            value for value in [
                query,
                enhanced_query.get('original_raw', ''),
                enhanced_query.get('rewritten', ''),
            ]
            if value
        )
        programme_reference_doc = build_programme_code_reference_doc(programme_reference_query)
        if programme_reference_doc and not any(d.get('id') == programme_reference_doc['id'] for d in top_results):
            top_results.insert(0, programme_reference_doc)
            enhanced_query['programme_code_reference_injected'] = True

        scholarship_reference_doc = build_scholarship_reference_doc(programme_reference_query)
        if scholarship_reference_doc and not any(d.get('id') == scholarship_reference_doc['id'] for d in top_results):
            insert_at = 1 if programme_reference_doc else 0
            top_results.insert(insert_at, scholarship_reference_doc)
            enhanced_query['scholarship_reference_injected'] = True

        context_parts = []
        max_doc_length = 2000

        for i, result in enumerate(top_results, start=1):
            metadata = dict(result.get('metadata', {}))
            metadata['context_doc_num'] = i
            result['metadata'] = metadata
            section = metadata.get('section', 'Unknown Section')
            if section == 'email':
                email_type = metadata.get('email_type', '')
                if email_type:
                    section = f"email: {email_type}"
            content = result['document']

            if metadata.get('type') not in {'programme_code_reference', 'scholarship_reference'} and len(content) > max_doc_length:
                content = content[:max_doc_length] + "... [truncated]"
            
            context_parts.append(f"[Document {i} - {section}] (Score: {result['retrieval_score']:.3f})\n{content}")

        context_string = "\n\n---\n\n".join(context_parts)
        
        # Limit total context size (approximately 8000 chars = ~2000 tokens)
        max_total_context = 8000
        if len(context_string) > max_total_context:
            # Keep the highest scoring documents
            context_parts = context_parts[:min(len(context_parts), 5)]
            context_string = "\n\n---\n\n".join(context_parts)
        
        # Append session file context if any files are uploaded
        session_file_context = self.format_session_file_context()
        if session_file_context:
            if context_string:
                context_string += "\n\n--- USER-UPLOADED DOCUMENTS ---\n\n" + session_file_context
            else:
                context_string = session_file_context
        
        return top_results, context_string, enhanced_query

    def generate_response(self, query: str, context: str, use_memory: bool = True) -> str:
        """Generate response with memory context"""
        # Get current date and time information
        dt_info = get_current_datetime_info()
        
        # Get user file context for prompt injection
        user_file_context = self.format_session_file_context() if self.session_files else None

        # Build prompts using prompt templates
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

        # Adaptive max_tokens
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
        print(f"\n[chatbot] processing query: '{query}'")

        # Start timing
        start_time = time.time()

        # Retrieve context
        retrieval_start = time.time()
        retrieved_docs, context, enhanced_query = self.retrieve_context(query, use_memory=use_memory)
        retrieval_time = time.time() - retrieval_start

        # Generate response
        generation_start = time.time()
        if not context:
            # For simple queries, provide a friendly response without retrieval
            if should_skip_retrieval(query):
                response_text = "How can I help you with SFU admission information?"
            else:
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

        print(f"[chatbot] response time: {total_time:.3f}s (retrieval: {retrieval_time:.3f}s, generation: {generation_time:.3f}s)")

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
