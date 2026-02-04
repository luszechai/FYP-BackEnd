"""Retrieval strategies for document search with context-aware enhancements"""
from typing import List, Dict, Optional, Any
from src.vector_db import ChromaDBManager
from src.utils import is_scholarship_query
from src.query_intelligence import QueryIntelligence, QueryClassification, QueryType, UserProfile


class HybridRetriever:
    """Handles hybrid retrieval strategies for document search with context-awareness"""

    def __init__(self, chroma_db: ChromaDBManager, retrieval_k: int = 5):
        self.db = chroma_db
        self.retrieval_k = retrieval_k
        self.query_intelligence = QueryIntelligence()
        
        # Intent-based retrieval configuration
        self.intent_retrieval_config = {
            QueryType.FACTUAL_LOOKUP: {
                'base_k_multiplier': 1.5,
                'similarity_threshold': 0.15,
                'max_docs': 5
            },
            QueryType.EXPLORATORY: {
                'base_k_multiplier': 3.0,
                'similarity_threshold': 0.08,
                'max_docs': 12
            },
            QueryType.COMPARATIVE: {
                'base_k_multiplier': 2.5,
                'similarity_threshold': 0.10,
                'max_docs': 10
            },
            QueryType.PROCEDURAL: {
                'base_k_multiplier': 2.0,
                'similarity_threshold': 0.10,
                'max_docs': 8
            },
            QueryType.ELIGIBILITY: {
                'base_k_multiplier': 2.5,
                'similarity_threshold': 0.08,
                'max_docs': 10
            },
            QueryType.TEMPORAL: {
                'base_k_multiplier': 3.0,
                'similarity_threshold': 0.05,
                'max_docs': 12
            }
        }

    def hybrid_retrieval(self, enhanced_query: Dict, use_memory: bool = True, 
                        query_classification: Optional[QueryClassification] = None) -> List[Dict]:
        """Perform hybrid retrieval with multiple query strategies and context awareness"""
        all_results = {}
        
        # Detect if this is a scholarship/deadline query
        query_lower = enhanced_query['original'].lower()
        is_scholarship = is_scholarship_query(enhanced_query['original'])
        
        # Get retrieval configuration based on query classification
        if query_classification:
            config = self.intent_retrieval_config.get(
                query_classification.query_type,
                self.intent_retrieval_config[QueryType.EXPLORATORY]
            )
            base_multiplier = config['base_k_multiplier']
        else:
            base_multiplier = 3.0 if is_scholarship else 2.0
        
        # Adjust retrieval size based on query type and classification
        base_n_results = int(self.retrieval_k * base_multiplier)

        print(f"🔍 Strategy 1: Original query (k={base_n_results})")
        results = self.db.query(query_text=enhanced_query['original'], n_results=base_n_results)
        for doc in self.db.format_results(results):
            doc_id = doc['id']
            if doc_id not in all_results:
                all_results[doc_id] = doc
                all_results[doc_id]['retrieval_score'] = doc['similarity']
            else:
                all_results[doc_id]['retrieval_score'] = max(
                    all_results[doc_id]['retrieval_score'],
                    doc['similarity']
                )
        
        # Strategy 1.5: Intent-based expanded queries from query intelligence
        if query_classification and query_classification.expanded_queries:
            print(f"🔍 Strategy 1.5: Intent-based expanded queries ({len(query_classification.expanded_queries)} queries)")
            for exp_query in query_classification.expanded_queries[:5]:
                if exp_query != enhanced_query['original']:  # Skip if same as original
                    results = self.db.query(query_text=exp_query, n_results=self.retrieval_k)
                    for doc in self.db.format_results(results):
                        doc_id = doc['id']
                        if doc_id not in all_results:
                            all_results[doc_id] = doc
                            all_results[doc_id]['retrieval_score'] = doc['similarity'] * 0.88
                        else:
                            all_results[doc_id]['retrieval_score'] = max(
                                all_results[doc_id]['retrieval_score'],
                                doc['similarity'] * 0.88
                            )
        
        # Strategy 1.6: Scholarship-specific queries (enhanced for listing queries)
        is_scholarship_enhanced = is_scholarship or enhanced_query.get('is_scholarship_query', False)
        if is_scholarship_enhanced:
            print(f"🔍 Strategy 1.6: Scholarship queries (listing + deadline)")
            
            # Check if this is a listing query (asking for available scholarships)
            list_patterns = ['list', 'what', 'which', 'available', 'types of', 
                            'show', 'tell me', 'give me', 'all', 'any']
            is_list_query = any(p in query_lower for p in list_patterns)
            
            # Scholarship search terms - always search these for scholarship queries
            scholarship_search_terms = [
                'scholarship_name', 'scholarships available', 
                'entrance scholarships', 'admission scholarships',
                'Academic Achievement Scholarships', 'scholarship eligibility'
            ]
            
            # Search for scholarship-related terms (regardless of whether they're in query)
            for term in scholarship_search_terms:
                results = self.db.query(query_text=term, n_results=self.retrieval_k * 2)
                for doc in self.db.format_results(results):
                    doc_id = doc['id']
                    # Higher score for listing queries
                    score_multiplier = 0.90 if is_list_query else 0.85
                    if doc_id not in all_results:
                        all_results[doc_id] = doc
                        all_results[doc_id]['retrieval_score'] = doc['similarity'] * score_multiplier
                    else:
                        all_results[doc_id]['retrieval_score'] = max(
                            all_results[doc_id]['retrieval_score'],
                            doc['similarity'] * score_multiplier
                        )
            
            # Additional deadline-specific terms if deadline keywords present
            deadline_terms = ['scholarship deadline', 'application deadline', 'due date', 
                             'scholarship application', 'deadline for']
            if any(term in query_lower for term in ['deadline', 'due', 'when', 'date']):
                for term in deadline_terms:
                    results = self.db.query(query_text=term, n_results=self.retrieval_k * 2)
                    for doc in self.db.format_results(results):
                        doc_id = doc['id']
                        if doc_id not in all_results:
                            all_results[doc_id] = doc
                            all_results[doc_id]['retrieval_score'] = doc['similarity'] * 0.85
                        else:
                            all_results[doc_id]['retrieval_score'] = max(
                                all_results[doc_id]['retrieval_score'],
                                doc['similarity'] * 0.85
                            )
            
            # Keyword boosting for scholarship documents
            scholarship_keywords = ['scholarship_name', 'scholarships', 'award', 
                                   'bursary', 'financial aid', 'eligibility', 'entrance_scholarships']
            for doc_id, doc in all_results.items():
                content_lower = doc['document'].lower()
                keyword_matches = sum(1 for sk in scholarship_keywords if sk in content_lower)
                if keyword_matches > 0:
                    boost = min(0.15, keyword_matches * 0.04)
                    doc['retrieval_score'] = min(1.0, doc['retrieval_score'] + boost)

        if enhanced_query['is_person_query'] and len(enhanced_query['expanded_queries']) > 1:
            print(f"🔍 Strategy 2: Expanded person queries ({len(enhanced_query['expanded_queries'])} variations)")

            for exp_query in enhanced_query['expanded_queries'][:3]:
                results = self.db.query(query_text=exp_query, n_results=self.retrieval_k)
                for doc in self.db.format_results(results):
                    doc_id = doc['id']
                    if doc_id not in all_results:
                        all_results[doc_id] = doc
                        all_results[doc_id]['retrieval_score'] = doc['similarity'] * 0.9
                    else:
                        all_results[doc_id]['retrieval_score'] = max(
                            all_results[doc_id]['retrieval_score'],
                            doc['similarity'] * 0.9
                        )

        # Strategy 2.3: Expanded Role Queries (for queries like "who is the programme leader of AI")
        if enhanced_query.get('is_role_query', False) and len(enhanced_query['expanded_queries']) > 1:
            print(f"🔍 Strategy 2.3: Expanded role queries ({len(enhanced_query['expanded_queries'])} variations)")

            for exp_query in enhanced_query['expanded_queries'][:5]:
                results = self.db.query(query_text=exp_query, n_results=self.retrieval_k)
                for doc in self.db.format_results(results):
                    doc_id = doc['id']
                    if doc_id not in all_results:
                        all_results[doc_id] = doc
                        all_results[doc_id]['retrieval_score'] = doc['similarity'] * 0.92
                    else:
                        all_results[doc_id]['retrieval_score'] = max(
                            all_results[doc_id]['retrieval_score'],
                            doc['similarity'] * 0.92
                        )

        # Strategy 2.5: Expanded Program Queries
        if enhanced_query.get('is_program_query', False) and len(enhanced_query['expanded_queries']) > 1:
            print(f"🔍 Strategy 2.5: Expanded Program queries ({len(enhanced_query['expanded_queries'])} variations)")

            for exp_query in enhanced_query['expanded_queries'][:4]:
                results = self.db.query(query_text=exp_query, n_results=self.retrieval_k)
                for doc in self.db.format_results(results):
                    doc_id = doc['id']
                    if doc_id not in all_results:
                        all_results[doc_id] = doc
                        all_results[doc_id]['retrieval_score'] = doc['similarity'] * 0.95
                    else:
                        all_results[doc_id]['retrieval_score'] = max(
                            all_results[doc_id]['retrieval_score'],
                            doc['similarity'] * 0.95
                        )

        # Strategy 2.6: Expanded Scholarship Queries
        if enhanced_query.get('is_scholarship_query', False) and len(enhanced_query['expanded_queries']) > 1:
            print(f"🔍 Strategy 2.6: Expanded Scholarship queries ({len(enhanced_query['expanded_queries'])} variations)")

            for exp_query in enhanced_query['expanded_queries'][:6]:
                results = self.db.query(query_text=exp_query, n_results=self.retrieval_k)
                for doc in self.db.format_results(results):
                    doc_id = doc['id']
                    if doc_id not in all_results:
                        all_results[doc_id] = doc
                        all_results[doc_id]['retrieval_score'] = doc['similarity'] * 0.92
                    else:
                        all_results[doc_id]['retrieval_score'] = max(
                            all_results[doc_id]['retrieval_score'],
                            doc['similarity'] * 0.92
                        )

        # Strategy 3: Keyword matching with context boosting
        if enhanced_query['keywords']:
            print(f"🔍 Strategy 3: Keyword matching")
            for doc_id, doc in all_results.items():
                content_lower = doc['document'].lower()

                keyword_matches = sum(1 for kw in enhanced_query['keywords']
                                    if kw.lower() in content_lower)
                
                # Boost for deadline-related keywords in scholarship queries
                deadline_keywords = ['deadline', 'due date', 'application', 'end of','january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'end of application', 'end of submission', 'end of deadline','submission deadline']
                deadline_matches = sum(1 for dk in deadline_keywords if dk in content_lower)
                
                if keyword_matches > 0:
                    if enhanced_query.get('is_program_query', False):
                        boost = min(0.4, keyword_matches * 0.15)
                    elif is_scholarship:
                        # Higher boost for scholarship queries with deadline keywords
                        boost = min(0.5, keyword_matches * 0.15 + deadline_matches * 0.1)
                    else:
                        boost = min(0.3, keyword_matches * 0.1)

                    doc['retrieval_score'] = min(1.0, doc['retrieval_score'] + boost)

        # Strategy 3.5: Role-specific keyword boosting
        if enhanced_query.get('is_role_query', False):
            print(f"🔍 Strategy 3.5: Role-specific keyword boosting")
            role_keywords = ['programme leader', 'program leader', 'director', 'head of', 
                           'role', 'coordinator', 'dean', 'chair']
            for doc_id, doc in all_results.items():
                content_lower = doc['document'].lower()
                # Check if document contains role-related keywords
                role_matches = sum(1 for rk in role_keywords if rk in content_lower)
                if role_matches > 0:
                    boost = min(0.15, role_matches * 0.05)
                    doc['retrieval_score'] = min(1.0, doc['retrieval_score'] + boost)

        # Strategy 4: Context-based document boosting (user profile aware)
        if query_classification:
            all_results = self._apply_context_boosting(
                all_results, 
                query_classification,
                enhanced_query
            )

        sorted_results = sorted(all_results.values(),
                              key=lambda x: x['retrieval_score'],
                              reverse=True)

        # Determine max results based on query type
        if query_classification:
            config = self.intent_retrieval_config.get(
                query_classification.query_type,
                self.intent_retrieval_config[QueryType.EXPLORATORY]
            )
            max_results = config['max_docs']
        else:
            max_results = self.retrieval_k * 3 if is_scholarship else self.retrieval_k * 2
        
        return sorted_results[:max_results]

    def _apply_context_boosting(self, all_results: Dict[str, Dict], 
                                classification: QueryClassification,
                                enhanced_query: Dict) -> Dict[str, Dict]:
        """Apply context-based boosting to documents based on user profile and intents"""
        
        # Get context boost terms from classification
        boost_terms = classification.context_boost_terms
        
        # User profile specific boosting
        profile_boost_patterns = {
            UserProfile.LOCAL_DSE: ['jupas', 'dse', 'hkdse', 'local', 'band a', 'band b', 'band c'],
            UserProfile.INTERNATIONAL: ['international', 'non-local', 'overseas', 'ielts', 'toefl', 'visa'],
            UserProfile.TRANSFER: ['transfer', 'articulation', 'credit', 'sub-degree', 'hd', 'senior year'],
            UserProfile.CURRENT_STUDENT: ['current student', 'enrolled', 'registration'],
            UserProfile.LOCAL_NON_JUPAS: ['non-jupas', 'direct application', 'mature'],
        }
        
        profile_terms = profile_boost_patterns.get(classification.user_profile, [])
        all_boost_terms = list(set(boost_terms + profile_terms))
        
        # Query type specific boosting
        type_boost_patterns = {
            QueryType.TEMPORAL: ['deadline', 'date', 'due', 'application period', 'start', 'end'],
            QueryType.ELIGIBILITY: ['requirement', 'eligible', 'qualify', 'minimum', 'criteria'],
            QueryType.PROCEDURAL: ['step', 'process', 'how to', 'apply', 'submit'],
            QueryType.EXPLORATORY: ['overview', 'program', 'course', 'career', 'employment'],
        }
        
        type_terms = type_boost_patterns.get(classification.query_type, [])
        
        print(f"🔍 Strategy 4: Context boosting (profile: {classification.user_profile.value}, type: {classification.query_type.value})")
        
        for doc_id, doc in all_results.items():
            content_lower = doc['document'].lower()
            metadata = doc.get('metadata', {})
            section_lower = metadata.get('section', '').lower()
            
            boost = 0.0
            
            # Boost for user profile terms
            profile_matches = sum(1 for term in all_boost_terms if term in content_lower or term in section_lower)
            if profile_matches > 0:
                boost += min(0.15, profile_matches * 0.05)
            
            # Boost for query type terms
            type_matches = sum(1 for term in type_terms if term in content_lower)
            if type_matches > 0:
                boost += min(0.1, type_matches * 0.03)
            
            # Boost for intent-related content
            for intent in classification.intents[:5]:
                if intent.lower() in content_lower:
                    boost += 0.03
            
            # Apply boost
            if boost > 0:
                doc['retrieval_score'] = min(1.0, doc['retrieval_score'] + boost)
                doc['context_boosted'] = True
        
        return all_results

    def context_aware_retrieval(self, query: str, conversation_history: Optional[List[Dict]] = None,
                               enhanced_query: Optional[Dict] = None) -> tuple[List[Dict], QueryClassification]:
        """
        Perform context-aware retrieval using query intelligence.
        Returns both retrieved documents and the query classification.
        """
        # Analyze query using query intelligence
        classification = self.query_intelligence.analyze(query, conversation_history)
        
        # If no enhanced_query provided, create a basic one
        if enhanced_query is None:
            enhanced_query = {
                'original': query,
                'is_person_query': False,
                'is_program_query': False,
                'is_anaphora_query': False,
                'expanded_queries': classification.expanded_queries,
                'keywords': query.split()
            }
        else:
            # Merge expanded queries from classification
            existing_expanded = set(enhanced_query.get('expanded_queries', []))
            for eq in classification.expanded_queries:
                existing_expanded.add(eq)
            enhanced_query['expanded_queries'] = list(existing_expanded)
        
        # Perform retrieval with classification context
        docs = self.hybrid_retrieval(enhanced_query, query_classification=classification)
        
        return docs, classification

    def get_retrieval_stats(self, docs: List[Dict], classification: QueryClassification) -> Dict[str, Any]:
        """Get statistics about the retrieval for debugging/logging"""
        if not docs:
            return {
                'num_docs': 0,
                'query_type': classification.query_type.value,
                'user_profile': classification.user_profile.value,
                'avg_score': 0,
                'context_boosted_count': 0
            }
        
        scores = [d.get('retrieval_score', 0) for d in docs]
        context_boosted = sum(1 for d in docs if d.get('context_boosted', False))
        
        return {
            'num_docs': len(docs),
            'query_type': classification.query_type.value,
            'user_profile': classification.user_profile.value,
            'confidence': classification.confidence,
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'context_boosted_count': context_boosted,
            'intents': classification.intents[:5],
            'expanded_queries_used': len(classification.expanded_queries)
        }
