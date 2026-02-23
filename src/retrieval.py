"""Retrieval strategies for document search"""
from typing import List, Dict
from src.vector_db import ChromaDBManager
from src.utils import is_scholarship_query


class HybridRetriever:
    """Handles hybrid retrieval strategies for document search"""

    def __init__(self, chroma_db: ChromaDBManager, retrieval_k: int = 5):
        self.db = chroma_db
        self.retrieval_k = retrieval_k

    def hybrid_retrieval(self, enhanced_query: Dict, use_memory: bool = True) -> List[Dict]:
        """Perform hybrid retrieval with multiple query strategies"""
        all_results = {}
        
        # Detect if this is a scholarship/deadline query
        query_lower = enhanced_query['original'].lower()
        is_scholarship = is_scholarship_query(enhanced_query['original'])
        
        # Adjust retrieval size for scholarship queries
        base_n_results = self.retrieval_k * 3 if is_scholarship else self.retrieval_k * 2

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

        deduplicated = self._deduplicate_results(list(all_results.values()))
        sorted_results = sorted(deduplicated,
                              key=lambda x: x['retrieval_score'],
                              reverse=True)

        # Return more results for scholarship queries
        max_results = self.retrieval_k * 3 if is_scholarship else self.retrieval_k * 2
        return sorted_results[:max_results]

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Two-layer dedup: parent_doc_id then content similarity."""
        # Layer 1: Keep best chunk per parent_doc_id
        by_parent = {}
        for doc in results:
            pid = doc['metadata'].get('parent_doc_id', doc['id'])
            if pid not in by_parent or doc['retrieval_score'] > by_parent[pid]['retrieval_score']:
                by_parent[pid] = doc

        unique_docs = list(by_parent.values())

        # Layer 2: Content-based dedup for cross-URL duplicates
        final = []
        seen_content = []
        for doc in sorted(unique_docs, key=lambda x: x['retrieval_score'], reverse=True):
            snippet = doc['document'][:200].strip().lower()
            is_dup = any(self._text_overlap(snippet, s) > 0.9 for s in seen_content)
            if not is_dup:
                final.append(doc)
                seen_content.append(snippet)

        return final

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """Word-level Jaccard similarity between two text snippets."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)
