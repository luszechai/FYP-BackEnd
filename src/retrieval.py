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

        print(f"🔍 Strategy 1: Original query")
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
        
        # Strategy 1.5: Scholarship-specific queries
        if is_scholarship:
            print(f"🔍 Strategy 1.5: Scholarship/deadline specific queries")
            scholarship_terms = ['scholarship deadline', 'application deadline', 'due date', 
                                 'scholarship application', 'deadline for']
            for term in scholarship_terms:
                if term in query_lower:
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

        sorted_results = sorted(all_results.values(),
                              key=lambda x: x['retrieval_score'],
                              reverse=True)

        # Return more results for scholarship queries
        max_results = self.retrieval_k * 3 if is_scholarship else self.retrieval_k * 2
        return sorted_results[:max_results]

