"""Retrieval strategies for document search"""
import re
from datetime import date, datetime
from typing import List, Dict, Optional
from src.vector_db import ChromaDBManager
from src.bm25_search import BM25Search, reciprocal_rank_fusion
from src.utils import is_scholarship_query, detect_email_category


class HybridRetriever:
    """Handles hybrid retrieval strategies for document search.

    Plain vector similarity always runs as the baseline retrieval path. BM25
    keyword search is gated behind ``use_bm25``. The enhanced non-BM25
    strategies (expanded-query variants, keyword boosting) are gated behind
    ``use_hybrid`` so the evaluation dashboard can toggle them independently
    per run without mutating globals. ``use_person_boost`` further gates the
    person-query expansion strategy, which is only meaningful when
    ``use_hybrid`` is enabled.
    """

    def __init__(self, chroma_db: ChromaDBManager, retrieval_k: int = 5,
                 bm25: Optional[BM25Search] = None,
                 use_bm25: bool = True,
                 use_hybrid: bool = False,
                 use_person_boost: bool = False):
        self.db = chroma_db
        self.retrieval_k = retrieval_k
        self.bm25 = bm25
        self.use_bm25 = use_bm25
        self.use_hybrid = use_hybrid
        self.use_person_boost = use_person_boost

    def hybrid_retrieval(self, enhanced_query: Dict, use_memory: bool = True,
                         reranker_mode: bool = False) -> List[Dict]:
        """Perform hybrid retrieval with multiple query strategies"""
        all_results = {}
        
        # Detect if this is a scholarship/deadline query
        query_lower = enhanced_query['original'].lower()
        is_scholarship = is_scholarship_query(enhanced_query['original'])
        
        # Adjust retrieval size for scholarship queries
        base_n_results = self.retrieval_k * 3 if is_scholarship else self.retrieval_k * 2

        print(f"[retrieval] strategy 1: baseline vector query (k={base_n_results})")
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
        if self.use_hybrid and is_scholarship_enhanced:
            print("[retrieval] strategy 1.6: scholarship queries (listing + deadline)")
            
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

        if self.use_hybrid and self.use_person_boost and enhanced_query.get('is_person_query', False) and len(enhanced_query.get('expanded_queries', [])) > 1:
            print(f"[retrieval] strategy 2: expanded person queries ({len(enhanced_query['expanded_queries'])} variations)")

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
        if self.use_hybrid and enhanced_query.get('is_role_query', False) and len(enhanced_query.get('expanded_queries', [])) > 1:
            print(f"[retrieval] strategy 2.3: expanded role queries ({len(enhanced_query['expanded_queries'])} variations)")

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
        if self.use_hybrid and enhanced_query.get('is_program_query', False) and len(enhanced_query.get('expanded_queries', [])) > 1:
            print(f"[retrieval] strategy 2.5: expanded program queries ({len(enhanced_query['expanded_queries'])} variations)")

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
        if self.use_hybrid and enhanced_query.get('is_scholarship_query', False) and len(enhanced_query.get('expanded_queries', [])) > 1:
            print(f"[retrieval] strategy 2.6: expanded scholarship queries ({len(enhanced_query['expanded_queries'])} variations)")

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
        if self.use_hybrid and enhanced_query.get('keywords'):
            print("[retrieval] strategy 3: keyword matching")
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
        if self.use_hybrid and enhanced_query.get('is_role_query', False):
            print("[retrieval] strategy 3.5: role-specific keyword boosting")
            role_keywords = ['programme leader', 'program leader', 'director', 'head of', 
                           'role', 'coordinator', 'dean', 'chair']
            for doc_id, doc in all_results.items():
                content_lower = doc['document'].lower()
                # Check if document contains role-related keywords
                role_matches = sum(1 for rk in role_keywords if rk in content_lower)
                if role_matches > 0:
                    boost = min(0.15, role_matches * 0.05)
                    doc['retrieval_score'] = min(1.0, doc['retrieval_score'] + boost)

        # ── BM25 keyword search (+ optional RRF fusion with other retrieval strategies) ──
        if self.use_bm25 and self.bm25 is not None and self.bm25.is_available:
            bm25_k = base_n_results
            raw_query = enhanced_query['original']
            # Strip prepended memory context if present
            if '\nCurrent question: ' in raw_query:
                raw_query = raw_query.split('\nCurrent question: ')[-1]

            print(f"[retrieval] strategy bm25: keyword search (k={bm25_k})")
            bm25_results = self.bm25.search(raw_query, k=bm25_k)

            if not self.use_hybrid or not all_results:
                # Use only BM25 ranking; no fusion with vector or other strategies
                for doc in bm25_results:
                    doc['retrieval_score'] = doc.get('bm25_score', 1.0 / (doc.get('rank', 1) + 1))
                all_results = {doc['id']: doc for doc in bm25_results}
            else:
                # Build a vector ranking list (sorted by current retrieval_score)
                vector_ranking = sorted(
                    all_results.values(),
                    key=lambda x: x.get('retrieval_score', 0),
                    reverse=True,
                )
                # Fuse vector + BM25 rankings with RRF
                fused = reciprocal_rank_fusion([vector_ranking, bm25_results], k=60)
                all_results = {doc['id']: doc for doc in fused}
        elif self.use_hybrid:
            # Assign retrieval_score for documents that may only have similarity.
            for doc in all_results.values():
                doc.setdefault('retrieval_score', doc.get('similarity', 0))

        # Email aggregate injection should not depend on BM25 being enabled/available.
        # Otherwise, category queries like "next workshop" regress when running in
        # vector-only mode (or BM25 index build fails).
        try:
            self._inject_email_aggregates(all_results, enhanced_query.get('original', ''))
        except Exception:
            # Retrieval must never fail due to email injection.
            pass

        deduplicated = self._deduplicate_results(list(all_results.values()))
        sorted_results = sorted(deduplicated,
                              key=lambda x: x['retrieval_score'],
                              reverse=True)

        if reranker_mode:
            max_results = self.retrieval_k * 4 if is_scholarship else self.retrieval_k * 3
        else:
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

    def _inject_email_aggregates(self, all_results: Dict[str, Dict], raw_query: str) -> None:
        """Inject one aggregate doc per matching email category, not one per chunk."""
        email_cat = detect_email_category(raw_query)
        if not email_cat:
            return

        # Chroma "where" filters are an implicit AND across fields. Avoid $and/$eq
        # because operator support differs across Chroma versions.
        # Some Chroma builds only accept a single top-level predicate in `where`.
        # Filter by `email_type` in Chroma, then validate `type` in Python.
        where_filter = {"email_type": email_cat}
        try:
            email_results = self.db.collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )
        except Exception:
            email_results = {"ids": [], "documents": [], "metadatas": []}

        # Ensure we're only aggregating true email chunks.
        email_ids = []
        for idx, _id in enumerate(email_results.get("ids", []) or []):
            meta = (email_results.get("metadatas", []) or [])
            m = meta[idx] if idx < len(meta) and meta[idx] else {}
            if (m.get("type") or "") == "email":
                email_ids.append(_id)
        if not email_ids:
            return

        email_groups = self._group_email_chunks(
            email_ids,
            email_results.get("documents", []),
            email_results.get("metadatas", []),
        )
        today = date.today()
        top_score = max(
            (d.get("retrieval_score", 0) for d in all_results.values()),
            default=1.0,
        )
        inject_score = max(top_score, 1.0)
        injected = 0
        skipped_expired = 0

        for email_key, group in email_groups.items():
            doc_text = "\n\n".join(
                chunk["document"] for chunk in group["chunks"] if chunk["document"]
            )
            metadata = dict(group["metadata"])
            metadata["parent_doc_id"] = email_key
            self._remove_email_chunks(all_results, email_key)

            if self._is_expired_email(metadata, doc_text, today):
                skipped_expired += 1
                continue

            result_id = f"email:{email_key}"
            if result_id not in all_results:
                all_results[result_id] = {
                    "id": result_id,
                    "document": doc_text,
                    "metadata": metadata,
                    "retrieval_score": inject_score,
                    "rank": 0,
                    "similarity": 0.0,
                }
                injected += 1
            else:
                all_results[result_id]["retrieval_score"] = max(
                    all_results[result_id]["retrieval_score"], inject_score
                )

        if injected:
            msg = f"[email] injected {injected} email(s) for category '{email_cat}'"
            if skipped_expired:
                msg += f" (skipped {skipped_expired} expired email(s))"
            print(msg)
        elif skipped_expired:
            print(
                f"[email] skipped {skipped_expired} expired email(s) "
                f"for category '{email_cat}'"
            )

    @classmethod
    def _group_email_chunks(
        cls,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict],
    ) -> Dict[str, Dict]:
        """Group Chroma email chunks back into one logical email."""
        groups: Dict[str, Dict] = {}
        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            email_key = cls._email_group_key(metadata, chunk_id)
            group = groups.setdefault(email_key, {"metadata": metadata, "chunks": []})
            if len(str(metadata)) > len(str(group["metadata"])):
                group["metadata"] = metadata
            group["chunks"].append({
                "id": chunk_id,
                "document": documents[idx] if idx < len(documents) else "",
                "index": metadata.get("chunk_index", idx),
            })

        for group in groups.values():
            group["chunks"].sort(key=lambda chunk: chunk["index"])
        return groups

    @staticmethod
    def _email_group_key(metadata: Dict, fallback_id: str) -> str:
        return (
            str(metadata.get("email_id") or "").strip()
            or str(metadata.get("source") or "").strip()
            or str(metadata.get("parent_doc_id") or "").strip()
            or fallback_id
        )

    @classmethod
    def _remove_email_chunks(cls, all_results: Dict[str, Dict], email_key: str) -> None:
        """Drop existing chunk-level hits for the same email before injecting aggregate."""
        for doc_id, doc in list(all_results.items()):
            metadata = doc.get("metadata", {}) or {}
            if cls._email_group_key(metadata, doc_id) == email_key:
                del all_results[doc_id]

    @classmethod
    def _is_expired_email(cls, metadata: Dict, document: str, today: date) -> bool:
        """Return True when an injected email is no longer actionable."""
        application_text = " ".join(
            str(metadata.get(key, "") or "")
            for key in ("email_application_period", "email_period")
        )
        application_dates = cls._extract_dates(application_text)
        if application_dates:
            return max(application_dates) < today

        event_text = " ".join(
            str(metadata.get(key, "") or "")
            for key in ("email_event_time", "email_time", "email_event_period")
        )
        event_dates = cls._extract_dates(event_text)
        if event_dates:
            return max(event_dates) < today

        # Metadata did not carry dates in older rows; inspect only structured headers.
        header_text = "\n".join(
            line for line in (document or "").splitlines()
            if line.startswith(("Application Period:", "Event Period:", "Event Time:"))
        )
        header_dates = cls._extract_dates(header_text)
        return bool(header_dates and max(header_dates) < today)

    @staticmethod
    def _extract_dates(text: str) -> List[date]:
        """Extract common email date formats; unknown/no-year dates use current year."""
        if not text:
            return []

        month_names = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }
        current_year = date.today().year
        dates: List[date] = []

        def add(y: int, m: int, d: int) -> None:
            try:
                dates.append(date(y, m, d))
            except ValueError:
                pass

        for match in re.finditer(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", text):
            add(int(match.group(1)), int(match.group(2)), int(match.group(3)))

        # HK emails commonly use day/month/year and compact ranges.
        for match in re.finditer(
            r"\b(\d{1,2})[/-](\d{1,2})\s*(?:-|–|to)\s*"
            r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
            text,
            re.IGNORECASE,
        ):
            year = int(match.group(5))
            add(year, int(match.group(2)), int(match.group(1)))
            add(year, int(match.group(4)), int(match.group(3)))

        for match in re.finditer(
            r"\b(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
            text,
            re.IGNORECASE,
        ):
            year = int(match.group(4))
            month = int(match.group(3))
            add(year, month, int(match.group(1)))
            add(year, month, int(match.group(2)))

        for match in re.finditer(r"\b(?!\d{4}[/-])(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", text):
            add(int(match.group(3)), int(match.group(2)), int(match.group(1)))

        for match in re.finditer(
            r"\b(\d{1,2})\s+"
            r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\.?,?\s*(\d{4})?\b",
            text,
            re.IGNORECASE,
        ):
            add(
                int(match.group(3) or current_year),
                month_names[match.group(2).lower().rstrip(".")],
                int(match.group(1)),
            )

        for match in re.finditer(
            r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\.?\s+(\d{1,2})(?:,\s*(\d{4}))?\b",
            text,
            re.IGNORECASE,
        ):
            add(
                int(match.group(3) or current_year),
                month_names[match.group(1).lower().rstrip(".")],
                int(match.group(2)),
            )

        for match in re.finditer(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日號]?", text):
            add(int(match.group(1)), int(match.group(2)), int(match.group(3)))

        for match in re.finditer(r"(?<!年)(\d{1,2})\s*月\s*(\d{1,2})\s*[日號]?", text):
            add(current_year, int(match.group(1)), int(match.group(2)))

        return dates

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """Word-level Jaccard similarity between two text snippets."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)
