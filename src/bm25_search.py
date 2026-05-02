"""BM25 sparse keyword search for hybrid retrieval with Reciprocal Rank Fusion."""
import re
import time
from typing import List, Dict, Optional


try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class BM25Search:
    """BM25 keyword search index over the document corpus."""

    def __init__(self):
        self.index: Optional[object] = None
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_metadatas: List[Dict] = []
        self._is_built = False

    @property
    def is_available(self) -> bool:
        return BM25_AVAILABLE and self._is_built

    def build_index(self, collection) -> None:
        """Build BM25 index from all documents in a ChromaDB collection."""
        if not BM25_AVAILABLE:
            print("[BM25] rank_bm25 not installed. BM25 search disabled.")
            return

        start = time.time()
        all_docs = collection.get(include=["documents", "metadatas"])

        self.doc_ids = all_docs["ids"]
        self.doc_texts = all_docs["documents"]
        self.doc_metadatas = all_docs["metadatas"]

        if not self.doc_ids:
            self.index = None
            self._is_built = False
            print("[BM25] No documents in collection — BM25 disabled until ingestion.")
            return

        tokenized = [self._tokenize(text) for text in self.doc_texts]
        self.index = BM25Okapi(tokenized)
        self._is_built = True

        elapsed = time.time() - start
        print(f"[BM25] Index built: {len(self.doc_ids)} docs in {elapsed:.1f}s")

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Return the top-k BM25 results for *query*."""
        if not self.is_available:
            return []

        tokens = self._tokenize(query)
        scores = self.index.get_scores(tokens)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append({
                    "rank": rank + 1,
                    "id": self.doc_ids[idx],
                    "document": self.doc_texts[idx],
                    "metadata": self.doc_metadatas[idx],
                    "bm25_score": float(scores[idx]),
                    "similarity": 0.0,
                })
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Whitespace + punctuation tokenizer with lowercasing."""
        return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(
    ranking_lists: List[List[Dict]],
    k: int = 60,
) -> List[Dict]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Each ranking list is a list of dicts that must contain an ``id`` key.
    Returns a single merged list sorted by descending RRF score.  The
    constant *k* (default 60) controls how much weight lower-ranked
    documents receive.
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    for ranking in ranking_lists:
        for rank_pos, doc in enumerate(ranking, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank_pos)
            # Keep the copy with the richest metadata
            if doc_id not in doc_map:
                doc_map[doc_id] = dict(doc)

    for doc_id, score in rrf_scores.items():
        doc_map[doc_id]["retrieval_score"] = score

    fused = sorted(doc_map.values(), key=lambda d: d["retrieval_score"], reverse=True)
    for i, doc in enumerate(fused):
        doc["rank"] = i + 1
    return fused
