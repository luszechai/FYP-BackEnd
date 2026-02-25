"""Reranker module using BAAI/bge-reranker-base cross-encoder model.

Uses HuggingFace transformers directly for maximum compatibility.
"""
import time
from typing import List, Dict


class Reranker:
    """Cross-encoder reranker that rescores (query, passage) pairs for more precise relevance."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", use_fp16: bool = True):
        """
        Initialize the reranker with a cross-encoder model.

        Args:
            model_name: HuggingFace model name for the cross-encoder reranker.
            use_fp16: Use half-precision for faster inference (slight accuracy trade-off).
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the reranker model using HuggingFace transformers directly."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            print(f"[Reranker] Loading model: {self.model_name} (fp16={self.use_fp16})")
            start = time.time()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()

            if self.use_fp16:
                try:
                    self.model = self.model.half()
                except Exception:
                    # fp16 may not be supported on all hardware (e.g., CPU-only without BFloat16)
                    pass

            elapsed = time.time() - start
            print(f"[Reranker] Model loaded in {elapsed:.1f}s")

        except ImportError as e:
            print(f"[Reranker] transformers/torch not installed: {e}. Reranker disabled.")
            self.model = None
            self.tokenizer = None
        except Exception as e:
            print(f"[Reranker] Failed to load model: {e}. Reranker disabled.")
            self.model = None
            self.tokenizer = None

    @property
    def is_available(self) -> bool:
        """Check if the reranker model is loaded and ready."""
        return self.model is not None and self.tokenizer is not None

    def _compute_scores(self, pairs: List[List[str]]) -> List[float]:
        """Compute relevance scores for (query, passage) pairs using the cross-encoder."""
        import torch

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            # Move inputs to same device/dtype as model
            if self.use_fp16 and self.model.dtype == torch.float16:
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            return scores.tolist()

    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank documents using the cross-encoder and return top_k sorted by relevance.

        Args:
            query: The user's original query string.
            documents: List of document dicts (must contain 'document' key with text content).
            top_k: Number of top documents to return after reranking.

        Returns:
            List of document dicts sorted by reranker score, truncated to top_k.
            If the reranker is unavailable, returns the original documents unchanged.
        """
        if not self.is_available:
            print("[Reranker] Not available, returning original ranking")
            return documents[:top_k]

        if not documents:
            return documents

        try:
            start = time.time()

            # Build (query, passage) pairs for the cross-encoder
            pairs = [[query, doc['document']] for doc in documents]

            # Compute relevance scores
            scores = self._compute_scores(pairs)

            # Normalize scores to [0, 1] using min-max normalization
            min_s = min(scores)
            max_s = max(scores)
            score_range = max_s - min_s

            for doc, raw_score in zip(documents, scores):
                doc['reranker_raw_score'] = raw_score
                if score_range > 1e-8:
                    doc['retrieval_score'] = (raw_score - min_s) / score_range
                else:
                    # All scores are identical - assign uniform score
                    doc['retrieval_score'] = 1.0

            # Sort by reranker score descending
            documents.sort(key=lambda x: x['retrieval_score'], reverse=True)

            elapsed = time.time() - start
            print(f"[Reranker] Reranked {len(documents)} docs in {elapsed:.3f}s -> returning top {top_k}")

            # Update ranks after reranking
            for i, doc in enumerate(documents):
                doc['rank'] = i + 1

            return documents[:top_k]

        except Exception as e:
            print(f"[Reranker] Failed during reranking: {e}. Returning original ranking.")
            return documents[:top_k]
