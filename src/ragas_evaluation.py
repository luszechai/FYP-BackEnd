"""
Shared Ragas evaluation module.

Provides reusable functions for loading testsets, running the RAG pipeline
against them, computing Ragas metrics, and formatting results.

Used by both the standalone run_ragas_evaluation.py script and the
/api/ragas/* API endpoints.
"""
import asyncio
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config


# ---------------------------------------------------------------------------
# 1. Load testset
# ---------------------------------------------------------------------------

def load_testset(path: str = "eval_testset.json") -> List[Dict]:
    """
    Load and validate the evaluation testset JSON file.

    Args:
        path: Path to the testset JSON file.

    Returns:
        List of testset question dicts, each with at least
        'user_input' (or 'question') and 'reference' (ground-truth answer).

    Raises:
        FileNotFoundError: If the testset file does not exist.
        ValueError: If the file is empty or malformed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Testset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both wrapped format {"testset": [...]} and raw list [...]
    if isinstance(data, dict):
        testset = data.get("testset", [])
    elif isinstance(data, list):
        testset = data
    else:
        raise ValueError("Unexpected testset format. Expected a list or {'testset': [...]}.")

    if not testset:
        raise ValueError("Testset is empty.")

    # Normalise field names – Ragas generates "user_input" but users might
    # create files with "question".  We unify to both being present.
    for item in testset:
        if "user_input" not in item and "question" in item:
            item["user_input"] = item["question"]
        if "question" not in item and "user_input" in item:
            item["question"] = item["user_input"]
        if "reference" not in item and "ground_truth" in item:
            item["reference"] = item["ground_truth"]
        if "ground_truth" not in item and "reference" in item:
            item["ground_truth"] = item["reference"]

    print(f"✅ Loaded {len(testset)} questions from {path}")
    return testset


# ---------------------------------------------------------------------------
# 2. Run the RAG pipeline on the testset
# ---------------------------------------------------------------------------

def run_pipeline_on_testset(
    chatbot,
    testset: List[Dict],
    max_questions: Optional[int] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> List[Dict]:
    """
    Run the RAG pipeline on each question in the testset and collect results.

    Args:
        chatbot: An initialised RAGChatbot instance.
        testset: List of testset items (each must have 'user_input').
        max_questions: Optional cap on how many questions to evaluate
                       (useful for quick test runs).
        progress_callback: Optional callable invoked with status events:
            {"type": "question_started", "index", "total", "question"}
            {"type": "question_done", "index", "total", "latency_s", "num_contexts"}
            {"type": "pipeline_done", "total"}
          Used by the SSE endpoint to stream per-question progress to the UI.

    Returns:
        List of result dicts, each containing:
        - user_input
        - retrieved_contexts  (list of chunk text strings)
        - response            (generated answer)
        - reference           (ground-truth from testset, may be None)
        - retrieved_docs_raw  (full doc dicts with metadata + scores, for Deep Dive)
    """
    items = testset[:max_questions] if max_questions else testset
    results: List[Dict] = []

    total = len(items)
    print(f"🚀 Running RAG pipeline on {total} questions...")

    for idx, item in enumerate(items, 1):
        question = item.get("user_input", item.get("question", ""))
        reference = item.get("reference", item.get("ground_truth"))

        if not question:
            print(f"   ⚠️  Skipping item {idx}: no question text")
            continue

        if progress_callback:
            try:
                progress_callback({
                    "type": "question_started",
                    "index": idx,
                    "total": total,
                    "question": question,
                })
            except Exception:
                pass

        print(f"   [{idx}/{total}] {question[:80]}...")

        start = time.time()
        retrieved_docs = []
        try:
            # Retrieve context
            retrieved_docs, context_string, _ = chatbot.retrieve_context(
                question, use_memory=False
            )

            # Extract individual chunk texts
            retrieved_contexts = [doc.get("document", "") for doc in retrieved_docs]

            # Generate response
            if context_string:
                response = chatbot.generate_response(
                    question, context_string, use_memory=False
                )
            else:
                response = "No relevant information found."

        except Exception as e:
            print(f"   ❌ Error on question {idx}: {e}")
            retrieved_contexts = []
            response = f"Error: {e}"

        elapsed = time.time() - start
        print(f"       ⏱️ {elapsed:.2f}s | contexts={len(retrieved_contexts)}")

        # Persist a lightweight copy of the source docs so the Deep Dive modal
        # can show chunk scores/sections/parent_doc_ids without another lookup.
        retrieved_docs_raw = []
        for doc in retrieved_docs:
            meta = doc.get("metadata", {}) or {}
            retrieved_docs_raw.append({
                "id": doc.get("id"),
                "document": doc.get("document", ""),
                "retrieval_score": round(float(doc.get("retrieval_score", 0.0)), 4),
                "similarity": round(float(doc.get("similarity", 0.0) or 0.0), 4),
                "section": meta.get("section"),
                "source": meta.get("source"),
                "parent_doc_id": meta.get("parent_doc_id"),
            })

        results.append(
            {
                "user_input": question,
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": reference,
                "retrieved_docs_raw": retrieved_docs_raw,
                "latency_s": round(elapsed, 3),
            }
        )

        if progress_callback:
            try:
                progress_callback({
                    "type": "question_done",
                    "index": idx,
                    "total": total,
                    "latency_s": round(elapsed, 3),
                    "num_contexts": len(retrieved_contexts),
                })
            except Exception:
                pass

    if progress_callback:
        try:
            progress_callback({"type": "pipeline_done", "total": len(results)})
        except Exception:
            pass

    print(f"✅ Pipeline completed for {len(results)} questions")
    return results


# ---------------------------------------------------------------------------
# 3. Evaluate with Ragas
# ---------------------------------------------------------------------------

class _ConcurrencyLimitedChatOpenAI(ChatOpenAI):
    """Caps simultaneous async Kimi calls — Ragas defaults to high parallelism and triggers org 429s."""

    def __init__(self, *args, max_concurrent: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self._sem = asyncio.Semaphore(max_concurrent)

    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        async with self._sem:
            return await super().agenerate_prompt(
                prompts, stop=stop, callbacks=callbacks, **kwargs
            )


def _get_ragas_llm() -> LangchainLLMWrapper:
    """Create a LangchainLLMWrapper around Kimi for Ragas metric computation."""
    llm = _ConcurrencyLimitedChatOpenAI(
        model=Config.KIMI_MODEL,
        base_url=Config.KIMI_BASE_URL,
        api_key=Config.KIMI_API_KEY,
        temperature=0.6,
        max_tokens=4096,
        max_concurrent=Config.RAGAS_KIMI_MAX_CONCURRENT,
        model_kwargs={
            "extra_body": {"thinking": {"type": "disabled"}},
        } if Config.KIMI_DISABLE_THINKING else {},
    )
    return LangchainLLMWrapper(llm, bypass_temperature=True)


def _get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Create a LangchainEmbeddingsWrapper around BGE-large for Ragas."""
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    return LangchainEmbeddingsWrapper(embeddings)


def evaluate_with_ragas(pipeline_results: List[Dict]) -> Dict:
    """
    Run Ragas metrics on the collected pipeline results.

    Args:
        pipeline_results: Output of run_pipeline_on_testset().

    Returns:
        Dict with:
        - 'aggregate': {metric_name: float}  (dataset-level averages)
        - 'per_question': list of per-row score dicts
    """
    print("📊 Running Ragas evaluation metrics...")

    # Build SingleTurnSamples
    samples: List[SingleTurnSample] = []
    for r in pipeline_results:
        sample = SingleTurnSample(
            user_input=r["user_input"],
            retrieved_contexts=r["retrieved_contexts"] or [""],
            response=r["response"],
            reference=r.get("reference") or "",
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)

    # Determine which metrics to use
    # context_recall and context_precision require a reference (ground truth)
    has_references = any(r.get("reference") for r in pipeline_results)

    ragas_llm = _get_ragas_llm()
    ragas_emb = _get_ragas_embeddings()

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb, strictness=1),
    ]
    if has_references:
        metrics.extend([
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ])
    else:
        print("   ⚠️  No ground-truth references found; skipping context_precision & context_recall")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(
            timeout=180,
            max_retries=6,
            max_wait=90,
            max_workers=Config.RAGAS_EVAL_MAX_WORKERS,
        ),
    )

    # Extract aggregate scores from EvaluationResult
    aggregate = {}
    for key, value in result._repr_dict.items():
        if isinstance(value, (int, float)):
            aggregate[key] = round(float(value), 4)

    # Extract per-question scores
    per_question = result.to_pandas().to_dict(orient="records")
    # Make numpy types JSON-serializable
    serializable_per_question = []
    for row in per_question:
        clean = {}
        for k, v in row.items():
            if hasattr(v, "item"):
                v = v.item()
            elif hasattr(v, "tolist"):
                v = v.tolist()
            clean[k] = v
        serializable_per_question.append(clean)

    print("✅ Ragas evaluation complete")
    return {
        "aggregate": aggregate,
        "per_question": serializable_per_question,
    }


# ---------------------------------------------------------------------------
# 4. Format results summary
# ---------------------------------------------------------------------------

def format_results_summary(eval_results: Dict) -> str:
    """
    Format a human-readable summary of Ragas evaluation results.

    Args:
        eval_results: Output of evaluate_with_ragas().

    Returns:
        Formatted multi-line string summary.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("RAGAS EVALUATION SUMMARY")
    lines.append("=" * 60)

    aggregate = eval_results.get("aggregate", {})
    per_question = eval_results.get("per_question", [])

    lines.append(f"\nTotal Questions Evaluated: {len(per_question)}")
    lines.append("\nAggregate Scores:")
    for metric, score in aggregate.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        lines.append(f"  {metric:<25s} {score:.4f}  [{bar}]")

    # Per-category breakdown (if 'category' is present)
    categories: Dict[str, List[Dict]] = {}
    for row in per_question:
        cat = row.get("category", "uncategorized")
        categories.setdefault(cat, []).append(row)

    if len(categories) > 1:
        lines.append("\nPer-Category Breakdown:")
        for cat, rows in sorted(categories.items()):
            lines.append(f"\n  [{cat}] ({len(rows)} questions)")
            # Average each metric for this category
            metric_keys = [k for k in rows[0].keys() if isinstance(rows[0].get(k), (int, float)) and k not in ("user_input", "response", "reference")]
            for mk in metric_keys:
                vals = [r[mk] for r in rows if isinstance(r.get(mk), (int, float))]
                if vals:
                    avg = sum(vals) / len(vals)
                    lines.append(f"    {mk:<23s} {avg:.4f}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Save / Load results
# ---------------------------------------------------------------------------

def save_results(eval_results: Dict, output_path: str = "eval_results.json"):
    """Save evaluation results to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"💾 Results saved to {output_path}")


def load_results(path: str = "eval_results.json") -> Optional[Dict]:
    """Load previously saved evaluation results. Returns None if not found."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 6. Per-run storage for the evaluation dashboard
# ---------------------------------------------------------------------------

EVAL_RUNS_DIR = "eval_runs"
_LABEL_SAFE_CHARS = re.compile(r"[^a-zA-Z0-9_\-]+")


def compute_testset_hash(testset: List[Dict]) -> str:
    """Short deterministic hash of the testset questions + references.

    Used so the dashboard can show whether two runs were scored against
    the same underlying testset (invalidates side-by-side comparisons when
    the testset was regenerated after a dataset swap).
    """
    h = hashlib.sha1()
    for item in testset:
        q = item.get("user_input") or item.get("question") or ""
        r = item.get("reference") or item.get("ground_truth") or ""
        h.update(q.encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(r.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def _slugify_label(label: str) -> str:
    slug = _LABEL_SAFE_CHARS.sub("_", (label or "").strip())
    slug = slug.strip("_")
    return slug[:60] or "run"


def build_run_id(label: Optional[str] = None, now: Optional[datetime] = None) -> str:
    """Produce a filesystem-safe run id like ``20260424T140512_Optimized``."""
    ts = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")
    slug = _slugify_label(label) if label else "run"
    return f"{ts}_{slug}"


def _sanitize_for_json(obj):
    """Recursively convert NaN/Infinity floats to None.

    Ragas occasionally emits NaN for per-question scores when a metric can't
    be computed (e.g. no retrieved contexts). Python's default ``json.dump``
    writes those as literal ``NaN`` which is not valid JSON and will choke
    the browser's ``JSON.parse``.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def save_run(run: Dict, runs_dir: str = EVAL_RUNS_DIR) -> str:
    """Persist a full evaluation run to ``runs_dir/<run_id>.json``.

    ``run`` must contain at least an ``id`` field. Returns the path written.
    """
    os.makedirs(runs_dir, exist_ok=True)
    run_id = run.get("id") or build_run_id(run.get("label"))
    run["id"] = run_id
    path = os.path.join(runs_dir, f"{run_id}.json")
    clean = _sanitize_for_json(run)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False, default=str, allow_nan=False)
    return path


def list_runs(runs_dir: str = EVAL_RUNS_DIR) -> List[Dict]:
    """Return run summaries (no per-question detail) sorted newest-first."""
    if not os.path.isdir(runs_dir):
        return []

    summaries: List[Dict] = []
    for fname in os.listdir(runs_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(runs_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        summaries.append({
            "id": data.get("id") or os.path.splitext(fname)[0],
            "label": data.get("label"),
            "timestamp": data.get("timestamp"),
            "llm_provider": data.get("llm_provider"),
            "llm_model": data.get("llm_model"),
            "strategies": data.get("strategies") or {},
            "aggregate": data.get("aggregate") or {},
            "question_count": len(data.get("per_question") or []),
            "runtime_s": data.get("runtime_s"),
            "dataset_file": data.get("dataset_file"),
            "chunk_count": data.get("chunk_count"),
            "testset_hash": data.get("testset_hash"),
        })

    summaries.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return summaries


def load_run(run_id: str, runs_dir: str = EVAL_RUNS_DIR) -> Optional[Dict]:
    """Load a single saved run by id. Returns None if not found."""
    if not run_id:
        return None
    safe_id = os.path.basename(run_id)
    path = os.path.join(runs_dir, f"{safe_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
