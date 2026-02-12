"""
Shared Ragas evaluation module.

Provides reusable functions for loading testsets, running the RAG pipeline
against them, computing Ragas metrics, and formatting results.

Used by both the standalone run_ragas_evaluation.py script and the
/api/ragas/* API endpoints.
"""
import json
import os
import time
from typing import Dict, List, Optional

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
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
) -> List[Dict]:
    """
    Run the RAG pipeline on each question in the testset and collect results.

    Args:
        chatbot: An initialised RAGChatbot instance.
        testset: List of testset items (each must have 'user_input').
        max_questions: Optional cap on how many questions to evaluate
                       (useful for quick test runs).

    Returns:
        List of result dicts, each containing:
        - user_input
        - retrieved_contexts  (list of chunk text strings)
        - response            (generated answer)
        - reference           (ground-truth from testset, may be None)
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

        print(f"   [{idx}/{total}] {question[:80]}...")

        start = time.time()
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

        results.append(
            {
                "user_input": question,
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": reference,
            }
        )

    print(f"✅ Pipeline completed for {len(results)} questions")
    return results


# ---------------------------------------------------------------------------
# 3. Evaluate with Ragas
# ---------------------------------------------------------------------------

def _get_ragas_llm() -> LangchainLLMWrapper:
    """Create a LangchainLLMWrapper around DeepSeek for Ragas metric computation."""
    llm = ChatOpenAI(
        model=Config.DEEPSEEK_MODEL,
        base_url=Config.DEEPSEEK_BASE_URL,
        api_key=Config.DEEPSEEK_API_KEY,
        temperature=0.0,
        max_tokens=4096,
    )
    return LangchainLLMWrapper(llm)


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
