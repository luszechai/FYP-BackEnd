"""
Run Ragas evaluation against the saved testset.

This standalone script:
1. Loads the testset from eval_testset.json
2. Initialises the RAG pipeline (same setup as api_server.py)
3. Runs each question through retrieve_context + generate_response
4. Computes Ragas metrics (faithfulness, answer_relevancy, context_precision, context_recall)
5. Saves full results to eval_results.json and prints a console summary

Usage:
    python run_ragas_evaluation.py
    python run_ragas_evaluation.py --testset eval_testset.json --output eval_results.json
    python run_ragas_evaluation.py --max-questions 10   # quick run on a subset
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from config import Config
from src.vector_db import ChromaDBManager
from src.llm_provider import LLMProvider
from src.chatbot import RAGChatbot
from src.ragas_evaluation import (
    load_testset,
    run_pipeline_on_testset,
    evaluate_with_ragas,
    format_results_summary,
    save_results,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Ragas evaluation on the RAG pipeline"
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="eval_testset.json",
        help="Path to the testset JSON file (default: eval_testset.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Output path for evaluation results (default: eval_results.json)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Evaluate only the first N questions (for quick testing)",
    )
    args = parser.parse_args()

    # ---- Validate config ----
    Config.validate()

    # ---- Load testset ----
    print("=" * 60)
    print("RAGAS EVALUATION PIPELINE")
    print("=" * 60)

    testset = load_testset(args.testset)

    # ---- Initialise RAG pipeline (same as api_server.py startup) ----
    print("\n🔧 Initialising RAG pipeline...")

    llm = LLMProvider(
        provider="deepseek",
        api_key=Config.DEEPSEEK_API_KEY,
        temperature=Config.LLM_TEMPERATURE,
        max_tokens=Config.LLM_MAX_TOKENS,
        enable_cache=False,  # Disable cache for evaluation
    )

    db = ChromaDBManager(
        persist_directory=Config.CHROMA_DB_DIR,
        collection_name=Config.CHROMA_COLLECTION_NAME,
    )

    if db.collection.count() == 0:
        if os.path.exists(Config.DATA_FILE):
            db.add_documents_from_json(Config.DATA_FILE)
        else:
            print(f"❌ No documents in ChromaDB and {Config.DATA_FILE} not found.")
            sys.exit(1)

    chatbot = RAGChatbot(
        chroma_db=db,
        llm_provider=llm,
        use_adaptive_config=Config.USE_ADAPTIVE_CONFIG,
    )

    print(f"   📚 {db.collection.count()} chunks in ChromaDB")

    # ---- Run pipeline on testset ----
    pipeline_results = run_pipeline_on_testset(
        chatbot, testset, max_questions=args.max_questions
    )

    if not pipeline_results:
        print("❌ No results collected. Exiting.")
        sys.exit(1)

    # ---- Evaluate with Ragas ----
    eval_results = evaluate_with_ragas(pipeline_results)

    # ---- Save results ----
    save_results(eval_results, args.output)

    # ---- Print summary ----
    summary = format_results_summary(eval_results)
    print(summary)

    print(f"\n📁 Full results saved to: {args.output}")
    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
