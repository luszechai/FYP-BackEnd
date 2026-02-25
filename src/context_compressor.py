"""Extractive context compression -- keeps only query-relevant sentences from each chunk."""
from typing import List, Dict, Optional

from src.llm_provider import LLMProvider

_COMPRESS_SYSTEM = (
    "You are an information extractor. Given a user question and a document chunk, "
    "extract ONLY the sentences that are directly relevant to answering the question. "
    "Rules:\n"
    "- Output the relevant sentences verbatim (do not paraphrase).\n"
    "- If nothing is relevant, output exactly: [IRRELEVANT]\n"
    "- Do NOT add any commentary, headers, or explanations.\n"
    "- Preserve all facts, numbers, names, and dates exactly as written.\n"
    "- The question may have been translated or rewritten from another language. "
    "Judge relevance based on the semantic meaning, not exact wording."
)

# Chunks shorter than this are passed through without compression
_MIN_CHARS_FOR_COMPRESSION = 400

# Maximum characters we send to the LLM for compression (to control cost)
_MAX_CHUNK_INPUT = 2000


def compress_context(
    llm: LLMProvider,
    query: str,
    documents: List[Dict],
    max_documents: int = 8,
) -> List[Dict]:
    """Compress each document chunk by extracting only the query-relevant sentences.

    Returns a new list of document dicts with the ``document`` field replaced
    by the compressed text.  Documents marked ``[IRRELEVANT]`` by the LLM are
    dropped entirely.
    """
    compressed: List[Dict] = []

    for doc in documents[:max_documents]:
        text = doc.get("document", "")

        # Short chunks are unlikely to contain much noise -- keep as-is
        if len(text) <= _MIN_CHARS_FOR_COMPRESSION:
            compressed.append(doc)
            continue

        truncated = text[:_MAX_CHUNK_INPUT]
        prompt = (
            f"Question: {query}\n\n"
            f"Document chunk:\n{truncated}\n\n"
            "Extract only the sentences relevant to the question."
        )

        try:
            original_max = llm.max_tokens
            original_temp = llm.temperature
            llm.max_tokens = 500
            llm.temperature = 0.0
            try:
                result = llm.generate_response(
                    prompt=prompt,
                    system_message=_COMPRESS_SYSTEM,
                    use_cache=True,
                )
            finally:
                llm.max_tokens = original_max
                llm.temperature = original_temp

            result = result.strip()
            if result and "[IRRELEVANT]" not in result:
                new_doc = dict(doc)
                new_doc["document"] = result
                new_doc["compressed"] = True
                compressed.append(new_doc)
                saved = len(text) - len(result)
                if saved > 0:
                    print(f"  🗜️ Compressed chunk (saved {saved} chars)")
            else:
                print(f"  🗑️ Dropped irrelevant chunk")
        except Exception as e:
            print(f"  ⚠️ Compression failed ({e}), keeping original")
            compressed.append(doc)

    # Fallback: if compression dropped ALL documents, keep the top originals
    if not compressed and documents:
        print("  ⚠️ All chunks dropped by compressor — keeping top originals as fallback")
        return list(documents[:3])

    return compressed
