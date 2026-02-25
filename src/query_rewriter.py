"""LLM-based query rewriter -- always rewrites every query into a precise English search query.

Handles non-English input (translates to English), abbreviation expansion, and
follow-up resolution using conversation memory.
"""
from src.llm_provider import LLMProvider
from src.utils import should_skip_retrieval

_REWRITE_SYSTEM = (
    "You are a search-query optimizer for a university admissions knowledge base. "
    "The knowledge base is in English. "
    "Rewrite the user's question into a concise, precise English search query.\n"
    "Rules:\n"
    "- Output ONLY the rewritten query, nothing else.\n"
    "- If the question is in a non-English language, translate it to English.\n"
    "- Expand abbreviations and slang into formal terms "
    "(e.g. bsai -> Bachelor of Science in Artificial Intelligence).\n"
    "- If conversation history is provided and the question is a follow-up, "
    "resolve all references (pronouns, 'that', 'the first one', etc.) "
    "using the conversation history so the rewritten query is fully self-contained.\n"
    "- Preserve specific names, course codes, and dates.\n"
    "- Keep the rewritten query under 30 words.\n"
    "- If the original query is already a precise English search query, "
    "return it unchanged."
)


def rewrite_query(llm: LLMProvider, query: str,
                  conversation_context: str = "") -> str:
    """Rewrite *query* into a precise English search query using the LLM.

    Always runs (no heuristic gate) so that non-English queries, follow-ups,
    and abbreviations are handled uniformly.  Skips only trivial inputs that
    would bypass retrieval anyway (greetings, "ok", etc.).

    Args:
        llm: The LLM provider used for rewriting.
        query: The raw user query.
        conversation_context: Formatted recent conversation history so the
            rewriter can resolve follow-up references.

    Returns:
        The rewritten query string, or the original if rewriting fails.
    """
    if should_skip_retrieval(query):
        return query

    prompt_parts = []
    if conversation_context:
        prompt_parts.append(
            f"Conversation history:\n{conversation_context}\n"
        )
    prompt_parts.append(f"User question: {query}")
    prompt = "\n".join(prompt_parts)

    try:
        original_max = llm.max_tokens
        original_temp = llm.temperature
        llm.max_tokens = 80
        llm.temperature = 0.0
        try:
            rewritten = llm.generate_response(
                prompt=prompt,
                system_message=_REWRITE_SYSTEM,
                use_cache=True,
            )
        finally:
            llm.max_tokens = original_max
            llm.temperature = original_temp

        rewritten = rewritten.strip().strip('"').strip("'")
        if rewritten and 3 < len(rewritten) < 300:
            print(f"✏️ Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        print(f"⚠️ Query rewrite failed ({e}), using original")

    return query
