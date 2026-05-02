"""LLM-based query rewriter -- always rewrites every query into a precise English search query.

Handles non-English input (translates to English), abbreviation expansion, and
follow-up resolution using conversation memory.
"""
from src.llm_provider import LLMProvider
from src.program_catalog import find_programmes_in_query
from src.scholarship_catalog import find_scholarships_in_query
from src.utils import should_skip_retrieval

_REWRITE_SYSTEM = (
    "You are a search-query optimizer for a university admissions knowledge base. "
    "The knowledge base is in English. "
    "Rewrite the user's question into a concise, precise English search query.\n"
    "Rules:\n"
    "- Output ONLY the rewritten query, nothing else.\n"
    "- If the question is in a non-English language, translate it to English.\n"
    "- Expand abbreviations and slang into formal terms only when a provided mapping confirms them.\n"
    "- For SFU programme and scholarship codes, use the Known code mappings in the user prompt exactly. "
    "Never guess names from a code.\n"
    "- If conversation history is provided and the question is a follow-up, "
    "resolve all references (pronouns, 'that', 'the first one', etc.) "
    "using the conversation history so the rewritten query is fully self-contained.\n"
    "- Preserve specific names, course codes, and dates.\n"
    "- Keep the rewritten query under 30 words.\n"
    "- If the original query is already a precise English search query, "
    "return it unchanged."
)


def _programme_mapping_lines(query: str) -> list[str]:
    programmes = find_programmes_in_query(query)
    return [
        f"- {programme['code']} = {programme['name']}"
        for programme in programmes
    ]


def _scholarship_mapping_lines(query: str) -> list[str]:
    scholarships = find_scholarships_in_query(query)
    return [
        f"- {scholarship['identifier']} = {scholarship['name']}"
        for scholarship in scholarships
        if scholarship.get('identifier')
    ]


def _apply_known_code_guard(original_query: str, rewritten: str) -> str:
    """Keep LLM rewrite output from replacing known SFU codes incorrectly."""
    programmes = find_programmes_in_query(original_query)
    scholarships = [
        scholarship
        for scholarship in find_scholarships_in_query(original_query)
        if scholarship.get('identifier')
    ]
    if not programmes and not scholarships:
        return rewritten

    mapping_entries = [
        f"{programme['name']} ({programme['code']})"
        for programme in programmes
    ] + [
        f"{scholarship['name']} ({scholarship['identifier']})"
        for scholarship in scholarships
    ]
    mapping_prefix = "; ".join(mapping_entries)

    rewritten_lower = rewritten.lower()
    has_all_programme_mappings = all(
        programme["code"].lower() in rewritten_lower
        and programme["name"].lower() in rewritten_lower
        for programme in programmes
    )
    has_all_scholarship_mappings = all(
        scholarship["identifier"].lower() in rewritten_lower
        and scholarship["name"].lower() in rewritten_lower
        for scholarship in scholarships
    )
    if has_all_programme_mappings and has_all_scholarship_mappings:
        return rewritten

    # Prefer the user's original intent text over a potentially hallucinated
    # code expansion from the LLM, but anchor it with the verified mapping.
    return f"{mapping_prefix}. User question: {original_query}"


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
    mapping_lines = _programme_mapping_lines(query) + _scholarship_mapping_lines(query)
    if mapping_lines:
        prompt_parts.append(
            "Known code mappings in this user question:\n"
            + "\n".join(mapping_lines)
            + "\n"
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
            rewritten = _apply_known_code_guard(query, rewritten)
            print(f"✏️ Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        print(f"⚠️ Query rewrite failed ({e}), using original")

    return query
