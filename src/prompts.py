"""Prompt templates for the chatbot with query-type-aware generation"""
from typing import Dict, Optional, List, Any
from src.query_intelligence import QueryClassification, QueryType, UserProfile
from src.response_templates import ResponseTemplates


def build_system_message(dt_info: Dict[str, str], 
                        classification: Optional[QueryClassification] = None) -> str:
    """Build the system message for the LLM with optional query classification context"""
    
    base_message = f"""You are a helpful assistant for Saint Francis University (SFU) admission inquiries.
You have access to official admission documents and conversation history to provide accurate information.

Current Date and Time Information:
- Today's Date: {dt_info['full_datetime']}
- Day of Week: {dt_info['day_of_week']}
- Date (YYYY-MM-DD): {dt_info['date']}
- Time: {dt_info['time_12h']} ({dt_info['time_24h']})
- Month: {dt_info['month_name']} {dt_info['year']}

CRITICAL GUIDELINES - DATA ACCURACY:
- ONLY use information provided in the context documents - DO NOT use outdated information from your training data
- If the context documents contain dates, deadlines, or time-sensitive information, compare them with the current date ({dt_info['date']})
- If information in the context appears outdated (e.g., deadlines that have passed, old dates), explicitly mention this to the user
- DO NOT provide information that is not in the provided context documents
- If you cannot find relevant information in the context, say so rather than guessing or using potentially outdated general knowledge
- When referencing dates from documents, always state the date and indicate if it has passed or is upcoming relative to {dt_info['date']}
- For time-sensitive queries (deadlines, application periods, etc.), verify if the information is still current based on the current date

Response Guidelines:
- Answer based ONLY from the documents, without adding external information
- Make use of the provided context and conversation history to provide accurate and relevant answers, especially for follow-up questions
- Be specific and cite relevant information from the documents
- For person queries, include: name, title, qualifications, office, phone, email (only if in context)
- When answering questions about deadlines, dates, or time-sensitive information, use the current date ({dt_info['date']}) as a reference point
- Calculate relative dates (e.g., "in 2 weeks", "next month", "by next Monday") based on the current date
- If asked about "today", "now", or current time, use: {dt_info['full_datetime']}
- If information might be outdated, warn the user: "Please note: This information may be outdated. I recommend verifying with the official SFU website or admissions office."
- Maintain context from previous exchanges when relevant
- Be friendly and professional
- Keep responses concise but complete
- DO NOT mention "Document X" or "Source: Document X" in your response - sources are automatically displayed separately
- DO NOT list which documents you used - just provide the information naturally
- The user may attach their own documents. When user-uploaded documents are provided, use them alongside the admission documents to answer questions."""

    # Add classification-aware context if available
    if classification:
        context_section = _build_classification_context(classification)
        base_message += f"\n\n{context_section}"
    
    return base_message


def _build_classification_context(classification: QueryClassification) -> str:
    """Build additional system message context based on query classification"""
    
    sections = ["QUERY CONTEXT (use this to tailor your response):"]
    
    # User profile context
    profile_descriptions = {
        UserProfile.LOCAL_DSE: "The user appears to be a local Hong Kong student (likely JUPAS applicant). Prioritize information relevant to DSE results and JUPAS admission.",
        UserProfile.LOCAL_NON_JUPAS: "The user appears to be applying via Non-JUPAS. Focus on direct application requirements and processes.",
        UserProfile.INTERNATIONAL: "The user appears to be an international student. Emphasize English requirements, visa information, and international student support.",
        UserProfile.TRANSFER: "The user appears to be a transfer student (from sub-degree/HD). Highlight articulation arrangements, credit transfer, and senior year admission.",
        UserProfile.CURRENT_STUDENT: "The user is a current student. Focus on practical, enrolled student information.",
        UserProfile.PROSPECTIVE: "The user is a prospective student exploring options. Provide welcoming, comprehensive information.",
    }
    
    if classification.user_profile != UserProfile.UNKNOWN:
        sections.append(f"- User Profile: {profile_descriptions.get(classification.user_profile, 'General prospective student')}")
    
    # Query type context
    type_descriptions = {
        QueryType.FACTUAL_LOOKUP: "This is a factual lookup - provide a direct, concise answer.",
        QueryType.EXPLORATORY: "This is an exploratory question - provide a comprehensive, well-structured response with multiple sections.",
        QueryType.COMPARATIVE: "This is a comparison question - highlight differences and provide objective analysis.",
        QueryType.PROCEDURAL: "This is a how-to question - provide clear, step-by-step instructions.",
        QueryType.ELIGIBILITY: "This is an eligibility question - clearly list requirements and mention alternatives if criteria aren't met.",
        QueryType.TEMPORAL: "This is a deadline/date question - state dates clearly, indicate if passed/upcoming, include related deadlines.",
    }
    
    sections.append(f"- Query Type: {type_descriptions.get(classification.query_type, 'General inquiry')}")
    
    # Detected intents
    if classification.intents:
        sections.append(f"- Detected Interests: {', '.join(classification.intents[:5])}")
    
    # Implicit needs / proactive info
    if classification.implicit_needs:
        sections.append(f"- Consider Including: {', '.join(classification.implicit_needs[:3])}")
    
    return "\n".join(sections)


def _get_response_structure_guidance(classification: QueryClassification) -> str:
    """Get response structure guidance based on query type"""
    return ResponseTemplates.get_response_structure_prompt(
        classification.query_type,
        classification.user_profile
    )


def _get_proactive_info_guidance(classification: QueryClassification) -> str:
    """Get guidance for proactive information inclusion"""
    return ResponseTemplates.get_proactive_info_prompt(
        classification.intents,
        classification.query_type
    )

def build_user_prompt(query: str, context: str, dt_info: Dict[str, str], 
                      previous_response: str = None,
                      classification: Optional[QueryClassification] = None,
                      user_file_context: Optional[str] = None) -> str:
    """Build the user prompt for the LLM with optional query classification and user-uploaded file context"""
    previous_context = ""
    if previous_response:
        previous_context = f"""
    PREVIOUS ASSISTANT RESPONSE (for resolving references):
    {previous_response}
    """
    
    # Get response structure guidance if classification is available
    structure_guidance = ""
    proactive_guidance = ""
    if classification:
        structure_guidance = f"""
    RESPONSE STRUCTURE GUIDANCE:
    {_get_response_structure_guidance(classification)}
    """
        proactive_info = _get_proactive_info_guidance(classification)
        if proactive_info:
            proactive_guidance = f"""
    {proactive_info}
    """
    
    # Build user-uploaded documents section
    user_file_section = ""
    if user_file_context:
        user_file_section = f"""

    IMPORTANT: The user has uploaded their own documents (shown below under "User-Uploaded Documents").
    If the query is brief or could relate to these uploaded files, focus your answer on analyzing
    the content of the user-uploaded documents. Only fall back to admission documents if the query
    clearly asks about admissions.

    User-Uploaded Documents:
    {user_file_context}
    """
    
    return f"""Based on the following admission documents and conversation history, please answer this question:

    Question: {query}

    Context from SFU Admission Documents:
    {context}{user_file_section}{previous_context}{structure_guidance}{proactive_guidance}

    CRITICAL INSTRUCTIONS - READ CAREFULLY:
    1. ANAPHORA RESOLUTION (for references like "the first one", "it", "that", "the second"):
       - If the user uses ordinal references ("first", "second", "the last one") or pronouns ("it", "that", "them"):
       - FIRST check if they're referring to something from MY PREVIOUS RESPONSE in the conversation history
       - If yes, confidently use that reference WITHOUT asking for clarification
       - ONLY search the documents if the reference is NOT clearly from our conversation
       - Example: If I listed 5 scholarships and user asks "details about the first one", they mean the first I listed, not the first in the documents
    
    2. YOU MUST search through ALL the context documents above to find the answer
    3. If the answer exists in the context (even if partially), you MUST provide it - DO NOT say "the documents do not specify"
    4. For deadline/date questions: Carefully search for dates, deadlines, application periods in the context
    5. If you find ANY mention of dates or deadlines in the context, you MUST include them in your answer
    6. ONLY say "not specified" or "not found" if you have thoroughly searched ALL context documents and confirmed the information is truly absent
    7. DO NOT use any information from your training data that is not in the context
    8. Compare any dates, deadlines, or time-sensitive information in the context with the current date ({dt_info['date']})
    9. When you find a deadline in the context, state it EXACTLY as written: "The deadline is [exact text from context]"
    10. If dates in the context have passed, explicitly inform the user that the information may be outdated
    11. If information appears outdated, DO NOT provide it to the user, unless the user asks for it explicitly.
    12. IMPORTANT: Do NOT mention "Document X", "Source: Document X", or list which documents you used in your response
    13. The sources are automatically displayed separately, so you don't need to reference them

    VERIFICATION STEP: Before saying information is not in the documents, ask yourself:
    - Have I searched through ALL the context documents above?
    - Did I look for variations of the question (e.g., "deadline", "due date", "application date")?
    - Is there ANY mention of this information, even if phrased differently?
    - If this is a follow-up question, did I check my previous response for relevant context?

    Please provide a helpful and accurate answer based ONLY on the context provided. If the information is in the context, you MUST include it."""


def build_context_aware_prompts(query: str, context: str, dt_info: Dict[str, str],
                                classification: QueryClassification,
                                previous_response: str = None,
                                user_file_context: Optional[str] = None) -> tuple[str, str]:
    """
    Build both system and user prompts with full context awareness.
    Returns (system_message, user_prompt) tuple.
    """
    system_message = build_system_message(dt_info, classification)
    user_prompt = build_user_prompt(query, context, dt_info, previous_response, classification, user_file_context)
    return system_message, user_prompt


def get_follow_up_suggestion(classification: QueryClassification) -> Optional[str]:
    """Get a contextual follow-up suggestion based on the query classification"""
    return ResponseTemplates.format_follow_up_question(
        classification.query_type,
        classification.user_profile
    )
