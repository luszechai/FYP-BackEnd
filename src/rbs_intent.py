"""Intent detection and parameter extraction for room-booking queries."""
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

_ROOM_PATTERN = re.compile(
    r"\b(?:room\s*)?([A-Za-z]{1,3}\d{1,4}(?:-\d{1,4})?|\d{3,4}(?:-\d{1,4})?[A-Za-z]?)\b",
    re.IGNORECASE,
)

# Fallback keyword lists when LLM is unavailable or fails
_RBS_KEYWORDS = [
    "room", "booking", "book", "booked", "available", "schedule",
    "rbs", "vacancy", "vacant", "occupied", "free room",
    "reserve", "reservation", "study pod", "discussion room",
]
_NON_RBS_KEYWORDS = [
    "admission", "tuition", "fee", "scholarship", "programme", "program",
    "faculty", "professor", "gpa", "ielts", "application form",
    "curriculum", "degree", "bachelor", "master", "credit",
]

_RBS_INTENT_SYSTEM = (
    "You are an intent classifier. Your task is to decide if the user's message is about "
    "room booking (e.g. booking a room, checking room availability, room schedules, "
    "study pods, discussion rooms, reservations, finding free rooms, viewing bookings). "
    "Reply with ONLY a JSON object, no other text. Use this exact format:\n"
    '{"is_rbs": true}  or  {"is_rbs": false}\n'
    "If the message is a follow-up in a room-booking context (e.g. 'how about tomorrow?', "
    "'what about room A?'), treat it as room-booking (is_rbs: true). "
    "If the message is clearly about admissions, programmes, fees, or other university topics "
    "unrelated to rooms, reply is_rbs: false."
)


def _detect_rbs_intent_keyword_fallback(query: str, previous_was_rbs: bool = False) -> bool:
    """Keyword-based fallback when LLM is unavailable or returns invalid response."""
    q = query.lower()
    for kw in _RBS_KEYWORDS:
        if kw in q:
            return True
    if _ROOM_PATTERN.search(query):
        remaining = _ROOM_PATTERN.sub("", query).lower()
        room_context_words = [
            "free", "available", "booked", "schedule", "open",
            "use", "check", "when", "today", "tomorrow",
            "book", "reserve", "status", "slot",
        ]
        if any(w in remaining for w in room_context_words):
            return True
    if previous_was_rbs:
        if any(kw in q for kw in _NON_RBS_KEYWORDS):
            return False
        return True
    return False


def detect_rbs_intent(
    llm: Any,
    query: str,
    previous_was_rbs: bool = False,
) -> bool:
    """Detect whether a query is room-booking related using the LLM.

    Args:
        llm: LLMProvider (or any object with generate_response(prompt=, system_message=)).
        query: The user's message.
        previous_was_rbs: True if the immediately preceding exchange was
            handled via the RBS path. Passed to the LLM so follow-ups are
            classified correctly; fallback also uses this.

    Returns:
        True if the query is about room booking, False otherwise.
    """
    if not query or not query.strip():
        return False

    context_note = ""
    if previous_was_rbs:
        context_note = (
            " The previous user message in this conversation was about room booking. "
            "Treat short follow-ups (e.g. 'tomorrow', 'room B', 'how about 3pm') as room-booking."
        )

    user_content = f"User message:{context_note}\n\n\"{query.strip()}\""
    system = _RBS_INTENT_SYSTEM

    try:
        raw = llm.generate_response(
            prompt=user_content,
            system_message=system,
            use_cache=False,
        )
        cleaned = (raw or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        obj = json.loads(cleaned)
        return bool(obj.get("is_rbs", False))
    except (json.JSONDecodeError, AttributeError, TypeError, Exception):
        return _detect_rbs_intent_keyword_fallback(query, previous_was_rbs)


def extract_rbs_params(llm, query: str, rooms_list: List[Dict], today: Optional[str] = None) -> Dict:
    """Use a single LLM call to extract structured parameters from the user query.

    Returns a dict with keys:
        intent:      one of list_rooms | room_schedule | find_free | search_all | my_bookings
        room_name:   str | None
        date:        YYYY-MM-DD | None
        time_start:  HH:MM | None
        time_end:    HH:MM | None
    """
    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")
    day_of_week = datetime.strptime(today, "%Y-%m-%d").strftime("%A")

    room_names = [f"{r['id']} ({r['name']})" for r in rooms_list] if rooms_list else ["(room list unavailable)"]

    system_prompt = (
        "You are a structured-data extraction assistant. "
        "Given a user query about room bookings, extract the parameters as JSON. "
        "Respond ONLY with a JSON object — no markdown, no explanation.\n\n"
        "Output schema:\n"
        "{\n"
        '  "intent": "list_rooms" | "room_schedule" | "find_free" | "search_all" | "my_bookings",\n'
        '  "room_name": "<room id or name>" | null,\n'
        '  "date": "YYYY-MM-DD" | null,\n'
        '  "time_start": "HH:MM" | null,\n'
        '  "time_end": "HH:MM" | null\n'
        "}\n\n"
        "Intent definitions:\n"
        "- list_rooms: user wants to see all rooms\n"
        "- room_schedule: user asks about a SPECIFIC room's schedule/bookings\n"
        "- find_free: user asks if a specific room is free at a certain time\n"
        "- search_all: user wants to find ANY available room (no specific room)\n"
        "- my_bookings: user asks about their own bookings\n\n"
        "Date resolution:\n"
        f'- Today is {today} ({day_of_week}).\n'
        '- "tomorrow" → the day after today.\n'
        '- "next Monday" → the next occurrence of that weekday.\n'
        "- If no date is mentioned, use today's date.\n\n"
        f"Valid rooms:\n{chr(10).join(room_names)}"
    )

    user_prompt = f"User query: {query}"

    try:
        raw = llm.generate_response(
            prompt=user_prompt,
            system_message=system_prompt,
            use_cache=False,
        )

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        params = json.loads(cleaned)
    except (json.JSONDecodeError, Exception):
        params = _fallback_extract(query, rooms_list, today)

    params.setdefault("intent", "room_schedule")
    params.setdefault("room_name", None)
    params.setdefault("date", today)
    params.setdefault("time_start", None)
    params.setdefault("time_end", None)

    return params


def _fallback_extract(query: str, rooms_list: List[Dict], today: str) -> Dict:
    """Regex-based fallback when LLM parsing fails."""
    q = query.lower()
    result: Dict = {"intent": "room_schedule", "room_name": None, "date": today, "time_start": None, "time_end": None}

    if any(w in q for w in ("my booking", "my reservation")):
        result["intent"] = "my_bookings"
        return result

    if any(w in q for w in ("list room", "all room", "show room", "what room")):
        result["intent"] = "list_rooms"
        return result

    room_match = _ROOM_PATTERN.search(query)
    if room_match:
        result["room_name"] = room_match.group(1)

    if "tomorrow" in q:
        result["date"] = (datetime.strptime(today, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
    if date_match:
        result["date"] = date_match.group(1)

    time_matches = re.findall(r"\b(\d{1,2}:\d{2})\b", query)
    if len(time_matches) >= 2:
        result["time_start"] = time_matches[0]
        result["time_end"] = time_matches[1]
    elif len(time_matches) == 1:
        result["time_start"] = time_matches[0]

    if result["room_name"]:
        if result["time_start"]:
            result["intent"] = "find_free"
        else:
            result["intent"] = "room_schedule"
    else:
        if any(w in q for w in ("available", "free", "open", "vacant", "find")):
            result["intent"] = "search_all"

    return result
