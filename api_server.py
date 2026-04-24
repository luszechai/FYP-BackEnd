"""FastAPI server for SFU Admission Chatbot"""
import os
# Load .env and set HF timeouts before ANY other import (avoids ReadTimeoutError on slow networks).
# Must patch constants and requests so no code path uses the default 10s for Hugging Face.
from dotenv import load_dotenv
load_dotenv()
_hf_timeout = int(os.environ.get("HF_HUB_ETAG_TIMEOUT", "300") or "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(_hf_timeout))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(_hf_timeout))
try:
    import huggingface_hub.constants as _hf_constants
    _t = _hf_timeout
    _d = int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "300") or "300")
    _hf_constants.HF_HUB_ETAG_TIMEOUT = _t
    _hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = _d
    _hf_constants.DEFAULT_ETAG_TIMEOUT = _t
    _hf_constants.DEFAULT_DOWNLOAD_TIMEOUT = _d
    if hasattr(_hf_constants, "DEFAULT_REQUEST_TIMEOUT"):
        _hf_constants.DEFAULT_REQUEST_TIMEOUT = _d
except ImportError:
    pass
# Ensure requests to huggingface.co never use a short timeout (some code paths may bypass constants).
try:
    import requests
    _orig_request = requests.Session.request
    def _request_with_hf_timeout(self, method, url, *args, **kwargs):
        if isinstance(url, str) and "huggingface.co" in url:
            kwargs["timeout"] = _hf_timeout  # force long timeout; hub often passes 10
        return _orig_request(self, method, url, *args, **kwargs)
    requests.Session.request = _request_with_hf_timeout
except ImportError:
    pass
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple, Callable
from datetime import datetime, timedelta
import uuid
import tempfile
import pandas as pd
from src.chatbot import RAGChatbot
from src.llm_provider import LLMProvider
from src.vector_db import ChromaDBManager
from src.document_loader import DocumentLoaderFactory
from src.evaluation import calculate_hit_rate, generate_evaluation_dashboard, get_available_evaluation_methods
from src.ragas_evaluation import (
    load_testset,
    run_pipeline_on_testset,
    evaluate_with_ragas,
    format_results_summary,
    save_results,
    load_results,
    build_run_id,
    compute_testset_hash,
    list_runs,
    load_run,
    save_run,
    EVAL_RUNS_DIR,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.room_booking import RBSClient
from src.rbs_intent import detect_rbs_intent, extract_rbs_params
from config import Config
import json
import re
import time
import asyncio

app = FastAPI(title="SFU Admission Chatbot API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_email_assets = os.path.abspath(Config.EMAIL_ASSETS_DIR)
os.makedirs(_email_assets, exist_ok=True)
app.mount("/email_assets", StaticFiles(directory=_email_assets), name="email_assets")

# Global chatbot instance
chatbot_instance: Optional[RAGChatbot] = None

# LLM providers keyed by name (populated at startup)
llm_providers: Dict[str, LLMProvider] = {}

# Global RBS client (lives for the duration of the server process)
rbs_client: Optional[RBSClient] = None

# Tracks whether the most recent chat exchange was handled via the RBS path,
# so follow-up queries like "how about march 5" stay in the RBS flow.
_last_exchange_was_rbs: bool = False

# Schedule cache: avoids redundant round-trips to RBS within a conversation.
_rbs_schedule_cache: Dict = {}
_RBS_CACHE_TTL = 300  # 5 minutes


def _get_cached_schedules(req_from: str, req_to: str) -> Optional[List[Dict]]:
    """Return cached room_schedules if the cache covers the requested date range and is fresh."""
    if not _rbs_schedule_cache:
        return None
    cached_from = _rbs_schedule_cache.get("date_from", "")
    cached_to = _rbs_schedule_cache.get("date_to", "")
    ts = _rbs_schedule_cache.get("timestamp", 0)
    if time.time() - ts > _RBS_CACHE_TTL:
        return None
    if req_from >= cached_from and req_to <= cached_to:
        return _rbs_schedule_cache["room_schedules"]
    return None


def _store_schedule_cache(date_from: str, date_to: str, room_schedules: List[Dict]):
    """Store fetched room schedules in the module-level cache."""
    global _rbs_schedule_cache
    _rbs_schedule_cache = {
        "date_from": date_from,
        "date_to": date_to,
        "room_schedules": room_schedules,
        "timestamp": time.time(),
    }


class ChatRequest(BaseModel):
    query: str
    use_memory: bool = True
    provider: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    query: str
    performance: Dict
    sources: List[Dict]
    enhanced_query: Dict


class StatsResponse(BaseModel):
    total_queries: int
    avg_response_time: float
    avg_similarity: float
    hit_rate: float
    metrics: List[Dict]


class HistoryResponse(BaseModel):
    history: List[Dict]
    count: int


class EvalStrategies(BaseModel):
    """Query-time strategy toggles for an evaluation run.

    Defaults mirror the evaluation "baseline" (everything off) so a plain
    POST with no body produces a reproducible legacy run.
    """
    use_reranker: bool = False
    use_adaptive: bool = False
    use_dedup: bool = False
    use_person_boost: bool = False
    use_hybrid: bool = False
    use_compression: bool = False


class EvalRunRequest(BaseModel):
    label: Optional[str] = None
    max_questions: Optional[int] = None
    testset_path: str = "eval_testset.json"
    strategies: EvalStrategies = EvalStrategies()


@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    global chatbot_instance, llm_providers
    try:
        Config.validate()
        
        print("🔧 Setting up chatbot components...")
        llm = LLMProvider(
            provider="deepseek",
            api_key=Config.DEEPSEEK_API_KEY,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            enable_cache=Config.LLM_ENABLE_CACHE,
            request_timeout=Config.LLM_REQUEST_TIMEOUT,
        )
        llm_providers["deepseek"] = llm
        
        if Config.KIMI_API_KEY:
            kimi_llm = LLMProvider(
                provider="kimi",
                api_key=Config.KIMI_API_KEY,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS,
                enable_cache=Config.LLM_ENABLE_CACHE,
                base_url=Config.KIMI_BASE_URL,
                model=Config.KIMI_MODEL,
                kimi_disable_thinking=Config.KIMI_DISABLE_THINKING,
                request_timeout=Config.LLM_REQUEST_TIMEOUT,
            )
            llm_providers["kimi"] = kimi_llm
        else:
            print("⚠️ KIMI_API_KEY not set – Kimi provider will be unavailable.")
        
        db = ChromaDBManager(
            persist_directory=Config.CHROMA_DB_DIR,
            collection_name=Config.CHROMA_COLLECTION_NAME
        )
        if db.collection.count() == 0:
            if os.path.exists(Config.DATA_FILE):
                db.add_documents_from_json(Config.DATA_FILE)
            else:
                print(f"⚠️ {Config.DATA_FILE} not found!")
        else:
            print(f"📚 Loaded {db.collection.count()} documents from persistence.")
        
        chatbot_instance = RAGChatbot(
            chroma_db=db,
            llm_provider=llm,
            use_adaptive_config=Config.USE_ADAPTIVE_CONFIG,
            use_reranker=Config.USE_RERANKER
        )
        
        print(f"✅ Chatbot initialized! Available providers: {list(llm_providers.keys())}")
        
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        import traceback
        traceback.print_exc()


def _get_llm(provider: Optional[str] = None) -> LLMProvider:
    """Resolve the LLM provider for a request, falling back to DeepSeek."""
    if provider and provider in llm_providers:
        return llm_providers[provider]
    return llm_providers.get("deepseek", chatbot_instance.llm)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "SFU Admission Chatbot API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check with chatbot status"""
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot_instance is not None
    }


@app.get("/api/providers")
async def get_providers():
    """List available LLM providers and their models"""
    providers = []
    for name, llm in llm_providers.items():
        providers.append({
            "id": name,
            "label": "DeepSeek" if name == "deepseek" else "Kimi",
            "model": llm.model_name,
        })
    return {"providers": providers, "default": "deepseek"}


# ---- RBS (Room Booking System) Endpoints ----

class RBSLoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/rbs/login")
async def rbs_login(request: RBSLoginRequest):
    """Authenticate with the Room Booking System."""
    global rbs_client
    try:
        client = RBSClient()
        success = client.login(request.username, request.password)
        if success:
            rbs_client = client
            return {"success": True, "username": request.username}
        return {"success": False, "message": "Invalid credentials or login failed."}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/rbs/logout")
async def rbs_logout():
    """Logout from the Room Booking System."""
    global rbs_client
    if rbs_client:
        rbs_client.logout()
    rbs_client = None
    return {"success": True}


@app.get("/api/rbs/status")
async def rbs_status():
    """Check current RBS login status."""
    if rbs_client and rbs_client.is_authenticated:
        return {"logged_in": True, "username": rbs_client.username}
    return {"logged_in": False, "username": None}


@app.get("/api/rbs/debug")
async def rbs_debug():
    """Return discovered rooms and their scheduler IDs as JSON."""
    if rbs_client is None or not rbs_client.is_authenticated:
        raise HTTPException(status_code=400, detail="Not logged in to RBS")
    try:
        rooms = rbs_client.get_rooms(force_refresh=True)
        return {
            "rooms_count": len(rooms),
            "rooms": [
                {"id": r["id"], "scheduler_id": r.get("scheduler_id", ""), "name": r["name"], "type": r.get("type", "")}
                for r in rooms
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def deduplicate_sources(raw_sources: list) -> list:
    """Deduplicate sources by parent_doc_id or source_url to avoid showing multiple chunks from the same document."""
    seen_sources = {}
    sources = []

    for source in raw_sources:
        metadata = source.get('metadata', {})
        section = metadata.get('section', 'Unknown Section')
        source_file = metadata.get('source', '')
        parent_doc_id = metadata.get('parent_doc_id', '')

        source_url = (
            metadata.get('url') or
            metadata.get('link') or
            metadata.get('source_url') or
            (source_file if source_file and (source_file.startswith('http://') or source_file.startswith('https://')) else None)
        )

        if not source_url and source_file:
            if source_file.startswith('/'):
                base_url = Config.SOURCE_BASE_URL
                source_url = base_url + source_file
            elif 'www.' in source_file or '.edu' in source_file or '.hk' in source_file:
                if not source_file.startswith('http'):
                    source_url = 'https://' + source_file
                else:
                    source_url = source_file

        unique_key = parent_doc_id if parent_doc_id else (source_url if source_url else source.get('id', ''))

        if unique_key and unique_key not in seen_sources:
            source_id = f"doc_{parent_doc_id}" if parent_doc_id else source.get('id', '')
            source_name = f"Document {len(seen_sources) + 1} - {section}"

            source_data = {
                "id": source.get('id', ''),
                "source_id": source_id,
                "source_name": source_name,
                "source_url": source_url if source_url else None,
                "section": section,
                "source_file": source_file,
                "document": source.get('document', '')[:500] + "..." if len(source.get('document', '')) > 500 else source.get('document', ''),
                "metadata": metadata,
                "similarity": float(source.get('similarity', 0)),
                "retrieval_score": float(source.get('retrieval_score', 0)),
                "rank": len(seen_sources) + 1
            }

            seen_sources[unique_key] = source_data
            sources.append(source_data)
        elif unique_key in seen_sources:
            existing = seen_sources[unique_key]
            if source.get('retrieval_score', 0) > existing.get('retrieval_score', 0):
                existing['similarity'] = float(source.get('similarity', 0))
                existing['retrieval_score'] = float(source.get('retrieval_score', 0))

    return sources


def filter_cited_sources(response_text: str, sources: list) -> tuple:
    """Keep only the sources that were actually cited as [N] in the response,
    remap citation numbers to be sequential, and return (remapped_text, filtered_sources).
    If no citations found, return originals unchanged."""
    cited_numbers = sorted(set(int(m) for m in re.findall(r'\[(\d+)\]', response_text)))

    if not cited_numbers or not sources:
        return response_text, sources

    old_to_new = {}
    filtered = []
    for new_num, old_num in enumerate(cited_numbers, start=1):
        idx = old_num - 1
        if 0 <= idx < len(sources):
            old_to_new[old_num] = new_num
            source_copy = dict(sources[idx])
            source_copy['rank'] = new_num
            source_copy['source_name'] = re.sub(
                r'^Document \d+', f'Document {new_num}',
                source_copy.get('source_name', f'Document {new_num}'),
            )
            filtered.append(source_copy)

    if not filtered:
        return response_text, sources

    remapped = response_text
    for old_num in sorted(old_to_new.keys(), reverse=True):
        new_num = old_to_new[old_num]
        if old_num != new_num:
            remapped = remapped.replace(f'[{old_num}]', f'[__CITE_{new_num}__]')
    for new_num in old_to_new.values():
        remapped = remapped.replace(f'[__CITE_{new_num}__]', f'[{new_num}]')

    return remapped, filtered


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat query"""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response = chatbot_instance.chat(request.query, use_memory=request.use_memory)
        
        performance = {
            "total_time": float(response['performance']['total_time']),
            "retrieval_time": float(response['performance']['retrieval_time']),
            "generation_time": float(response['performance']['generation_time'])
        }
        
        sources = deduplicate_sources(response.get('sources', []))
        answer, sources = filter_cited_sources(response['answer'], sources)
        
        return ChatResponse(
            answer=answer,
            query=response['query'],
            performance=performance,
            sources=sources,
            enhanced_query=response.get('enhanced_query', {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


def _build_rbs_context(
    params: dict,
    rooms_list: list,
    client: RBSClient,
    user_query: str = "",
) -> str:
    """Dispatch to the correct RBSClient method based on extracted intent and format the result.

    When critical information is missing (e.g. only a start time like "6pm"
    without an end time, or a room-specific intent with no resolvable room),
    this returns a compact context string with ``MISSING:`` markers so the RBS
    prompt can drive a clarifying follow-up question instead of guessing.
    """
    intent = params.get("intent", "room_schedule")
    room_name = params.get("room_name")
    date = params.get("date")
    date_from = params.get("date_from")
    date_to = params.get("date_to")
    time_start = params.get("time_start")
    time_end = params.get("time_end")

    print(
        f"[RBS] intent={intent} | room={room_name} | date={date} | "
        f"range={date_from}->{date_to} | time={time_start}-{time_end}"
    )

    if intent == "book_room":
        intent = "search_all"
        params["intent"] = "search_all"

    # Optional capacity filter inferred from the user query, e.g. "80 people".
    min_capacity: Optional[int] = None
    if user_query:
        m = re.search(r"\b(\d{1,3})\s*(people|persons|seats|capacity)\b", user_query, re.IGNORECASE)
        if m:
            try:
                min_capacity = int(m.group(1))
            except ValueError:
                min_capacity = None

    # --------------------------------------------------------------
    # Pre-search validation: catch booking-rule violations BEFORE
    # any expensive room fetching.
    # --------------------------------------------------------------
    def _bookable_dates_list() -> List[str]:
        today_dt = datetime.now()
        bookable = []
        for off in range(1, 15):
            d = today_dt + timedelta(days=off)
            if d.weekday() == 6:
                continue
            bookable.append(d.strftime("%A %b %d, %Y"))
            if len(bookable) >= 7:
                break
        return bookable

    def _time_options_from(start: str) -> str:
        try:
            ts = datetime.strptime(start, "%H:%M")
            opt1 = (ts + timedelta(hours=1)).strftime("%I:%M %p").lstrip("0")
            opt2 = (ts + timedelta(hours=2)).strftime("%I:%M %p").lstrip("0")
            ts_label = ts.strftime("%I:%M %p").lstrip("0")
            return f"{ts_label} – {opt1} (1 hour); {ts_label} – {opt2} (2 hours)"
        except ValueError:
            return ""

    def _invalid_context(reason: str, detail: str, helpers: List[str]) -> str:
        core_params = {
            "intent": intent,
            "room_name": room_name,
            "date": date,
            "date_from": date_from,
            "date_to": date_to,
            "time_start": time_start,
            "time_end": time_end,
        }
        lines = [
            f"INVALID: {reason}",
            f"DETAIL: {detail}",
            f"USER_QUESTION: {user_query or ''}",
            f"PARAMS: {json.dumps(core_params, ensure_ascii=False)}",
        ]
        if rooms_list:
            room_labels = [f"{r['id']} ({r.get('type', '')})".strip() for r in rooms_list[:15]]
            lines.append("AVAILABLE_ROOMS: " + "; ".join(room_labels))
        lines.extend(helpers)
        return "\n".join(lines)

    if intent not in ("list_rooms", "my_bookings"):
        today_dt = datetime.now()
        tomorrow_dt = (today_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        max_date_dt = (today_dt + timedelta(days=14)).replace(hour=23, minute=59, second=59, microsecond=0)

        # --- Duration validation ---
        if time_start and time_end:
            try:
                ts = datetime.strptime(time_start, "%H:%M")
                te = datetime.strptime(time_end, "%H:%M")
                duration_mins = (te - ts).total_seconds() / 60
                if duration_mins > 120:
                    opts = _time_options_from(time_start)
                    return _invalid_context(
                        "duration_too_long",
                        f"Requested duration is {int(duration_mins)} minutes ({duration_mins/60:.1f} hours). "
                        "The maximum booking duration is 2 hours.",
                        [f"TIME_OPTIONS: {opts}"] if opts else [],
                    )
                if duration_mins < 60:
                    opts = _time_options_from(time_start)
                    return _invalid_context(
                        "duration_too_short",
                        f"Requested duration is {int(duration_mins)} minutes. "
                        "The minimum booking duration is 1 hour.",
                        [f"TIME_OPTIONS: {opts}"] if opts else [],
                    )
            except ValueError:
                pass

        # --- Single-date validation ---
        if date:
            try:
                date_dt = datetime.strptime(date, "%Y-%m-%d")
                if date_dt.weekday() == 6:
                    return _invalid_context(
                        "sunday",
                        f"{date} falls on a Sunday. Rooms cannot be booked on Sundays.",
                        ["BOOKABLE_DATES: " + "; ".join(_bookable_dates_list())],
                    )
                if date_dt < tomorrow_dt:
                    return _invalid_context(
                        "date_past",
                        f"{date} is in the past or is today. Rooms can only be booked from tomorrow onwards.",
                        ["BOOKABLE_DATES: " + "; ".join(_bookable_dates_list())],
                    )
                if date_dt > max_date_dt:
                    return _invalid_context(
                        "date_too_far",
                        f"{date} is more than 14 days from today. "
                        "Rooms can only be booked within a 2-week window.",
                        ["BOOKABLE_DATES: " + "; ".join(_bookable_dates_list())],
                    )
            except ValueError:
                pass

        # --- Date-range pre-filtering ---
        if date_from and date_to:
            try:
                start_dt = datetime.strptime(date_from, "%Y-%m-%d")
                end_dt = datetime.strptime(date_to, "%Y-%m-%d")
                if end_dt < start_dt:
                    start_dt, end_dt = end_dt, start_dt
                valid_dates = []
                cur = start_dt
                while cur <= end_dt:
                    if cur.weekday() != 6 and tomorrow_dt <= cur <= max_date_dt:
                        valid_dates.append(cur.strftime("%Y-%m-%d"))
                    cur += timedelta(days=1)
                if not valid_dates:
                    return _invalid_context(
                        "date_range_empty",
                        f"No bookable dates exist in the range {date_from} to {date_to}. "
                        "Sundays, past dates, and dates beyond 14 days are excluded.",
                        ["BOOKABLE_DATES: " + "; ".join(_bookable_dates_list())],
                    )
                # Narrow the range to only valid dates
                params["date_from"] = valid_dates[0]
                params["date_to"] = valid_dates[-1]
                date_from = valid_dates[0]
                date_to = valid_dates[-1]
            except ValueError:
                pass

    # --------------------------------------------------------------
    # Detect missing critical fields so the LLM can ask follow-ups
    # instead of us guessing (e.g. "6pm" with no end time).
    # Intents that don't need date/time/room: list_rooms, my_bookings
    # --------------------------------------------------------------
    q_lower = (user_query or "").lower()
    is_free_query = any(kw in q_lower for kw in ("free slot", "free room", "free rooms", "all free", "available room", "available slot", "unoccupied"))
    is_details_query = any(kw in q_lower for kw in ("occupied", "details", "slot")) and not is_free_query
    is_summary_query = any(kw in q_lower for kw in ("schedule", "status", "summary")) and not is_details_query and not is_free_query

    if intent in ("room_schedule", "find_free") and not room_name and (is_details_query or is_summary_query or is_free_query):
        intent = "search_all"
        params["intent"] = "search_all"

    needs_date = intent not in ("list_rooms", "my_bookings")
    needs_time = intent in ("search_all", "find_free") and not is_details_query and not is_summary_query and not is_free_query
    missing_fields: List[str] = []

    # Always prioritize asking for the date/date range first when nothing
    # has been established yet, to keep clarification lightweight.
    if needs_date and not (date or date_from or date_to):
        missing_fields.append("date")

    # Then handle time-related gaps (end time or generic "time").
    if time_start and not time_end:
        missing_fields.append("time_end")

    if needs_time and not time_start and not time_end:
        missing_fields.append("time")

    # Room name is only required for room-specific intents.
    if intent in ("room_schedule", "find_free") and not room_name:
        missing_fields.append("room_name")

    if missing_fields:
        core_params = {
            "intent": intent,
            "room_name": room_name,
            "date": date,
            "date_from": date_from,
            "date_to": date_to,
            "time_start": time_start,
            "time_end": time_end,
        }
        # Preserve the order we appended in (date first, then time, then room_name).
        seen = set()
        ordered_missing = [m for m in missing_fields if not (m in seen or seen.add(m))]
        missing_str = ", ".join(ordered_missing)

        lines = [
            f"MISSING: {missing_str}.",
            f"USER_QUESTION: {user_query or ''}",
            f"PARAMS: {json.dumps(core_params, ensure_ascii=False)}",
        ]

        if rooms_list:
            room_labels = [f"{r['id']} ({r.get('type', '')})".strip() for r in rooms_list[:15]]
            lines.append("AVAILABLE_ROOMS: " + "; ".join(room_labels))

        if "date" in missing_fields:
            lines.append("BOOKABLE_DATES: " + "; ".join(_bookable_dates_list()))

        if "time_end" in missing_fields and time_start:
            opts = _time_options_from(time_start)
            if opts:
                lines.append(f"TIME_OPTIONS: {opts}")

        if "time" in missing_fields:
            lines.append(
                "TIME_HINT: Ask the user what time they want. "
                "Bookable hours are 09:00–22:00 on weekdays, 09:00–18:00 on Saturdays. "
                "Booking duration must be 1–2 hours."
            )

        return "\n".join(lines)

    def _resolve_room(name: str) -> Optional[Dict]:
        """Find the room dict matching the user's room reference."""
        if not name:
            return None
        name_lower = name.lower()
        for r in rooms_list:
            if name_lower == r["id"].lower():
                return r
            if name_lower in r.get("name", "").lower():
                return r
        for r in rooms_list:
            if name_lower in r["id"].lower():
                return r
        return None

    def _shorten_title(title: str) -> str:
        """'SOWK302-Social Work Theory...-Tut-[E,F]' → 'SOWK302 Tut [E,F]'"""
        if not title:
            return "Booked"
        parts = title.split("-")
        code = parts[0].strip()
        session = ""
        groups = ""
        for p in parts:
            p = p.strip()
            if p in ("Lect", "Tut", "Lab", "Sem", "Prac"):
                session = p
            if p.startswith("[") and p.endswith("]"):
                groups = " " + p
        if session:
            return f"{code} {session}{groups}"
        return title

    def _format_status_label(booking_type: str, status: str) -> str:
        """Map raw booking_type + status to a user-friendly label."""
        type_label = "Course" if booking_type == "class" else "Reserved"
        status_label = {"approved": "Approved", "confirmed": "Confirmed", "pending": "Pending"}.get(status, status.capitalize() if status else "")
        if status_label:
            return f"{type_label} ({status_label})"
        return type_label

    def _room_area_label(code: str) -> str:
        """Return the area label for a single room code."""
        upper = code.upper()
        if upper.startswith("SP"):
            return "Library/Study Pod"
        if upper.startswith("A"):
            return "SFU"
        if code[0].isdigit():
            return f"CBCC Floor {code[0]}"
        return "Other"

    def _group_rooms_by_area(codes: List[str]) -> str:
        """Group sorted room codes by area/floor for compact display."""
        from collections import OrderedDict
        groups: Dict[str, List[str]] = OrderedDict()
        for code in sorted(codes, key=lambda c: (not c[0].isdigit(), c)):
            upper = code.upper()
            if upper.startswith("SP"):
                key = "Library/Study Pod"
            elif upper.startswith("A"):
                key = "SFU"
            elif code[0].isdigit():
                key = f"CBCC Floor {code[0]}"
            else:
                key = "Other"
            groups.setdefault(key, []).append(code)
        return "\n".join(f"  {area}: {', '.join(rooms)}" for area, rooms in groups.items())

    if intent == "list_rooms":
        return RBSClient.format_rooms_as_text(rooms_list)

    if intent == "my_bookings":
        bookings = client.get_my_bookings()
        return RBSClient.format_my_bookings_as_text(bookings)

    if intent == "search_all":
        # Decide on single-date vs date-range search.
        # Apply optional capacity filter if the user specified a group size.
        rooms_for_search = rooms_list
        if min_capacity is not None:
            def _cap_ok(room: dict) -> bool:
                try:
                    return int(room.get("capacity") or 0) >= min_capacity
                except (ValueError, TypeError):
                    return False

            rooms_for_search = [r for r in rooms_list if _cap_ok(r)]

        if date_from and date_to:
            try:
                start_dt = datetime.strptime(date_from, "%Y-%m-%d")
                end_dt = datetime.strptime(date_to, "%Y-%m-%d")
            except ValueError:
                effective_date = date or date_from or date_to or ""
                available = client.search_available_rooms(effective_date, time_start, time_end)
                return RBSClient.format_available_rooms_as_text(
                    available, effective_date, time_start or "", time_end or ""
                )

            if end_dt < start_dt:
                start_dt, end_dt = end_dt, start_dt

            # Collect valid dates (Sundays already pre-filtered by validation above)
            range_dates: List[str] = []
            cur = start_dt
            while cur <= end_dt:
                range_dates.append(cur.strftime("%Y-%m-%d"))
                cur += timedelta(days=1)

            # Try cache first; fetch from RBS only on miss.
            cached = _get_cached_schedules(date_from, date_to)
            if cached is not None:
                print("[RBS] Cache HIT for date-range search")
                room_schedules = cached
            else:
                print("[RBS] Cache MISS — fetching from RBS")

                def _check_room_range(room: dict) -> Optional[Dict]:
                    sid = room.get("scheduler_id")
                    if not sid:
                        return None
                    schedule = client.get_room_schedule(
                        sid, date_from, date_to, room_code=room.get("id", "")
                    )
                    if schedule is None:
                        return None
                    return {**room, "schedule": schedule}

            room_schedules: List[Dict] = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(_check_room_range, r): r for r in rooms_for_search}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        room_schedules.append(result)

            _store_schedule_cache(date_from, date_to, room_schedules)

            # Build per-day summary by checking each room's cached schedule in-memory
            summary_lines: List[str] = []
            for d_str in range_dates:
                free_codes = []
                for rs in room_schedules:
                    if RBSClient._is_free(rs["schedule"], d_str, time_start, time_end):
                        code = rs.get("id") or rs.get("name") or ""
                        if code:
                            free_codes.append(code)
                if not free_codes:
                    summary_lines.append(f"{d_str}: no free rooms")
                else:
                    summary_lines.append(f"{d_str}: {len(free_codes)} rooms free")
                    summary_lines.append(_group_rooms_by_area(free_codes))

            return "\n".join(summary_lines)

        # No explicit range: single-date search with both free + occupied data.
        effective_date = date or ""

        # Try cache first (single date is a subset of any range that covers it).
        cached_single = _get_cached_schedules(effective_date, effective_date)
        if cached_single is not None:
            print("[RBS] Cache HIT for single-date search")
            room_schedules_single = cached_single
        else:
            print("[RBS] Cache MISS — fetching from RBS (single date)")

            def _fetch_room_single(room: dict) -> Optional[Dict]:
                sid = room.get("scheduler_id")
                if not sid:
                    return None
                sched = client.get_room_schedule(sid, effective_date, room_code=room.get("id", ""))
                if sched is None:
                    return None
                return {**room, "schedule": sched}

            room_schedules_single: List[Dict] = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futs = {executor.submit(_fetch_room_single, r): r for r in rooms_for_search}
                for f in as_completed(futs):
                    res = f.result()
                    if res:
                        room_schedules_single.append(res)

            _store_schedule_cache(effective_date, effective_date, room_schedules_single)

        free_codes: List[str] = []
        occupied_rooms_data: List[Dict] = []
        for rs in room_schedules_single:
            code = rs.get("id") or rs.get("name") or ""
            if RBSClient._is_free(rs["schedule"], effective_date, time_start, time_end):
                free_codes.append(code)
            else:
                occupied_rooms_data.append(rs)

        time_desc = f" between {time_start} and {time_end}" if time_start and time_end else ""
        lines: List[str] = []

        # Free rooms — grouped by area
        if free_codes:
            lines.append(f"FREE rooms on {effective_date}{time_desc} ({len(free_codes)} rooms):")
            lines.append(_group_rooms_by_area(free_codes))
        else:
            lines.append(f"No free rooms found on {effective_date}{time_desc}.")

        if is_free_query:
            # Room-grouped free slots: like OCCUPIED_GROUPED but for free intervals.
            room_free_slots: Dict[str, List[Tuple[str, str]]] = {}
            for rs in room_schedules_single:
                code = rs.get("id") or rs.get("name") or ""
                slots = RBSClient._get_free_slots(
                    rs["schedule"], effective_date,
                    time_filter_start=time_start, time_filter_end=time_end,
                )
                if slots:
                    room_free_slots[code] = slots
            sorted_codes = sorted(room_free_slots, key=lambda c: (not c[0].isdigit(), c))
            lines.append("")
            lines.append("FREE_GROUPED:")
            for code in sorted_codes:
                area = _room_area_label(code)
                lines.append(f"\n**Room {code}** ({area})")
                lines.append("| Time | Status |")
                lines.append("|------|--------|")
                for slot_start, slot_end in room_free_slots[code]:
                    lines.append(f"| {slot_start}–{slot_end} | Free |")

        elif is_summary_query:
            # Compact status summary: one row per room (free + occupied), for the user's time window.
            lines.append("")
            lines.append("STATUS_SUMMARY:")
            lines.append("| Room | Status | Details |")
            lines.append("|------|--------|---------|")
            all_codes = sorted(
                [rs.get("id") or rs.get("name") or "" for rs in room_schedules_single],
                key=lambda c: (not c[0].isdigit(), c),
            )
            occupied_map: Dict[str, List[Dict]] = {}
            for rs in occupied_rooms_data:
                code = rs.get("id") or rs.get("name") or ""
                for ev in rs["schedule"]:
                    if ev.get("date") != effective_date:
                        continue
                    occupied_map.setdefault(code, []).append(ev)
            free_set = set(free_codes)
            for code in all_codes:
                if code in free_set:
                    lines.append(f"| {code} | Free | – |")
                else:
                    evts = occupied_map.get(code, [])
                    if evts:
                        ev = evts[0]
                        btype = ev.get("booking_type", "")
                        status = ev.get("status", "")
                        title = "Reserved" if btype == "reserved" else (ev.get("title") or "Booked")
                        label = _format_status_label(btype, status)
                        lines.append(f"| {code} | {label} | {title} |")
                    else:
                        lines.append(f"| {code} | Occupied | – |")

        elif is_details_query and occupied_rooms_data:
            # Room-grouped occupied table with status column.
            room_events: Dict[str, List[Dict]] = {}
            for rs in occupied_rooms_data:
                code = rs.get("id") or rs.get("name") or ""
                for ev in rs["schedule"]:
                    if ev.get("date") != effective_date:
                        continue
                    room_events.setdefault(code, []).append(ev)
            sorted_codes = sorted(room_events, key=lambda c: (not c[0].isdigit(), c))
            for code in sorted_codes:
                room_events[code].sort(key=lambda e: e.get("start_time", ""))

            lines.append("")
            lines.append("OCCUPIED_GROUPED:")
            for code in sorted_codes:
                area = _room_area_label(code)
                lines.append(f"\n**Room {code}** ({area})")
                lines.append("| Time | Booking | Status |")
                lines.append("|------|---------|--------|")
                for ev in room_events[code]:
                    btype = ev.get("booking_type", "")
                    status = ev.get("status", "")
                    title = "Reserved" if btype == "reserved" else (ev.get("title") or "Booked")
                    label = _format_status_label(btype, status)
                    lines.append(f"| {ev.get('start_time', '?')}–{ev.get('end_time', '?')} | {title} | {label} |")

        elif occupied_rooms_data:
            lines.append(f"\n{len(occupied_rooms_data)} rooms are occupied on {effective_date}.")

        return "\n".join(lines)

    if intent in ("room_schedule", "find_free"):
        room = _resolve_room(room_name)
        if not room:
            return "Could not identify the room. Please specify a room name or number."
        scheduler_id = room.get("scheduler_id")
        if not scheduler_id:
            return (
                f"Room {room['id']} was found but has no scheduler ID in the system. "
                "Cannot fetch its schedule — availability CANNOT be confirmed."
            )

        # If the user asked about a specific room across a date range, build a
        # compact FREE_DATES / BUSY_DATES summary instead of a long schedule.
        if date_from and date_to:
            try:
                start_dt = datetime.strptime(date_from, "%Y-%m-%d")
                end_dt = datetime.strptime(date_to, "%Y-%m-%d")
            except ValueError:
                # Fall back to single-date behaviour if the range is malformed.
                schedule = client.get_room_schedule(scheduler_id, date or "", room_code=room["id"])
                display_name = room_name or room["id"]
                return RBSClient.format_schedule_as_text(schedule, display_name, date or "")

            if end_dt < start_dt:
                start_dt, end_dt = end_dt, start_dt

            schedule = client.get_room_schedule(
                scheduler_id, date_from, date_to, room_code=room["id"]
            )
            display_name = room_name or room["id"]

            if schedule is None:
                # Reuse the standard formatter so the prompt can see that the
                # schedule could not be retrieved.
                return RBSClient.format_schedule_as_text(schedule, display_name, "")

            free_dates: List[str] = []
            busy_dates: List[str] = []
            current = start_dt
            while current <= end_dt:
                d_str = current.strftime("%Y-%m-%d")
                is_free = client._is_free(schedule, d_str, time_start, time_end)
                if is_free:
                    free_dates.append(d_str)
                else:
                    busy_dates.append(d_str)
                current += timedelta(days=1)

            free_str = ", ".join(free_dates) if free_dates else "none"
            busy_str = ", ".join(busy_dates) if busy_dates else "none"
            return f"FREE_DATES: {free_str}; BUSY_DATES: {busy_str}."

        # Default single-date behaviour (existing semantics).
        schedule = client.get_room_schedule(scheduler_id, date or "", room_code=room["id"])
        display_name = room_name or room["id"]
        return RBSClient.format_schedule_as_text(schedule, display_name, date or "")

    return "Unsupported room booking request."


def _extract_date_label(rbs_context: str) -> str:
    """Extract a human-readable date label from the context."""
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", rbs_context)
    if date_match:
        try:
            dt = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            return dt.strftime("%A, %B %d, %Y")
        except ValueError:
            return date_match.group(1)
    return "the requested date"


def _extract_time_window(rbs_context: str) -> str:
    """Extract the time window description (e.g. 'between 15:00 and 16:00') from context."""
    m = re.search(r"between (\d{2}:\d{2}) and (\d{2}:\d{2})", rbs_context)
    if m:
        return f"**{m.group(1)} – {m.group(2)}**"
    return ""


def _extract_free_rooms_text(rbs_context: str) -> str:
    """Extract the FREE rooms lines from rbs_context for inclusion in responses."""
    result = []
    for line in rbs_context.split("\n"):
        if line.startswith("FREE rooms") or (result and not line.startswith(("STATUS_SUMMARY:", "OCCUPIED_GROUPED:", "OCCUPIED_TABLE:"))):
            if line.startswith(("STATUS_SUMMARY:", "OCCUPIED_GROUPED:", "OCCUPIED_TABLE:")):
                break
            result.append(line)
        elif result and line.strip() == "":
            break
    return "\n".join(result) if result else ""


def _build_free_grouped_response(rbs_context: str) -> str:
    """Build room-grouped free slot tables (Time | Status), bypass LLM."""
    date_label = _extract_date_label(rbs_context)
    time_window = _extract_time_window(rbs_context)
    booking_url = Config.RBS_BOOKING_URL

    grouped_lines: List[str] = []
    in_section = False
    for line in rbs_context.split("\n"):
        if line == "FREE_GROUPED:":
            in_section = True
            continue
        if in_section:
            if line.startswith(("STATUS_SUMMARY:", "OCCUPIED_GROUPED:", "FREE rooms on", "No free rooms")):
                break
            grouped_lines.append(line)

    free_count = sum(1 for l in grouped_lines if l.strip().startswith("**Room "))
    time_label = f" {time_window}" if time_window else ""
    if free_count == 0:
        parts: List[str] = [f"No free rooms found on **{date_label}**{time_label}.\n"]
    else:
        parts = [f"Here are the free rooms on **{date_label}**{time_label} ({free_count} rooms):\n"]

    if grouped_lines:
        parts.append("\n".join(grouped_lines))
        parts.append("")

    parts.append(f"You can book a room here: **{booking_url}**\n")
    parts.append("Suggested follow-ups:")
    if time_window:
        parts.append(f"- See room schedule for {time_window} on {date_label}")
    parts.append(f"- See all occupied slots for {date_label}")
    parts.append("- Book a room")
    parts.append("- Check a different date")

    return "\n".join(parts)


def _build_status_summary_response(rbs_context: str) -> str:
    """Build a compact room-status-summary response (free + occupied), bypass LLM."""
    date_label = _extract_date_label(rbs_context)
    time_window = _extract_time_window(rbs_context)
    booking_url = Config.RBS_BOOKING_URL

    table_lines: List[str] = []
    in_table = False
    for line in rbs_context.split("\n"):
        if line == "STATUS_SUMMARY:":
            in_table = True
            continue
        if in_table:
            if line.strip() == "" and table_lines:
                break
            table_lines.append(line)

    time_label = f" for {time_window}" if time_window else ""
    parts: List[str] = [
        f"Here is the room schedule{time_label} on **{date_label}**:\n",
    ]

    if table_lines:
        parts.append("\n".join(table_lines))
        parts.append("")

    parts.append(f"You can book a room here: **{booking_url}**\n")
    parts.append("Suggested follow-ups:")
    parts.append(f"- See all free rooms for {date_label}")
    parts.append(f"- See all occupied slots for {date_label}")
    parts.append("- Check a different date")
    parts.append("- Check a different time")

    return "\n".join(parts)


def _build_occupied_grouped_response(rbs_context: str) -> str:
    """Build a room-grouped occupied details response with status, bypass LLM."""
    date_label = _extract_date_label(rbs_context)
    booking_url = Config.RBS_BOOKING_URL

    grouped_lines: List[str] = []
    in_section = False
    for line in rbs_context.split("\n"):
        if line == "OCCUPIED_GROUPED:":
            in_section = True
            continue
        if in_section:
            grouped_lines.append(line)

    parts: List[str] = [
        f"Here are the occupied slot details for **{date_label}**:\n",
    ]

    if grouped_lines:
        parts.append("\n".join(grouped_lines))
        parts.append("")

    parts.append(f"You can book a room here: **{booking_url}**\n")
    parts.append("Suggested follow-ups:")
    parts.append(f"- See all free rooms for {date_label}")
    time_window = _extract_time_window(rbs_context)
    if time_window:
        parts.append(f"- See room schedule for {time_window} on {date_label}")
    parts.append("- Check a different date")
    parts.append("- Check a different time")

    return "\n".join(parts)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Process a chat query with streaming response"""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    async def generate():
        global _last_exchange_was_rbs
        try:
            from src.utils import get_current_datetime_info
            from src.prompts import build_system_message, build_user_prompt, build_rbs_system_message, build_rbs_user_prompt

            selected_llm = _get_llm(request.provider)

            is_rbs = detect_rbs_intent(selected_llm, request.query, previous_was_rbs=_last_exchange_was_rbs)

            # ---- RBS path ----
            if is_rbs:
                if rbs_client is None or not rbs_client.is_authenticated:
                    _last_exchange_was_rbs = False
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Please log in to the Room Booking System first using the RBS button in the header.'})}\n\n"
                    return

                yield f"data: {json.dumps({'type': 'status', 'message': 'Checking room booking system...'})}\n\n"
                await asyncio.sleep(0)

                dt_info = get_current_datetime_info()
                rooms_list = rbs_client.get_rooms()

                rbs_history = None
                if request.use_memory and len(chatbot_instance.memory.history) > 0:
                    rbs_history = chatbot_instance.memory.get_recent_history(n=3)

                params = extract_rbs_params(
                    chatbot_instance.llm,
                    request.query,
                    rooms_list,
                    today=dt_info['date'],
                    conversation_history=rbs_history,
                )

                yield f"data: {json.dumps({'type': 'status', 'message': 'Fetching room schedules...'})}\n\n"
                await asyncio.sleep(0)

                rbs_context = _build_rbs_context(params, rooms_list, rbs_client, user_query=request.query)

                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
                yield f"data: {json.dumps({'type': 'metadata', 'sources': [], 'enhanced_query': {'original': request.query, 'is_rbs': True}})}\n\n"
                await asyncio.sleep(0)

                # --- Direct-response bypass: build structured response, skip LLM ---
                if "FREE_GROUPED:" in rbs_context:
                    generation_start = time.time()
                    full_response = _build_free_grouped_response(rbs_context)
                elif "STATUS_SUMMARY:" in rbs_context:
                    generation_start = time.time()
                    full_response = _build_status_summary_response(rbs_context)
                elif "OCCUPIED_GROUPED:" in rbs_context:
                    generation_start = time.time()
                    full_response = _build_occupied_grouped_response(rbs_context)
                else:
                    full_response = None

                if full_response is not None:
                    chunk_size = 180
                    for i in range(0, len(full_response), chunk_size):
                        yield f"data: {json.dumps({'type': 'chunk', 'content': full_response[i:i+chunk_size]})}\n\n"
                        await asyncio.sleep(0)
                    generation_time = time.time() - generation_start
                    chatbot_instance.memory.add_exchange(request.query, full_response, [])
                    _last_exchange_was_rbs = True
                    performance = {"total_time": round(generation_time, 3), "retrieval_time": 0.0, "generation_time": round(generation_time, 3)}
                    yield f"data: {json.dumps({'type': 'done', 'full_response': full_response, 'performance': performance})}\n\n"
                    return

                # --- Normal LLM path with dynamic max_tokens and token budget ---
                system_message = build_rbs_system_message(dt_info)
                user_prompt = build_rbs_user_prompt(request.query, rbs_context, dt_info)

                conversation_history = None
                if request.use_memory and len(chatbot_instance.memory.history) > 0:
                    conversation_history = chatbot_instance.memory.get_recent_history(n=3)

                # Approximate token usage (including conversation history)
                # and ensure prompt + completion stay within the model
                # context window. We estimate ~4 characters per token.
                CONTEXT_LIMIT = 8192
                MIN_COMPLETION = 512

                def _estimate_tokens(sys_msg, usr_msg, history):
                    total = len(sys_msg or '') + len(usr_msg or '')
                    if history:
                        for ex in history:
                            total += len(ex.get('user_query', ''))
                            total += len(ex.get('bot_response', ''))
                    return max(total // 4, 1)

                approx_prompt_tokens = _estimate_tokens(
                    system_message, user_prompt, conversation_history
                )

                available = CONTEXT_LIMIT - approx_prompt_tokens - 200
                # If history makes the prompt too large, progressively
                # trim it until the completion budget is acceptable.
                while available < MIN_COMPLETION and conversation_history:
                    conversation_history = conversation_history[1:]
                    if not conversation_history:
                        conversation_history = None
                    approx_prompt_tokens = _estimate_tokens(
                        system_message, user_prompt, conversation_history
                    )
                    available = CONTEXT_LIMIT - approx_prompt_tokens - 200

                available_for_completion = max(available, 256)

                original_max_tokens = selected_llm.max_tokens
                selected_llm.max_tokens = min(original_max_tokens, available_for_completion)

                generation_start = time.time()
                full_response = ""
                try:
                    for chunk in selected_llm.generate_response_stream(
                        prompt=user_prompt,
                        system_message=system_message,
                        conversation_history=conversation_history,
                    ):
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                        await asyncio.sleep(0)
                finally:
                    selected_llm.max_tokens = original_max_tokens

                generation_time = time.time() - generation_start

                chatbot_instance.memory.add_exchange(request.query, full_response, [])
                _last_exchange_was_rbs = True

                performance = {
                    "total_time": round(generation_time, 3),
                    "retrieval_time": 0.0,
                    "generation_time": round(generation_time, 3),
                }
                yield f"data: {json.dumps({'type': 'done', 'full_response': full_response, 'performance': performance})}\n\n"
                return

            # ---- Normal RAG path ----
            _last_exchange_was_rbs = False

            yield f"data: {json.dumps({'type': 'status', 'message': 'Thinking...'})}\n\n"
            await asyncio.sleep(0)

            retrieval_start = time.time()

            retrieved_docs, context, enhanced_query = chatbot_instance.retrieve_context(
                request.query, 
                use_memory=request.use_memory
            )

            retrieval_time = time.time() - retrieval_start

            
            sources = deduplicate_sources(retrieved_docs)

            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
            yield f"data: {json.dumps({'type': 'metadata', 'sources': sources, 'enhanced_query': enhanced_query})}\n\n"
            await asyncio.sleep(0)

            if request.use_memory and len(chatbot_instance.memory.history) > 0:
                memory_n = 3
                conversation_history = chatbot_instance.memory.get_recent_history(n=memory_n)
            else:
                conversation_history = None
            
            dt_info = get_current_datetime_info()
            
            user_file_context = chatbot_instance.format_session_file_context() if chatbot_instance.session_files else None

            system_message = build_system_message(dt_info)
            user_prompt = build_user_prompt(request.query, context, dt_info, user_file_context=user_file_context)
            
            generation_start = time.time()
            full_response = ""
            for chunk in selected_llm.generate_response_stream(
                prompt=user_prompt,
                system_message=system_message,
                conversation_history=conversation_history
            ):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                await asyncio.sleep(0)

            generation_time = time.time() - generation_start
            
            chatbot_instance.memory.add_exchange(
                request.query,
                full_response,
                [doc['id'] for doc in retrieved_docs]
            )

            final_response, cited_sources = filter_cited_sources(full_response, sources)
            
            performance = {
                "total_time": round(retrieval_time + generation_time, 3),
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": round(generation_time, 3),
            }
            yield f"data: {json.dumps({'type': 'done', 'full_response': final_response, 'sources': cited_sources, 'performance': performance})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/clear")
async def clear_memory():
    """Clear conversation memory and metrics"""
    global _last_exchange_was_rbs, _rbs_schedule_cache
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    chatbot_instance.memory.clear()
    chatbot_instance.session_metrics = []
    chatbot_instance.clear_session_files()
    _last_exchange_was_rbs = False
    # Also clear any cached RBS schedules so subsequent RBS queries
    # fetch fresh data instead of reusing potentially stale cache.
    _rbs_schedule_cache = {}
    
    return {
        "message": "Memory, metrics, session files, and RBS schedule cache cleared successfully"
    }


# ---- File Upload Endpoints ----

ALLOWED_UPLOAD_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.txt', '.csv', '.docx', '.xlsx'}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file, extract text, and store in session"""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    # Validate file extension
    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}"
        )
    
    # Check session file limit
    if len(chatbot_instance.session_files) >= chatbot_instance.MAX_SESSION_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum of {chatbot_instance.MAX_SESSION_FILES} files allowed. Remove a file before uploading."
        )
    
    # Read and validate file size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
        )
    
    # Save to temp file for processing
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        # Extract text using DocumentLoaderFactory
        loader = DocumentLoaderFactory()
        documents = loader.load(tmp_path)
        
        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract any text from the file.")
        
        # Combine all extracted text
        extracted_text = "\n\n".join(doc['content'] for doc in documents)
        
        # Generate a unique file ID
        file_id = str(uuid.uuid4())[:8]
        
        # Store in chatbot session
        chatbot_instance.add_session_file(file_id, file.filename, extracted_text)
        
        # Build preview (first 200 chars)
        preview = extracted_text[:200] + ("..." if len(extracted_text) > 200 else "")
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "size": len(contents),
            "text_length": len(extracted_text),
            "preview": preview,
            "truncated": len(extracted_text) > chatbot_instance.MAX_FILE_CHARS
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.delete("/api/upload/{file_id}")
async def remove_uploaded_file(file_id: str):
    """Remove a specific uploaded file from the session"""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    removed = chatbot_instance.remove_session_file(file_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"File with id '{file_id}' not found")
    
    return {"message": f"File '{file_id}' removed successfully"}


@app.get("/api/upload")
async def list_uploaded_files():
    """List all currently uploaded session files (metadata only)"""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    files = chatbot_instance.get_session_files()
    return {"files": files, "count": len(files), "max_files": chatbot_instance.MAX_SESSION_FILES}


@app.get("/api/history", response_model=HistoryResponse)
async def get_history():
    """Get conversation history"""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    history = chatbot_instance.memory.get_recent_history()
    
    return HistoryResponse(
        history=history,
        count=len(history)
    )


@app.get("/api/sources/{source_id}")
async def get_source(source_id: str):
    """Get full source document by ID"""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # Query the vector database for all chunks of this source
        # Extract parent_doc_id from source_id
        if source_id.startswith("doc_"):
            # Find all chunks with this parent_doc_id
            results = chatbot_instance.db.collection.get(
                where={"parent_doc_id": source_id}
            )
            
            if not results['ids']:
                raise HTTPException(status_code=404, detail="Source not found")
            
            # Combine all chunks and sort by chunk_index
            chunks = []
            for i, doc_id in enumerate(results['ids']):
                doc_index = results['metadatas'][i].get('chunk_index', 0)
                chunks.append({
                    'index': doc_index,
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            chunks.sort(key=lambda x: x['index'])
            full_content = '\n\n'.join([chunk['content'] for chunk in chunks])
            
            metadata = chunks[0]['metadata'] if chunks else {}
            
            return {
                "source_id": source_id,
                "section": metadata.get('section', 'Unknown Section'),
                "source_file": metadata.get('source', ''),
                "content": full_content,
                "metadata": metadata,
                "total_chunks": len(chunks)
            }
        else:
            # Single chunk lookup
            results = chatbot_instance.db.collection.get(ids=[source_id])
            if not results['ids']:
                raise HTTPException(status_code=404, detail="Source not found")
            
            return {
                "source_id": source_id,
                "section": results['metadatas'][0].get('section', 'Unknown Section'),
                "source_file": results['metadatas'][0].get('source', ''),
                "content": results['documents'][0],
                "metadata": results['metadatas'][0],
                "total_chunks": 1
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving source: {str(e)}")


@app.get("/api/emails")
async def get_emails():
    """Get all ingested emails grouped by category (scholarship, events, recruitment)."""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    try:
        results = chatbot_instance.db.collection.get(
            where={"type": "email"},
            include=["metadatas", "documents"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying emails: {e}")

    email_chunks: dict = {}
    metadatas = results.get("metadatas", [])
    documents = results.get("documents", [])

    for i, meta in enumerate(metadatas):
        source = meta.get("source", "")
        if not source:
            continue
        if source not in email_chunks:
            email_chunks[source] = {
                "meta": meta,
                "chunks": [],
            }
        email_chunks[source]["chunks"].append({
            "index": meta.get("chunk_index", 0),
            "text": documents[i] if i < len(documents) else "",
        })

    seen_sources: dict = {}
    for source, data in email_chunks.items():
        meta = data["meta"]
        sorted_chunks = sorted(data["chunks"], key=lambda c: c["index"])
        full_content = "\n\n".join(c["text"] for c in sorted_chunks if c["text"])

        eid = meta.get("email_id", "")

        cat_links_raw = meta.get("email_categorized_links", "")
        try:
            cat_links = json.loads(cat_links_raw) if cat_links_raw else []
        except (json.JSONDecodeError, TypeError):
            flat = meta.get("email_links", "")
            cat_links = [{"url": u, "category": "other"} for u in flat.split("\n") if u]

        images_raw = meta.get("email_images", "")
        try:
            image_names = json.loads(images_raw) if images_raw else []
        except (json.JSONDecodeError, TypeError):
            image_names = []
        image_urls = [
            f"/email_assets/{eid}/images/{fname}" for fname in image_names
        ] if eid else []

        has_html = meta.get("email_has_html", "false") == "true"

        seen_sources[source] = {
            "name": meta.get("email_name", ""),
            "subject": meta.get("email_subject", ""),
            "date": meta.get("email_date", ""),
            "type": meta.get("email_type", ""),
            "introduction": meta.get("email_introduction", ""),
            "period": meta.get("email_period", ""),
            "application_period": meta.get("email_application_period", ""),
            "event_period": meta.get("email_event_period", ""),
            "details": meta.get("email_details", ""),
            "fees": meta.get("email_fees", ""),
            "time": meta.get("email_time", ""),
            "event_time": meta.get("email_event_time", ""),
            "requirements": meta.get("email_requirements", ""),
            "links": meta.get("email_links", ""),
            "content": full_content,
            "email_id": eid,
            "categorized_links": cat_links,
            "images": image_urls,
            "has_html": has_html,
        }

    grouped: dict = {
        "scholarship": [],
        "events": [],
        "Member Recruitment": [],
        "Job Recruitment": [],
        "workshop": [],
        "other": [],
    }
    for email_info in seen_sources.values():
        category = (email_info.get("type") or "").strip()
        cat_lower = category.lower()
        if category in ("scholarship", "events"):
            bucket = category
        elif category in ("Member Recruitment", "Job Recruitment"):
            bucket = category
        elif cat_lower == "workshop":
            bucket = "workshop"
        elif category == "recruitment":
            bucket = "other"
        else:
            bucket = "other"
        grouped[bucket].append(email_info)

    total = sum(len(v) for v in grouped.values())
    return {"emails": grouped, "total": total}


@app.get("/api/emails/{email_id}/html")
async def get_email_html(email_id: str):
    """Return the original HTML of an email as plain text."""
    html_path = os.path.join(_email_assets, email_id, "original.html")
    if not os.path.isfile(html_path):
        raise HTTPException(status_code=404, detail="HTML not found for this email")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return PlainTextResponse(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading HTML: {e}")


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats(
    hit_rate_method: str = 'max_similarity',
    hit_rate_threshold: float = 0.5
):
    """Get session statistics with configurable evaluation method"""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not chatbot_instance.session_metrics:
        return StatsResponse(
            total_queries=0,
            avg_response_time=0.0,
            avg_similarity=0.0,
            hit_rate=0.0,
            metrics=[]
        )
    
    df = pd.DataFrame(chatbot_instance.session_metrics)
    
    # Use new evaluation method instead of old 'hit' field
    hit_rate = calculate_hit_rate(
        chatbot_instance.session_metrics,
        method=hit_rate_method,
        threshold=hit_rate_threshold
    ) * 100  # Convert to percentage
    
    # Convert metrics to JSON-serializable format
    metrics = []
    for metric in chatbot_instance.session_metrics:
        metrics.append({
            "query": metric['query'],
            "category": metric['category'],
            "hit": bool(metric['hit']),  # Keep for backward compatibility
            "avg_similarity": float(metric['avg_similarity']),
            "max_similarity": float(metric['max_similarity']),
            "min_similarity": float(metric['min_similarity']),
            "num_docs": int(metric['num_docs']),
            "response_time": float(metric['response_time']),
            "retrieval_time": float(metric['retrieval_time']),
            "generation_time": float(metric['generation_time'])
        })
    
    return StatsResponse(
        total_queries=len(df),
        avg_response_time=float(df['response_time'].mean()),
        avg_similarity=float(df['avg_similarity'].mean()),
        hit_rate=float(hit_rate),  # Use new calculation
        metrics=metrics
    )


@app.post("/api/evaluate")
async def evaluate(
    hit_rate_method: str = 'max_similarity',
    hit_rate_threshold: float = 0.5
):
    """Generate evaluation dashboard"""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not chatbot_instance.session_metrics:
        raise HTTPException(status_code=400, detail="No metrics available. Make some chat requests first.")
    
    try:
        generate_evaluation_dashboard(
            chatbot_instance.session_metrics,
            hit_rate_method=hit_rate_method,
            hit_rate_threshold=hit_rate_threshold
        )
        
        return {
            "message": "Evaluation dashboard generated successfully",
            "method": hit_rate_method,
            "threshold": hit_rate_threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating dashboard: {str(e)}")


@app.get("/api/evaluation/methods")
async def get_evaluation_methods():
    """Get available evaluation methods"""
    return get_available_evaluation_methods()


# ---- Ragas Evaluation Endpoints ----

@app.post("/api/ragas/evaluate")
async def ragas_evaluate(
    testset_path: str = "eval_testset.json",
    max_questions: Optional[int] = None,
    output_path: str = "eval_results.json",
):
    """
    Trigger a Ragas evaluation run using the saved testset.

    Query params:
        testset_path: Path to the testset JSON file (default: eval_testset.json)
        max_questions: Limit evaluation to the first N questions (for quick runs)
        output_path: Where to save the results JSON (default: eval_results.json)

    Returns:
        JSON with aggregate metrics and per-question breakdown.
    """
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    try:
        # Load testset
        testset = load_testset(testset_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Testset file not found: {testset_path}. Run generate_testset.py first.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        # Run pipeline (this is CPU/IO-heavy; run in a thread to avoid blocking)
        loop = asyncio.get_event_loop()
        pipeline_results = await loop.run_in_executor(
            None,
            lambda: run_pipeline_on_testset(chatbot_instance, testset, max_questions),
        )

        if not pipeline_results:
            raise HTTPException(status_code=500, detail="No results from pipeline run.")

        # Evaluate with Ragas
        eval_results = await loop.run_in_executor(
            None,
            lambda: evaluate_with_ragas(pipeline_results),
        )

        # Save results
        save_results(eval_results, output_path)

        return {
            "status": "success",
            "questions_evaluated": len(pipeline_results),
            "aggregate": eval_results.get("aggregate", {}),
            "per_question": eval_results.get("per_question", []),
            "results_saved_to": output_path,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ragas evaluation error: {str(e)}")


@app.get("/api/ragas/results")
async def ragas_results(results_path: str = "eval_results.json"):
    """
    Return the latest saved Ragas evaluation results.

    Query params:
        results_path: Path to the results JSON (default: eval_results.json)

    Returns:
        The full evaluation results JSON (aggregate + per-question breakdown).
    """
    results = load_results(results_path)
    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation results found at {results_path}. Run /api/ragas/evaluate or run_ragas_evaluation.py first.",
        )
    return results


@app.get("/api/ragas/testset")
async def ragas_testset(testset_path: str = "eval_testset.json"):
    """
    Return metadata about the current evaluation testset.

    Query params:
        testset_path: Path to the testset JSON (default: eval_testset.json)

    Returns:
        Testset metadata: question count, sample questions, etc.
    """
    try:
        testset = load_testset(testset_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Testset not found at {testset_path}. Run generate_testset.py first.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build summary metadata
    categories: dict = {}
    for item in testset:
        cat = item.get("category", item.get("metadata", {}).get("category", "unknown"))
        categories[cat] = categories.get(cat, 0) + 1

    # Sample questions (first 5)
    sample_questions = [
        {
            "question": item.get("user_input", item.get("question", "")),
            "has_reference": bool(item.get("reference") or item.get("ground_truth")),
        }
        for item in testset[:5]
    ]

    return {
        "total_questions": len(testset),
        "categories": categories,
        "sample_questions": sample_questions,
        "testset_path": testset_path,
    }


# ---- Evaluation Dashboard: per-run endpoints --------------------------------


def _build_transient_chatbot(strategies: EvalStrategies) -> "RAGChatbot":
    """Build a throwaway RAGChatbot configured for a one-off evaluation run.

    Reuses the already-loaded ChromaDB, LLM provider, and (if requested) the
    existing reranker instance so we don't pay the BGE model-load cost every
    time. The shared ``/api/chat`` singleton is left untouched.
    """
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    shared_reranker = None
    if strategies.use_reranker and getattr(chatbot_instance, "reranker", None) is not None:
        shared_reranker = chatbot_instance.reranker

    return RAGChatbot(
        chroma_db=chatbot_instance.db,
        llm_provider=chatbot_instance.llm,
        use_adaptive_config=strategies.use_adaptive,
        use_reranker=strategies.use_reranker,
        use_dedup=strategies.use_dedup,
        use_compression=strategies.use_compression,
        use_hybrid=strategies.use_hybrid,
        use_person_boost=strategies.use_person_boost,
        reranker=shared_reranker,
    )


def _execute_eval_run(
    req: EvalRunRequest,
    progress_callback: Optional["Callable"] = None,
) -> Dict:
    """Run a full evaluation (pipeline + Ragas) and persist the result.

    Returns the saved run dict. Caller is responsible for surfacing errors.
    """
    try:
        testset = load_testset(req.testset_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Testset file not found: {req.testset_path}. Run generate_testset.py first.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    transient_chatbot = _build_transient_chatbot(req.strategies)

    run_start = time.time()
    pipeline_results = run_pipeline_on_testset(
        transient_chatbot,
        testset,
        max_questions=req.max_questions,
        progress_callback=progress_callback,
    )

    if not pipeline_results:
        raise HTTPException(status_code=500, detail="No results from pipeline run.")

    if progress_callback:
        try:
            progress_callback({"type": "ragas_started", "total": len(pipeline_results)})
        except Exception:
            pass

    eval_results = evaluate_with_ragas(pipeline_results)
    runtime_s = round(time.time() - run_start, 2)

    # Merge per-question Ragas scores with pipeline-level retrieval detail so
    # the Deep Dive modal has retrieval + generation + metrics in one record.
    ragas_by_input: Dict[str, Dict] = {}
    for row in eval_results.get("per_question", []):
        key = row.get("user_input") or row.get("question") or ""
        ragas_by_input[key] = row

    combined_rows: List[Dict] = []
    for pr in pipeline_results:
        key = pr["user_input"]
        scores = ragas_by_input.get(key, {})
        metric_scores = {
            k: v
            for k, v in scores.items()
            if isinstance(v, (int, float)) and k not in ("latency_s",)
        }
        combined_rows.append({
            "user_input": pr["user_input"],
            "reference": pr.get("reference"),
            "response": pr.get("response"),
            "retrieved_contexts": pr.get("retrieved_contexts", []),
            "retrieved_docs": pr.get("retrieved_docs_raw", []),
            "latency_s": pr.get("latency_s"),
            "scores": metric_scores,
        })

    # Dataset metadata for reproducibility / chunking-A-B labeling
    dataset_path = Config.DATA_FILE
    dataset_mtime = None
    try:
        if os.path.exists(dataset_path):
            dataset_mtime = datetime.fromtimestamp(
                os.path.getmtime(dataset_path)
            ).isoformat(timespec="seconds")
    except Exception:
        pass

    try:
        chunk_count = chatbot_instance.db.collection.count()
    except Exception:
        chunk_count = None

    run_id = build_run_id(req.label)
    timestamp = datetime.now().isoformat(timespec="seconds")

    run_doc = {
        "id": run_id,
        "label": req.label or run_id,
        "timestamp": timestamp,
        "strategies": req.strategies.dict(),
        "testset_path": req.testset_path,
        "testset_hash": compute_testset_hash(testset),
        "max_questions": req.max_questions,
        "aggregate": eval_results.get("aggregate", {}),
        "per_question": combined_rows,
        "runtime_s": runtime_s,
        "dataset_file": dataset_path,
        "dataset_mtime": dataset_mtime,
        "chunk_count": chunk_count,
    }

    save_run(run_doc, runs_dir=EVAL_RUNS_DIR)

    if progress_callback:
        try:
            progress_callback({"type": "done", "run_id": run_id})
        except Exception:
            pass

    return run_doc


@app.post("/api/ragas/run")
async def ragas_run(req: EvalRunRequest):
    """
    Run a one-off evaluation with the given strategy toggles and persist it
    under ``eval_runs/<timestamp>_<label>.json``. Non-streaming: the HTTP
    response returns once the run (pipeline + Ragas) completes.

    For a live progress feed use ``/api/ragas/run/stream``.
    """
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    try:
        loop = asyncio.get_event_loop()
        run_doc = await loop.run_in_executor(None, lambda: _execute_eval_run(req))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation run error: {str(e)}")

    return run_doc


@app.post("/api/ragas/run/stream")
async def ragas_run_stream(req: EvalRunRequest):
    """
    Server-Sent Events variant of ``/api/ragas/run``.

    Emits progress events as the pipeline processes each question and a
    final ``event: done`` with the saved run summary. The full run body is
    also persisted and retrievable via ``/api/ragas/runs/{id}``.
    """
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    progress_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _on_progress(event: Dict):
        try:
            loop.call_soon_threadsafe(progress_queue.put_nowait, event)
        except Exception:
            pass

    async def _run_and_signal():
        try:
            run_doc = await loop.run_in_executor(
                None, lambda: _execute_eval_run(req, progress_callback=_on_progress)
            )
            await progress_queue.put({
                "type": "run_saved",
                "run_id": run_doc["id"],
                "aggregate": run_doc.get("aggregate", {}),
                "runtime_s": run_doc.get("runtime_s"),
            })
        except HTTPException as he:
            await progress_queue.put({"type": "error", "status": he.status_code, "detail": he.detail})
        except Exception as e:
            await progress_queue.put({"type": "error", "detail": str(e)})
        finally:
            await progress_queue.put({"type": "__stream_end__"})

    async def event_stream():
        task = asyncio.create_task(_run_and_signal())
        try:
            while True:
                event = await progress_queue.get()
                if event.get("type") == "__stream_end__":
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ragas/runs")
async def ragas_runs_list():
    """Return summaries for every persisted evaluation run, newest first."""
    return {"runs": list_runs(EVAL_RUNS_DIR)}


@app.get("/api/ragas/runs/{run_id}")
async def ragas_runs_detail(run_id: str):
    """Return full detail (strategies, aggregate, per-question) for one run."""
    run = load_run(run_id, EVAL_RUNS_DIR)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
