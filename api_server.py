"""FastAPI server for SFU Admission Chatbot"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import os
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

# Global chatbot instance
chatbot_instance: Optional[RAGChatbot] = None

# LLM providers keyed by name (populated at startup)
llm_providers: Dict[str, LLMProvider] = {}

# Global RBS client (lives for the duration of the server process)
rbs_client: Optional[RBSClient] = None

# Tracks whether the most recent chat exchange was handled via the RBS path,
# so follow-up queries like "how about march 5" stay in the RBS flow.
_last_exchange_was_rbs: bool = False


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
            enable_cache=Config.LLM_ENABLE_CACHE
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
        
        return ChatResponse(
            answer=response['answer'],
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
    is_details_query = any(kw in q_lower for kw in ("occupied", "details", "slot"))

    if intent in ("room_schedule", "find_free") and not room_name and is_details_query:
        intent = "search_all"
        params["intent"] = "search_all"

    needs_date = intent not in ("list_rooms", "my_bookings")
    needs_time = intent in ("search_all", "find_free") and not is_details_query
    missing_fields = []

    if time_start and not time_end:
        missing_fields.append("time_end")

    if needs_time and not time_start and not time_end:
        missing_fields.append("time")

    if intent in ("room_schedule", "find_free") and not room_name:
        missing_fields.append("room_name")

    if needs_date and not (date or date_from or date_to):
        missing_fields.append("date")

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
        missing_str = ", ".join(sorted(set(missing_fields)))

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

            # Fetch each room's schedule once for the full range, in parallel.
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
                futures = {executor.submit(_check_room_range, r): r for r in rooms_list}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        room_schedules.append(result)

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
            futs = {executor.submit(_fetch_room_single, r): r for r in rooms_list}
            for f in as_completed(futs):
                res = f.result()
                if res:
                    room_schedules_single.append(res)

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

        if is_details_query and occupied_rooms_data:
            # Pre-formatted markdown table sorted by time (LLM includes this verbatim)
            all_events = []
            for rs in occupied_rooms_data:
                code = rs.get("id") or rs.get("name") or ""
                for ev in rs["schedule"]:
                    if ev.get("date") != effective_date:
                        continue
                    btype = ev.get("booking_type", "")
                    title = "Reserved" if btype == "reserved" else (ev.get("title") or "Booked")
                    all_events.append({
                        "start": ev.get("start_time", "?"),
                        "end": ev.get("end_time", "?"),
                        "room": code,
                        "booking": title,
                    })
            all_events.sort(key=lambda x: (x["start"], x["room"]))

            lines.append("")
            lines.append("OCCUPIED_TABLE:")
            lines.append("| Time | Room | Booking |")
            lines.append("|------|------|---------|")
            for ev in all_events:
                lines.append(f"| {ev['start']}–{ev['end']} | {ev['room']} | {ev['booking']} |")
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


def _build_direct_occupied_response(rbs_context: str) -> str:
    """Build a complete markdown response for occupied-details queries without the LLM."""
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", rbs_context)
    if date_match:
        try:
            dt = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            date_label = dt.strftime("%A, %B %d, %Y")
        except ValueError:
            date_label = date_match.group(1)
    else:
        date_label = "the requested date"

    table_lines: List[str] = []
    in_table = False
    for line in rbs_context.split("\n"):
        if line == "OCCUPIED_TABLE:":
            in_table = True
            continue
        if in_table:
            table_lines.append(line)

    parts: List[str] = [
        f"Here are the occupied slot details for **{date_label}**:\n",
    ]

    if table_lines:
        parts.append("\n".join(table_lines))
        parts.append("")

    booking_url = Config.RBS_BOOKING_URL
    parts.append(f"You can book a room here: **{booking_url}**\n")
    parts.append("Suggested follow-ups:")
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

                # --- Occupied-details bypass: build the response directly, skip LLM ---
                if "OCCUPIED_TABLE:" in rbs_context:
                    generation_start = time.time()
                    full_response = _build_direct_occupied_response(rbs_context)
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

                # --- Normal LLM path with dynamic max_tokens ---
                system_message = build_rbs_system_message(dt_info)
                user_prompt = build_rbs_user_prompt(request.query, rbs_context, dt_info)

                conversation_history = None
                if request.use_memory and len(chatbot_instance.memory.history) > 0:
                    conversation_history = chatbot_instance.memory.get_recent_history(n=3)

                original_max_tokens = selected_llm.max_tokens
                estimated = len(rbs_context) // 3 + 500
                selected_llm.max_tokens = max(original_max_tokens, min(estimated, 8192))

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
            
            performance = {
                "total_time": round(retrieval_time + generation_time, 3),
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": round(generation_time, 3),
            }
            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response, 'performance': performance})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/clear")
async def clear_memory():
    """Clear conversation memory and metrics"""
    global _last_exchange_was_rbs
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    chatbot_instance.memory.clear()
    chatbot_instance.session_metrics = []
    chatbot_instance.clear_session_files()
    _last_exchange_was_rbs = False
    
    return {"message": "Memory, metrics, and session files cleared successfully"}


# ---- File Upload Endpoints ----

ALLOWED_UPLOAD_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.txt', '.csv', '.docx'}
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
