"""FastAPI server for SFU Admission Chatbot"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
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

# Global RBS client (lives for the duration of the server process)
rbs_client: Optional[RBSClient] = None

# Tracks whether the most recent chat exchange was handled via the RBS path,
# so follow-up queries like "how about march 5" stay in the RBS flow.
_last_exchange_was_rbs: bool = False


class ChatRequest(BaseModel):
    query: str
    use_memory: bool = True


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
    global chatbot_instance
    
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
        
        print("✅ Chatbot initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        import traceback
        traceback.print_exc()


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


def _build_rbs_context(params: dict, rooms_list: list, client: RBSClient) -> str:
    """Dispatch to the correct RBSClient method based on extracted intent and format the result."""
    intent = params.get("intent", "room_schedule")
    room_name = params.get("room_name")
    date = params.get("date")
    time_start = params.get("time_start")
    time_end = params.get("time_end")

    print(f"[RBS] intent={intent} | room={room_name} | date={date} | time={time_start}-{time_end}")

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

    if intent == "list_rooms":
        return RBSClient.format_rooms_as_text(rooms_list)

    if intent == "my_bookings":
        bookings = client.get_my_bookings()
        return RBSClient.format_my_bookings_as_text(bookings)

    if intent == "search_all":
        available = client.search_available_rooms(date or "", time_start, time_end)
        return RBSClient.format_available_rooms_as_text(available, date or "", time_start or "", time_end or "")

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
        schedule = client.get_room_schedule(scheduler_id, date or "", room_code=room["id"])
        display_name = room_name or room["id"]
        return RBSClient.format_schedule_as_text(schedule, display_name, date or "")

    return "Unsupported room booking request."


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

            is_rbs = detect_rbs_intent(request.query, previous_was_rbs=_last_exchange_was_rbs)

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
                params = extract_rbs_params(chatbot_instance.llm, request.query, rooms_list, today=dt_info['date'])

                yield f"data: {json.dumps({'type': 'status', 'message': 'Fetching room schedules...'})}\n\n"
                await asyncio.sleep(0)

                rbs_context = _build_rbs_context(params, rooms_list, rbs_client)

                system_message = build_rbs_system_message(dt_info)
                user_prompt = build_rbs_user_prompt(request.query, rbs_context, dt_info)

                conversation_history = None
                if request.use_memory and len(chatbot_instance.memory.history) > 0:
                    conversation_history = chatbot_instance.memory.get_recent_history(n=3)

                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
                yield f"data: {json.dumps({'type': 'metadata', 'sources': [], 'enhanced_query': {'original': request.query, 'is_rbs': True}})}\n\n"
                await asyncio.sleep(0)

                generation_start = time.time()
                full_response = ""
                for chunk in chatbot_instance.llm.generate_response_stream(
                    prompt=user_prompt,
                    system_message=system_message,
                    conversation_history=conversation_history,
                ):
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    await asyncio.sleep(0)

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
            for chunk in chatbot_instance.llm.generate_response_stream(
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
