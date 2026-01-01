"""FastAPI server for SFU Admission Chatbot"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import pandas as pd
from src.chatbot import RAGChatbot
from src.llm_provider import LLMProvider
from src.vector_db import ChromaDBManager
from config import Config
import json
import time

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
            use_adaptive_config=Config.USE_ADAPTIVE_CONFIG
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


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat query"""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response = chatbot_instance.chat(request.query, use_memory=request.use_memory)
        
        # Convert numpy types to native Python types for JSON serialization
        performance = {
            "total_time": float(response['performance']['total_time']),
            "retrieval_time": float(response['performance']['retrieval_time']),
            "generation_time": float(response['performance']['generation_time'])
        }
        
        # Clean up sources for JSON serialization
        sources = []
        for source in response.get('sources', []):
            sources.append({
                "id": source.get('id', ''),
                "document": source.get('document', '')[:500] + "..." if len(source.get('document', '')) > 500 else source.get('document', ''),
                "metadata": source.get('metadata', {}),
                "similarity": float(source.get('similarity', 0)),
                "retrieval_score": float(source.get('retrieval_score', 0))
            })
        
        return ChatResponse(
            answer=response['answer'],
            query=response['query'],
            performance=performance,
            sources=sources,
            enhanced_query=response.get('enhanced_query', {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Process a chat query with streaming response"""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    async def generate():
        try:
            # Retrieve context first
            retrieved_docs, context, enhanced_query = chatbot_instance.retrieve_context(
                request.query, 
                use_memory=request.use_memory
            )
            
            if not context:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant information found'})}\n\n"
                return
            
            # Get conversation history for streaming
            if request.use_memory and len(chatbot_instance.memory.history) > 0:
                memory_n = 3
                conversation_history = chatbot_instance.memory.get_recent_history(n=memory_n)
            else:
                conversation_history = None
            
            # Get system message
            from src.utils import get_current_datetime_info
            from src.prompts import build_system_message, build_user_prompt
            
            dt_info = get_current_datetime_info()
            system_message = build_system_message(dt_info)
            user_prompt = build_user_prompt(request.query, context, dt_info)
            
            # Stream response
            full_response = ""
            for chunk in chatbot_instance.llm.generate_response_stream(
                prompt=user_prompt,
                system_message=system_message,
                conversation_history=conversation_history
            ):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            # Update memory
            chatbot_instance.memory.add_exchange(
                request.query,
                full_response,
                [doc['id'] for doc in retrieved_docs]
            )
            
            # Send completion
            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/clear")
async def clear_memory():
    """Clear conversation memory and metrics"""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    chatbot_instance.memory.clear()
    chatbot_instance.session_metrics = []
    
    return {"message": "Memory and metrics cleared successfully"}


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


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get session statistics"""
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
    
    # Convert metrics to JSON-serializable format
    metrics = []
    for metric in chatbot_instance.session_metrics:
        metrics.append({
            "query": metric['query'],
            "category": metric['category'],
            "hit": bool(metric['hit']),
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
        hit_rate=float(df['hit'].sum() / len(df) * 100),
        metrics=metrics
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

