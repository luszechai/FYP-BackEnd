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
from src.evaluation import calculate_hit_rate, generate_evaluation_dashboard, get_available_evaluation_methods
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
        
        # Clean up sources for JSON serialization and add source links
        # Deduplicate sources by parent_doc_id or source_url to avoid showing multiple chunks from same document
        seen_sources = {}  # key: (parent_doc_id or source_url), value: source data
        sources = []
        
        for source in response.get('sources', []):
            metadata = source.get('metadata', {})
            section = metadata.get('section', 'Unknown Section')
            source_file = metadata.get('source', '')
            parent_doc_id = metadata.get('parent_doc_id', '')
            
            # Extract source URL - the 'source' field in metadata contains the URL
            # Check for URL in metadata (could be 'url', 'link', 'source_url', or 'source' field)
            source_url = (
                metadata.get('url') or 
                metadata.get('link') or 
                metadata.get('source_url') or
                (source_file if source_file and (source_file.startswith('http://') or source_file.startswith('https://')) else None)
            )
            
            # If no URL found, try to construct from source_file
            if not source_url and source_file:
                # If source_file looks like a URL path, construct full URL
                if source_file.startswith('/'):
                    # Use base URL from config
                    base_url = Config.SOURCE_BASE_URL
                    source_url = base_url + source_file
                elif 'www.' in source_file or '.edu' in source_file or '.hk' in source_file:
                    # Add https:// if missing
                    if not source_file.startswith('http'):
                        source_url = 'https://' + source_file
                    else:
                        source_url = source_file
            
            # Use parent_doc_id or source_url as unique key for deduplication
            # Prefer parent_doc_id if available, otherwise use source_url
            unique_key = parent_doc_id if parent_doc_id else (source_url if source_url else source.get('id', ''))
            
            # Only add if we haven't seen this source before
            if unique_key and unique_key not in seen_sources:
                # Generate source identifier
                source_id = f"doc_{parent_doc_id}" if parent_doc_id else source.get('id', '')
                source_name = f"Document {len(seen_sources) + 1} - {section}"
                
                source_data = {
                    "id": source.get('id', ''),
                    "source_id": source_id,
                    "source_name": source_name,
                    "source_url": source_url if source_url else None,  # External URL to original source (None if not available)
                    "section": section,
                    "source_file": source_file,
                    "document": source.get('document', '')[:500] + "..." if len(source.get('document', '')) > 500 else source.get('document', ''),
                    "metadata": metadata,
                    "similarity": float(source.get('similarity', 0)),
                    "retrieval_score": float(source.get('retrieval_score', 0)),
                    "rank": len(seen_sources) + 1  # Re-rank after deduplication
                }
                
                seen_sources[unique_key] = source_data
                sources.append(source_data)
            elif unique_key in seen_sources:
                # Update similarity/score if this chunk has higher score
                existing = seen_sources[unique_key]
                if source.get('retrieval_score', 0) > existing.get('retrieval_score', 0):
                    existing['similarity'] = float(source.get('similarity', 0))
                    existing['retrieval_score'] = float(source.get('retrieval_score', 0))
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

