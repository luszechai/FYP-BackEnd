# SFU Admission Chatbot API Server

FastAPI server that exposes the chatbot functionality via REST API.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have a `.env` file with your API key:
```env
DEEPSEEK_API_KEY=your_api_key_here
```

## Running the Server

### Development Mode
```bash
python api_server.py
```

Or using uvicorn directly:
```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Production Mode
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health check with chatbot status

### Chat
- `POST /api/chat` - Send a chat message and get response
  ```json
  {
    "query": "What courses are available?",
    "use_memory": true
  }
  ```

- `POST /api/chat/stream` - Stream chat response (Server-Sent Events)
  ```json
  {
    "query": "What courses are available?",
    "use_memory": true
  }
  ```

### Memory Management
- `POST /api/clear` - Clear conversation memory and metrics
- `GET /api/history` - Get conversation history
- `GET /api/stats` - Get session statistics

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## CORS Configuration

The API is configured to allow CORS from all origins by default. For production, update the `allow_origins` in `api_server.py`:

```python
allow_origins=["http://localhost:3000", "https://yourdomain.com"]
```

## Notes

- The chatbot is initialized on server startup
- All conversation state is maintained in memory (resets on server restart)
- For persistent storage, consider adding database integration

