# SFU Admission Chatbot - Complete Setup Guide

This guide will help you set up both the backend API server and frontend interface.

## Prerequisites

- Python 3.8+ with pip
- Node.js 18+ and npm
- DeepSeek API key

## Backend Setup

### 1. Navigate to Backend Directory
```bash
cd FYP-BackEnd
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in `FYP-BackEnd/`:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### 5. Start the API Server
```bash
python api_server.py
```

The API will run on `http://localhost:8000`

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Frontend Setup

### 1. Navigate to Frontend Directory
```bash
cd FYP-FrontEnd
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Configure API URL (Optional)
Create a `.env` file in `FYP-FrontEnd/`:
```env
VITE_API_URL=http://localhost:8000
```

If not set, it defaults to `http://localhost:8000`

### 4. Start the Development Server
```bash
npm run dev
```

The frontend will run on `http://localhost:3000`

## Running Both Services

### Option 1: Two Terminal Windows
1. Terminal 1: Run backend (`python api_server.py`)
2. Terminal 2: Run frontend (`npm run dev`)

### Option 2: Using a Process Manager
You can use tools like `concurrently` or `pm2` to run both services together.

## Testing the Application

1. **Start Backend**: Ensure API server is running on port 8000
2. **Start Frontend**: Ensure frontend is running on port 3000
3. **Open Browser**: Navigate to `http://localhost:3000`
4. **Test Chat**: Try asking questions like:
   - "What courses are available in Computer Science?"
   - "Tell me about admission requirements"
   - "What are the scholarship deadlines?"

## Features

### Frontend Features
- 💬 Real-time chat interface
- 📊 Session statistics dashboard
- 📜 Conversation history viewer
- 🎨 Modern, responsive UI
- 📱 Mobile-friendly design

### Backend Features
- RESTful API with FastAPI
- Server-Sent Events (SSE) for streaming
- Conversation memory management
- Performance metrics tracking
- CORS enabled for frontend integration

## Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
# Change port in api_server.py or use:
uvicorn api_server:app --port 8001
```

**Module not found errors:**
```bash
# Make sure virtual environment is activated
pip install -r requirements.txt
```

**API key not found:**
- Ensure `.env` file exists in `FYP-BackEnd/`
- Check that `DEEPSEEK_API_KEY` is set correctly

### Frontend Issues

**Cannot connect to API:**
- Ensure backend is running on port 8000
- Check CORS settings in backend
- Verify `VITE_API_URL` in frontend `.env` file

**Port 3000 already in use:**
```bash
# Vite will automatically use next available port
# Or specify: npm run dev -- --port 3001
```

**Build errors:**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

## Production Deployment

### Backend
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend
```bash
npm run build
# Serve the dist/ folder with a web server (nginx, Apache, etc.)
```

## API Endpoints Reference

- `POST /api/chat` - Send chat message
- `POST /api/chat/stream` - Stream chat response
- `POST /api/clear` - Clear memory
- `GET /api/history` - Get conversation history
- `GET /api/stats` - Get session statistics
- `GET /health` - Health check

## Next Steps

- Customize the UI theme in `tailwind.config.js`
- Add authentication if needed
- Implement persistent storage for conversations
- Add more advanced features (file uploads, etc.)

