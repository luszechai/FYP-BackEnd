# SFU Admission Chatbot API Server



FastAPI server exposing chat, retrieval, uploads, room booking (RBS), email

listing, and Ragas evaluation. For data files, Chroma rebuild, and project

overview see [`README.md`](./README.md).



## Installation



1. Install dependencies:



```bash

pip install -r requirements.txt

```



2. Create a `.env` file in the project root (at minimum):



```env

DEEPSEEK_API_KEY=your_api_key_here

```



Optional (used when enabled in the UI or by evaluation):



```env

KIMI_API_KEY=...

GEMINI_API_KEY=...

```



`GEMINI_API_KEY` is used by the Ragas evaluation path for judging; set it if

you run `/api/ragas/run` or related endpoints.



## Running the Server



### Development Mode



```bash

python api_server.py

```



The `if __name__ == "__main__"` block in `api_server.py` starts Uvicorn on

**port 8000**. Point the frontend `VITE_API_URL` (or your client) at that URL

when using this entrypoint.



Or using uvicorn directly:



```bash

uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

```



The API will be available at `http://localhost:8000` when you use the port

shown in the command above.



### Production Mode



```bash

uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4

```



## API Endpoints



Replace the host/port with whatever you bound (e.g. `localhost:8000` when

running `python api_server.py`).



### Health



| Method | Path | Description |

|--------|------|-------------|

| `GET` | `/` | Basic health |

| `GET` | `/health` | Health + chatbot init status |



### Chat



| Method | Path | Description |

|--------|------|-------------|

| `POST` | `/api/chat` | Non-streaming reply + sources + metrics |

| `POST` | `/api/chat/stream` | Server-Sent Events: metadata, token chunks, done |



Request body (both):



```json

{

  "query": "What courses are available?",

  "use_memory": true,

  "provider": "deepseek"

}

```



`provider` is optional (`"deepseek"` or `"kimi"` when `KIMI_API_KEY` is set).



### Session & stats



| Method | Path | Description |

|--------|------|-------------|

| `POST` | `/api/clear` | Clear conversation memory and session metrics |

| `GET` | `/api/history` | Conversation history for the current session |

| `GET` | `/api/stats` | Session statistics (latency, hit rate, citations, …) |



### LLM providers



| Method | Path | Description |

|--------|------|-------------|

| `GET` | `/api/providers` | List configured providers (DeepSeek, Kimi if key set) |



### File upload (session-scoped)



| Method | Path | Description |

|--------|------|-------------|

| `POST` | `/api/upload` | Multipart: attach files for the current session |

| `GET` | `/api/upload` | List uploaded session files |

| `DELETE` | `/api/upload/{file_id}` | Remove one uploaded file |



### Sources & emails



| Method | Path | Description |

|--------|------|-------------|

| `GET` | `/api/sources/{source_id}` | Fetch source document payload for citations |

| `GET` | `/api/emails` | List ingested email groups (Chroma metadata) |

| `GET` | `/api/emails/{email_id}/html` | HTML body for an ingested email |



Static email assets are mounted at `/email_assets` when present.



### Room Booking System (RBS)



| Method | Path | Description |

|--------|------|-------------|

| `POST` | `/api/rbs/login` | Authenticate (session cookie for subsequent RBS use) |

| `POST` | `/api/rbs/logout` | Clear RBS session |

| `GET` | `/api/rbs/status` | Whether RBS session is active |

| `GET` | `/api/rbs/debug` | Debug / diagnostic (development) |



### Session evaluation (legacy hit-rate dashboard)



| Method | Path | Description |

|--------|------|-------------|

| `POST` | `/api/evaluate` | Build in-memory session evaluation dashboard from chat metrics |

| `GET` | `/api/evaluation/methods` | Available hit-rate methods / thresholds |



### Ragas (testset + saved runs — powers the frontend Evaluation Dashboard)



| Method | Path | Description |

|--------|------|-------------|

| `GET` | `/api/ragas/testset` | Metadata for `eval_testset.json` |

| `POST` | `/api/ragas/run` | Run pipeline + Ragas; persist JSON under `eval_runs/` |

| `POST` | `/api/ragas/run/stream` | Same as `run`, SSE progress + final `run_saved` |

| `GET` | `/api/ragas/runs` | List saved run summaries |

| `GET` | `/api/ragas/runs/{run_id}` | Full run document (per-question + aggregates) |



Legacy (file-based, CLI-friendly):



| Method | Path | Description |

|--------|------|-------------|

| `POST` | `/api/ragas/evaluate` | Run Ragas, write default `eval_results.json` |

| `GET` | `/api/ragas/results` | Read `eval_results.json` |



Minimal body for `/api/ragas/run` and `/api/ragas/run/stream`:



```json

{

  "label": "my_run",

  "max_questions": 20,

  "testset_path": "eval_testset.json",

  "provider": "deepseek",

  "strategies": {

    "use_reranker": true,

    "use_adaptive": true,

    "use_dedup": true,

    "use_bm25": true,

    "use_person_boost": true,

    "use_hybrid": true

  }

}

```



Omit fields to use defaults (baseline strategies are all `false` if you send an

empty body).



## API Documentation



With the server up, OpenAPI is served at:



- Swagger UI: `http://localhost:8000/docs` (same host/port as your Uvicorn bind)

- ReDoc: `http://localhost:8000/redoc`



## CORS Configuration



The API uses `CORSMiddleware` with `allow_origins=["*"]` by default. For

production, restrict origins in `api_server.py`:



```python

allow_origins=["http://localhost:3000", "https://yourdomain.com"]

```



## Notes



- The **chatbot singleton** is created at startup; first request does not pay

  full model load cost beyond startup.

- **Conversation state** (memory, uploads, session metrics) is in-process and

  resets on restart.

- **ChromaDB** is persisted under `CHROMA_DB_DIR` (default `./chroma_db`); see

  `config.py`.

- **Ragas runs** from `/api/ragas/run{,/stream}` are written to `eval_runs/`

  as JSON for the dashboard and reproducibility.

- Long-running requests (Ragas, large chats) may need a higher client timeout;

  the frontend uses extended timeouts for evaluation calls.

