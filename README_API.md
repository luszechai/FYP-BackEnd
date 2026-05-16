# SFU Admission Chatbot API Server

FastAPI service: chat, retrieval, uploads, room booking (RBS), email listing, and Ragas evaluation. **Site** crawl: `python src/web_crawler.py --depth 3 --crawl4ai` → `output/sfu_sitemap_depth*.json`; convert to ingest JSON with `python scripts/crawl_to_merged_rag.py -i … -o merged_rag_data.json` (see [`README.md`](./README.md) §3). Chroma loads from `DATA_FILE` (default `merged_rag_data.json`) **only when the collection is empty** — delete `./chroma_db` after changing the JSON, then restart. Repo layout: [`README.md`](./README.md).

## Install

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` in the project root (minimum):

```env
DEEPSEEK_API_KEY=your_api_key_here
```

Optional (when using certain UI or eval features):

```env
KIMI_API_KEY=...
GEMINI_API_KEY=...
```

`GEMINI_API_KEY` is read from `config.py` / `src/ragas_evaluation.py` for Ragas judging; configure it when running `/api/ragas/run` or related Ragas pipelines.

## Run the server

### Option A: `python api_server.py` (matches code)

At the bottom of `api_server.py`, `if __name__ == "__main__"` calls `uvicorn.run(app, host="0.0.0.0", port=8001)`, so the default port is **8001**. Point the frontend `VITE_API_URL` at `http://localhost:8001` (or your host/port).

```bash
python api_server.py
```

### Option B: `uvicorn` CLI (port from `--port`)

May **differ** from option A: use whatever `--port` you pass, and update clients / example URLs below accordingly.

```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8001
```

Production-style example:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8001 --workers 4
```

Example URLs below assume **`http://localhost:8001`** (same as option A). If you use `--port 8000` or another port, replace the host/port everywhere.

## API endpoints

Paths match `@app.get` / `@app.post` / `@app.delete` in `api_server.py` in this repo (source of truth).

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Basic health |
| `GET` | `/health` | Health + chatbot init status |

### Chat

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Non-streaming answer + sources + metrics |
| `POST` | `/api/chat/stream` | SSE: metadata, token stream, `done` |

Request body (`ChatRequest`, same for both routes):

```json
{
  "query": "What courses are available?",
  "use_memory": true,
  "provider": "deepseek"
}
```

`provider` is optional (e.g. `"deepseek"`; use `"kimi"` if `KIMI_API_KEY` is set).

### Session and stats

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/clear` | Clear session memory and session metrics |
| `GET` | `/api/history` | Current session conversation history |
| `GET` | `/api/stats` | Session stats (latency, hit rate, citations, etc.) |

### LLM providers

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/providers` | Configured providers (DeepSeek; Kimi if key is set) |

### File upload (session-scoped)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload` | Multipart: attach file for this session |
| `GET` | `/api/upload` | List uploaded files for this session |
| `DELETE` | `/api/upload/{file_id}` | Remove one uploaded file |

### Sources and email

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sources/{source_id}` | Fetch cited source document by ID |
| `GET` | `/api/emails` | Grouped list of ingested emails (Chroma metadata) |
| `GET` | `/api/emails/{email_id}/html` | HTML body for one ingested email |

Static files: `/email_assets` is mounted at startup (`StaticFiles`, root directory `Config.EMAIL_ASSETS_DIR` in `config.py`, default `./email_assets`). The directory may exist but be empty if no email assets were written.

### Room booking (RBS)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/rbs/login` | Log in (session cookie for later RBS calls) |
| `POST` | `/api/rbs/logout` | Log out |
| `GET` | `/api/rbs/status` | Whether the RBS session is valid |
| `GET` | `/api/rbs/debug` | Debug / diagnostics (development) |

### Session evaluation (legacy hit-rate panel)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/evaluate` | Build in-memory session evaluation dashboard from chat metrics |
| `GET` | `/api/evaluation/methods` | Available hit-rate methods / thresholds |

`POST /api/evaluate` has no JSON body; optional **query** params: `hit_rate_method` (default `max_similarity`), `hit_rate_threshold` (default `0.5`), matching `api_server.py`.

### Ragas (test set + persisted runs for the Evaluation Dashboard)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/ragas/testset` | Metadata for `eval_testset.json` |
| `POST` | `/api/ragas/run` | Run pipeline + Ragas; JSON written under `eval_runs/` |
| `POST` | `/api/ragas/run/stream` | Same; SSE progress + final `run_saved` |
| `GET` | `/api/ragas/runs` | Summaries of saved runs |
| `GET` | `/api/ragas/runs/{run_id}` | Full run (per-question + aggregate) |

#### Legacy endpoints (file / CLI oriented)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/ragas/evaluate` | Run Ragas; default output `eval_results.json` |
| `GET` | `/api/ragas/results` | Read `eval_results.json` |

These two routes plus `GET /api/ragas/testset` use **query parameters** (not an `EvalRunRequest` JSON body):

| Path | Query parameters (all optional; defaults in `api_server.py`) |
|------|-------------------------------------------------------------|
| `POST /api/ragas/evaluate` | `testset_path` (default `eval_testset.json`), `max_questions`, `output_path` (default `eval_results.json`) |
| `GET /api/ragas/results` | `results_path` (default `eval_results.json`) |
| `GET /api/ragas/testset` | `testset_path` (default `eval_testset.json`) |

#### Request body for `POST /api/ragas/run` and `POST /api/ragas/run/stream`

Matches Pydantic models **`EvalRunRequest`** / **`EvalStrategies`** in `api_server.py`:

| Field | Type | Default | Description |
|------|------|--------|-------------|
| `label` | `string` \| null | `null` | Optional run label |
| `max_questions` | `int` \| null | `null` | Optional cap on number of questions |
| `testset_path` | `string` | `"eval_testset.json"` | Path to test set |
| `provider` | `string` \| null | `null` | LLM provider (e.g. `deepseek`) |
| `strategies` | object | see below | Query-time strategy toggles |

**`EvalStrategies`** (each field defaults to **`false`** in code; empty body = baseline “all off”):

| Field | Default |
|------|---------|
| `use_reranker` | `false` |
| `use_adaptive` | `false` |
| `use_dedup` | `false` |
| `use_bm25` | `false` |
| `use_person_boost` | `false` |
| `use_hybrid` | `false` |

Example (all strategies on; opposite of defaults, for demo only):

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

These toggles only affect the **temporary** `RAGChatbot` built for evaluation, not the `/api/chat` singleton; flipping them does **not** require re-ingesting Chroma.

## OpenAPI docs

After the server is up (use your bound port):

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## CORS

By default `CORSMiddleware` uses `allow_origins=["*"]`. Tighten in `api_server.py` for production, e.g.:

```python
allow_origins=["http://localhost:3000", "https://yourdomain.com"]
```

## Other notes

- **Chat singleton** is created at startup so the first request does not pay full cold-start cost (vs lazy creation on first request).
- **Session state** (memory, uploads, session metrics) is in-process and is lost on restart.
- **ChromaDB** persistence directory is controlled by `CHROMA_DB_DIR` (default `./chroma_db`); see `config.py`. Ingest JSON defaults to `merged_rag_data.json` (`DATA_FILE`). Refreshing vectors after a new crawl: run `scripts/crawl_to_merged_rag.py`, delete `./chroma_db`, restart the server (see [`README.md`](./README.md) §3).
- **`/api/ragas/run` / `run/stream`** write runs as JSON under `eval_runs/` for dashboard comparison and reproducibility.
- Ragas and long chats may need higher client timeouts; the frontend usually increases timeouts for evaluation calls.
