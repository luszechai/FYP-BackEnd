# SFU Admission Chatbot - Backend

A RAG (Retrieval-Augmented Generation) based chatbot for Saint Francis University admission inquiries.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root (this repo may not ship `.env.example`; names must match `config.py`).

Edit `.env` and add your DeepSeek API key:

```
DEEPSEEK_API_KEY=your_actual_api_key_here
```

**Important:** Never commit your `.env` file to Git! It's already in `.gitignore`.

### 3. Prepare data

**Site crawl (current dataset recipe)** — from the repo root, regenerate the sitemap crawl with depth **3** and Crawl4AI-backed fetches:

```bash
python src/web_crawler.py --depth 4 --crawl4ai
```

- `--crawl4ai` forces Crawl4AI (headless browser) for HTML fetches; install with `pip install crawl4ai` and `python -m playwright install chromium` (see `config.py` / crawler logs if deps are missing).
- Raw crawl is written to **`output/sfu_sitemap_depth3_raw.json`**. With default AI refinement (omit `--no-ai`), a refined bundle is written to **`output/sfu_sitemap_depth3_refined.json`**.
- Those files are **crawl bundles** (`crawled_pages` on raw, or `raw_crawled_data` on refined), not the ingest shape below. Convert them with **`scripts/crawl_to_merged_rag.py`** (repo root):

```bash
python scripts/crawl_to_merged_rag.py -i output/sfu_sitemap_depth3_refined.json -o merged_rag_data.json
# or: -i output/sfu_sitemap_depth3_refined.json  (uses raw_crawled_data; do not use the AI "pages" list for RAG text)
```

**Load into Chroma** — `api_server.py` / `main.py` call `add_documents_from_json` **only when the Chroma collection is empty**. After updating `merged_rag_data.json` (or your `DATA_FILE` path): stop the server, **delete `./chroma_db`** (or wipe the target collection), then start again so chunks are rebuilt from the new JSON.

**Chroma ingest format** — whatever file `DATA_FILE` points to (by default `merged_rag_data.json` in the project root) must follow this structure for `ChromaDBManager.add_documents_from_json`:

```json
{
  "documents": [
    {
      "content": "Your document text content here...",
      "section": "Admission Requirements",
      "metadata": {
        "source": "admission_guide.pdf",
        "structured": false
      }
    }
  ]
}
```

### 4. Run the Application

**Interactive CLI (local terminal chat)**

```bash
python main.py
```

**HTTP API (for the React frontend or other clients)**

```bash
python api_server.py
```

`python api_server.py` uses `uvicorn.run(..., port=8001)` at the bottom of `api_server.py`, so it listens on **8001** by default. If you start with `uvicorn api_server:app --port <N>`, **the CLI `--port` wins** and may differ from the script entrypoint. Full routes and request bodies: [`README_API.md`](./README_API.md).

## Features

- **RAG-based Architecture**: Retrieves relevant documents before generating responses
- **Adaptive Configuration**: Automatically adjusts parameters based on query complexity
- **Query Enhancement**: Expands queries for better retrieval (person names, program codes, etc.)
- **Conversation Memory**: Maintains context across multiple exchanges
- **Performance Tracking**: Tracks response times and retrieval metrics
- **Date/Time Awareness**: Provides current date/time context for deadline queries

## Project Structure

```
FYP-BackEnd/
├── main.py                  # CLI chat entry point
├── api_server.py            # FastAPI HTTP server (chat, RBS, Ragas, uploads, …)
├── config.py                # Configuration (reads from .env)
├── fetch_emails_to_rag.py   # Optional: IMAP → Chroma email chunks
├── generate_testset.py      # Build / refresh eval_testset.json
├── run_ragas_evaluation.py  # CLI Ragas helper (legacy file output)
├── scripts/
│   └── crawl_to_merged_rag.py  # output/*_raw.json or refined → merged_rag_data.json
├── output/                  # Crawler writes sfu_sitemap_depth{N}_*.json here
├── merged_rag_data.json     # Default DATA_FILE: root `documents` list for Chroma ingest
├── eval_testset.json        # Ragas question set
├── eval_runs/               # Saved per-dashboard evaluation runs (JSON)
├── chroma_db/               # Persistent Chroma store (local, not always in git)
├── requirements.txt
└── src/
    ├── chatbot.py           # RAGChatbot orchestration
    ├── retrieval.py         # HybridRetriever (dense + BM25 + fusion + boosts)
    ├── bm25_search.py       # Sparse index over Chroma documents
    ├── reranker.py          # Cross-encoder reranking
    ├── vector_db.py         # ChromaDB + chunking / ingest from JSON
    ├── web_crawler.py       # CLI: e.g. --depth 3 --crawl4ai → output/sfu_sitemap_depth*.json
    ├── query_enhancer.py    # Query-type detection + expansions
    ├── query_rewriter.py    # LLM rewrite for retrieval
    ├── adaptive_config.py   # Dynamic retrieval / token / memory knobs
    ├── room_booking.py      # Room booking system client
    ├── rbs_intent.py        # RBS intent detection
    ├── document_loader.py   # PDF / image / office formats for uploads & tooling
    ├── programme_catalog.py # Static programme reference injection
    ├── scholarship_catalog.py
    ├── evaluation.py        # Legacy session metrics dashboard helper
    ├── ragas_evaluation.py  # Ragas scoring (used by API + scripts)
    ├── prompts.py
    ├── memory.py
    ├── llm_provider.py
    └── utils.py
```

## Configuration

Most settings can be adjusted in `config.py`. Key settings:

- `USE_ADAPTIVE_CONFIG`: Enable/disable automatic parameter adjustment (default: True)
- `RETRIEVAL_K`: Base number of documents to retrieve (default: 5)
- `LLM_MAX_TOKENS`: Maximum response length (default: 1024)
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: Defaults in `config.py` (1600 / 200); the active splitter is configured in `src/vector_db.py` — update both if you change chunking policy.

## RAG Evaluation Dashboard

The backend exposes a per-run evaluation API that powers the frontend
Evaluation Dashboard (header control in the chat UI). It runs Ragas (Faithfulness, Answer Relevancy, Context
Precision, Context Recall) against `eval_testset.json` using the currently
loaded ChromaDB index. Each click in the dashboard produces two saved runs —
an **all-off Baseline** and a user-selected **Optimized** run — so the panel
can be compared side-by-side.

### Endpoints (subset)

Common paths for the evaluation dashboard; **full HTTP list and field reference**: [`README_API.md`](./README_API.md).

| Method | Path                       | Description |
| ------ | -------------------------- | ----------- |
| GET    | `/api/ragas/testset`       | Test set metadata (counts, categories, sample questions). |
| POST   | `/api/ragas/run`           | One-shot eval; blocks until done; returns full run document. |
| POST   | `/api/ragas/run/stream`    | Same logic; SSE progress; final `run_saved` event. |
| GET    | `/api/ragas/runs`          | List summaries of saved runs (no per-question detail). |
| GET    | `/api/ragas/runs/{run_id}` | Full run (strategies, aggregate, per-question retrieval/answer/metrics). |

Legacy file/CLI flow: `POST /api/ragas/evaluate`, `GET /api/ragas/results` (**query** parameters, not an `EvalRunRequest` JSON body; aligns with `run_ragas_evaluation.py`, etc.). Parameter table: [`README_API.md`](./README_API.md).

### Request body for `/api/ragas/run` and `/api/ragas/run/stream` (fields as in code)

Matches `EvalRunRequest` / `EvalStrategies`. Every boolean in `strategies` **defaults to `false`** in code (empty body = baseline all-off). Example below turns **all** on for illustration.

```json
{
  "label": "my_optional_run_label",
  "max_questions": 10,
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

Toggles apply only to the **evaluation** query path; you do **not** need to re-ingest Chroma when switching them. The `/api/chat` singleton is unchanged; evaluation builds a **temporary** `RAGChatbot` reusing the loaded Chroma, LLM, and (if enabled) reranker.

### Run storage format

Runs are persisted as JSON files under `eval_runs/` (one file per run). Key fields each run captures:

- `strategies` — the exact toggle combination used.
- `aggregate` — dataset-level Ragas averages (the 4 scorecard numbers).
- `per_question` — per-question retrieved chunks, generated answer, latency,
  and per-metric Ragas scores (powers the Deep Dive modal).
- `dataset_file`, `dataset_mtime`, `chunk_count`, `testset_hash` —
  reproducibility metadata so two runs taken at different times / different
  datasets can be compared unambiguously.

If the folder already contains past runs, the dashboard can load them
immediately; otherwise run one baseline + one optimized evaluation from the UI.

### Chunking / dataset A/B workflow

Chunking decisions happen in `src/vector_db.py` (e.g. `RecursiveCharacterTextSplitter`
chunk size / overlap) and in the contents of the ingest JSON (`merged_rag_data.json` or your `DATA_FILE`), not in the
dashboard. Site HTML is produced separately by `src/web_crawler.py` (e.g. `--depth 3 --crawl4ai`). Ingestion path: on startup, **`main.py` and `api_server.py` both load
from `Config.DATA_FILE` (default `merged_rag_data.json`) only when the Chroma collection is empty** (see
`ChromaDBManager.add_documents_from_json`). To compare two chunking or dataset
variants:

1. Point your ingest JSON (`merged_rag_data.json` or `DATA_FILE` in `config.py`) (and/or edit chunking constants in
   `src/vector_db.py`) to variant **A**.
2. **Rebuild Chroma** so the collection is empty on next boot: stop any running
   server, delete the `./chroma_db` directory (or remove only the target
   collection), then start `python api_server.py` or `python main.py` so documents
   are re-added from JSON.
3. Open the dashboard → run `/api/ragas/run` (or the UI) with a label like
   `legacy_chunking_v1`.
4. Switch the data / chunking to variant **B**, repeat step 2 to rebuild Chroma.
5. Run again with a label like `recursive_chunking_v2`.
6. Use the run-picker on the comparison panels to load both labeled runs
   side-by-side.

If you swapped **content** (not just chunking), also run
`python generate_testset.py` after step 5 so the two runs score against a
valid testset. The Deep Dive table shows a warning banner when the two
selected runs have different `testset_hash` values.

## Security Notes

- **Never commit API keys or `.env` files to Git**
- The `.env` file is already in `.gitignore`
- Use environment variables for all sensitive data
- Rotate API keys if accidentally exposed

## Getting Your DeepSeek API Key

1. Visit https://platform.deepseek.com/
2. Sign up or log in
3. Navigate to API keys section
4. Create a new API key
5. Copy it to your `.env` file

## Troubleshooting

- **"DEEPSEEK_API_KEY is not set"**: Make sure you've created a `.env` file with your API key
- **`merged_rag_data.json` not found** (or wrong `DATA_FILE`): place that file at the project root (default name `merged_rag_data.json`) or change `DATA_FILE` in `config.py` to a JSON with a top-level `documents` array
- **Chat still answers from old facts**: you updated JSON but Chroma was already populated — delete `./chroma_db` and restart so the empty collection re-ingests `DATA_FILE`
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

