# SFU Admission Chatbot - Backend

A RAG (Retrieval-Augmented Generation) based chatbot for Saint Francis University admission inquiries.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your DeepSeek API key:

```
DEEPSEEK_API_KEY=your_actual_api_key_here
```

**Important:** Never commit your `.env` file to Git! It's already in `.gitignore`.

### 3. Prepare Data File

Place your `merged_rag_data.json` file in the project root directory.

The JSON file should have the following structure:

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

```bash
python main.py
```

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
├── main.py                 # Entry point
├── config.py               # Configuration (reads from .env)
├── .env                    # Environment variables (NOT in git)
├── .env.example            # Template for .env
├── requirements.txt        # Python dependencies
├── merged_rag_data.json    # Your data file
└── src/
    ├── chatbot.py          # Main chatbot class
    ├── query_enhancer.py   # Query enhancement logic
    ├── retrieval.py        # Hybrid retrieval strategies
    ├── prompts.py          # Prompt templates
    ├── adaptive_config.py  # Adaptive parameter adjustment
    ├── utils.py            # Utility functions
    ├── memory.py           # Conversation memory
    ├── llm_provider.py     # LLM API interactions
    ├── vector_db.py        # ChromaDB operations
    └── evaluation.py       # Evaluation dashboard
```

## Configuration

Most settings can be adjusted in `config.py`. Key settings:

- `USE_ADAPTIVE_CONFIG`: Enable/disable automatic parameter adjustment (default: True)
- `RETRIEVAL_K`: Base number of documents to retrieve (default: 5)
- `LLM_MAX_TOKENS`: Maximum response length (default: 1000)
- `CHUNK_SIZE`: Document chunk size for vector DB (default: 1600)

## RAG Evaluation Dashboard

The backend exposes a per-run evaluation API that powers the frontend
Evaluation Dashboard (accessible via the flask-icon button in the header of
the chat UI). It runs Ragas (Faithfulness, Answer Relevancy, Context
Precision, Context Recall) against `eval_testset.json` using the currently
loaded ChromaDB index. Each click in the dashboard produces two saved runs —
an **all-off Baseline** and a user-selected **Optimized** run — so the panel
can be compared side-by-side.

### Endpoints

| Method | Path                          | Description                                                                                     |
| ------ | ----------------------------- | ----------------------------------------------------------------------------------------------- |
| GET    | `/api/ragas/testset`          | Testset metadata (total questions, category breakdown, sample questions).                       |
| POST   | `/api/ragas/run`              | Run one evaluation with explicit strategy toggles. Blocks until complete, returns full run doc. |
| POST   | `/api/ragas/run/stream`       | SSE variant of `/api/ragas/run` — streams per-question progress then a final `run_saved` event. |
| GET    | `/api/ragas/runs`             | List summaries of every saved run, newest first (no per-question detail).                       |
| GET    | `/api/ragas/runs/{run_id}`    | Full saved run (strategies + aggregate + per-question retrieval, answer, and metric scores).    |

Legacy endpoints (`/api/ragas/evaluate`, `/api/ragas/results`) remain in
place for backward compatibility with `run_ragas_evaluation.py`.

### Request body for `/api/ragas/run{,/stream}`

```json
{
  "label": "my_optional_run_label",
  "max_questions": 10,
  "testset_path": "eval_testset.json",
  "strategies": {
    "use_reranker": true,
    "use_adaptive": true,
    "use_dedup": true,
    "use_person_boost": true,
    "use_hybrid": true,
    "use_compression": false
  }
}
```

Each strategy is query-time only, so toggling them does **not** require
re-ingesting ChromaDB. The `/api/chat` singleton is unaffected — evaluation
runs build a transient `RAGChatbot` that shares the already-loaded
ChromaDB, LLM provider, and (if enabled) reranker.

### Run storage format

Runs are persisted as JSON under [`eval_runs/`](./eval_runs/README.md) —
see that README for the full schema. Key fields each run captures:

- `strategies` — the exact toggle combination used.
- `aggregate` — dataset-level Ragas averages (the 4 scorecard numbers).
- `per_question` — per-question retrieved chunks, generated answer, latency,
  and per-metric Ragas scores (powers the Deep Dive modal).
- `dataset_file`, `dataset_mtime`, `chunk_count`, `testset_hash` —
  reproducibility metadata so two runs taken at different times / different
  datasets can be compared unambiguously.

A seeded example (`eval_runs/20260101T000000_baseline_all_off.json`) ships
with the repo so the dashboard is non-empty on first open; it is safe to
delete.

### Chunking / dataset A/B workflow

Chunking decisions happen in `src/vector_db.py` and `merged_rag_data.json`,
not in the dashboard. To compare two chunking or dataset variants:

1. Set `merged_rag_data.json` (or the chunking knobs in
   `ChromaDBManager.create_collection`) to variant **A**.
2. `python ingest_documents.py`.
3. Open the dashboard → Execute A/B with a label like `legacy_chunking_v1`.
4. Set the data / chunking to variant **B**.
5. `python ingest_documents.py`.
6. Open the dashboard → Execute A/B with a label like `recursive_chunking_v2`.
7. Use the run-picker dropdowns on the two comparison panels to load both
   labeled runs side-by-side.

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
- **"merged_rag_data.json not found"**: Place your data file in the project root
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

