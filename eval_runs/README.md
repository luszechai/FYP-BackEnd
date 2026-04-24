# `eval_runs/` — per-run evaluation storage

This directory stores the JSON artefacts produced by the RAG Evaluation
Dashboard (`/api/ragas/run` / `/api/ragas/run/stream`). Every A/B click in
the dashboard persists **two** files here: one for the all-off Baseline and
one for the configured Optimized run.

File naming: `<YYYYMMDD>T<HHMMSS>_<label>.json`

The seeded file `20260101T000000_baseline_all_off.json` is a minimal 3-question
example so the dashboard has something to show on first open and so new
contributors can see the schema without running an evaluation first. It is
safe to delete.

## Run schema

```jsonc
{
  "id": "20260101T000000_baseline_all_off",  // stable run id (filename stem)
  "label": "Seed baseline_all_off",           // human-readable label
  "timestamp": "2026-01-01T00:00:00",         // ISO-8601, used for sort order
  "strategies": {                             // exact toggles the run used
    "use_reranker": false,
    "use_adaptive": false,
    "use_dedup": false,
    "use_person_boost": false,
    "use_hybrid": false,
    "use_compression": false
  },
  "testset_path": "eval_testset.json",
  "testset_hash": "seed00000000",             // sha1 prefix of the testset
  "max_questions": 3,
  "aggregate": {                              // Ragas dataset-level averages
    "faithfulness": 0.8709,
    "answer_relevancy": 0.8044,
    "context_precision": 0.706,
    "context_recall": 0.6903
  },
  "runtime_s": 0,                             // wall-clock for the run
  "dataset_file": "merged_rag_data.json",     // reproducibility metadata
  "dataset_mtime": "2026-01-01T00:00:00",
  "chunk_count": null,                        // ChromaDB chunk count at run time
  "per_question": [
    {
      "user_input": "...",                    // question text
      "reference": "...",                     // ground-truth
      "response": "...",                      // generated answer
      "retrieved_contexts": ["..."],          // chunk text (Ragas input)
      "retrieved_docs": [                     // chunk metadata for Deep Dive
        {
          "id": "...",
          "document": "...",
          "retrieval_score": 0.81,
          "similarity": 0.81,
          "section": "...",
          "source": "...",
          "parent_doc_id": "..."
        }
      ],
      "latency_s": 1.42,                      // pipeline latency
      "scores": {                             // per-question Ragas scores
        "faithfulness": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0,
        "context_recall": 1.0
      }
    }
  ]
}
```

## Chunking / dataset A/B

Chunking isn't a live toggle — it's an ingest-time decision. To compare two
chunking strategies (or two datasets) in the dashboard:

1. Edit `merged_rag_data.json` (or the chunking knobs in `src/vector_db.py`) to
   the "before" configuration.
2. Re-ingest: `python ingest_documents.py`.
3. Open the dashboard → Execute A/B. Give the run a descriptive label (e.g.
   `legacy_chunking_v1`).
4. Swap `merged_rag_data.json` / chunking knobs to the "after" configuration.
5. Re-ingest: `python ingest_documents.py`.
6. Open the dashboard again → Execute A/B with a new label (e.g.
   `recursive_chunking_v2`).
7. In the comparison panels, use the run-picker dropdowns to load your two
   labeled runs side-by-side. `dataset_file`, `dataset_mtime`, `chunk_count`
   and `testset_hash` are captured in each run so the comparison is
   reproducible and auditable.

**Caveat:** when the two datasets differ in content (not just chunking) the
same `eval_testset.json` may no longer be valid. Regenerate it with
`python generate_testset.py` after the second ingest. The dashboard will show
an amber warning banner on the Deep Dive table if the two selected runs have
different `testset_hash` values.
