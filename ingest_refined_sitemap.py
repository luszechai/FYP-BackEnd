"""
Convert sfu_sitemap_depth1_refined.json to RAG documents format and ingest into ChromaDB.

Usage (from project root):
  python ingest_refined_sitemap.py
  python ingest_refined_sitemap.py --file output/sfu_sitemap_depth1_refined.json
  python ingest_refined_sitemap.py --no-ingest   # only convert, do not add to ChromaDB
"""
import argparse
import json
import os
import sys

# Project root on path for config and src
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from src.vector_db import ChromaDBManager


DEFAULT_REFINED_FILE = "output/sfu_sitemap_depth4_refined.json"
DEFAULT_OUTPUT_FILE = "output/sfu_refined_for_rag.json"


def _page_to_doc(page: dict, category_name: str = "General") -> dict | None:
    """Convert a single page dict (with content + sitemap_item) to a RAG document, or None if skip."""
    content_obj = page.get("content", {})
    sitemap_item = page.get("sitemap_item", {})

    if not content_obj.get("crawl_success", True):
        return None

    text = (content_obj.get("text_content") or "").strip()
    if not text:
        paras = content_obj.get("paragraphs", [])
        if paras:
            text = "\n\n".join(p for p in paras if isinstance(p, str) and p.strip())
        if not text or len(text) < 50:
            return None

    title = content_obj.get("title") or sitemap_item.get("title") or category_name
    url = content_obj.get("url") or sitemap_item.get("url", "")

    return {
        "content": text,
        "section": title,
        "metadata": {
            "source": "sfu_sitemap_refined",
            "url": url,
            "source_url": url,
            "section": title,
            "category": category_name,
        },
    }


def refined_json_to_documents(refined_path: str) -> list:
    """Load refined sitemap JSON and return list of {content, section, metadata} for RAG.
    Supports both basic_refinement (categories_analysis[].pages) and AI refinement (raw_crawled_data).
    """
    with open(refined_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    # Source 1: categories_analysis[].pages (basic refinement or when AI includes full pages)
    for cat in data.get("categories_analysis", []):
        category_name = cat.get("name", "General")
        for page in cat.get("pages", []):
            doc = _page_to_doc(page, category_name)
            if doc:
                documents.append(doc)

    # Source 2: raw_crawled_data (AI refinement merges raw pages here; no full pages in categories_analysis)
    if not documents:
        for page in data.get("raw_crawled_data", []):
            cat = page.get("sitemap_item", {}).get("category", "General")
            doc = _page_to_doc(page, cat)
            if doc:
                documents.append(doc)

    return documents


def main():
    parser = argparse.ArgumentParser(description="Convert refined sitemap JSON and ingest into ChromaDB")
    parser.add_argument("--file", default=DEFAULT_REFINED_FILE, help="Path to refined JSON file")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_FILE, help="Path for converted RAG JSON output")
    parser.add_argument("--no-ingest", action="store_true", help="Only convert; do not add to ChromaDB")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"File not found: {args.file}")
        sys.exit(1)

    print(f"Loading refined sitemap: {args.file}")
    documents = refined_json_to_documents(args.file)
    print(f"Extracted {len(documents)} documents")

    rag_data = {"documents": documents}

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rag_data, f, indent=2, ensure_ascii=False)
    print(f"Saved RAG documents to: {args.out}")

    if args.no_ingest:
        print("Skipping ingestion (--no-ingest)")
        return

    if len(documents) == 0:
        print("No documents to ingest; skipping ChromaDB add (Chroma requires non-empty lists).")
        return

    print("\nIngesting into ChromaDB...")
    db = ChromaDBManager()
    db.add_documents_from_json(args.out)
    print(f"Done. Total documents in collection: {db.collection.count()}")


if __name__ == "__main__":
    main()
