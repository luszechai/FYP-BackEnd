#!/usr/bin/env python3
"""
Convert crawler output (raw or refined) into merged_rag_data.json for ChromaDBManager.add_documents_from_json.

Input:
  - *_raw.json: uses top-level "crawled_pages"
  - *_refined.json: uses "raw_crawled_data" if present (full crawl payload); do NOT use the AI "pages" list for RAG text.

Usage (from repo root):
  python scripts/crawl_to_merged_rag.py -i output/sfu_sitemap_depth3_raw.json -o merged_rag_data.json
Then delete ./chroma_db (or the collection) and start api_server / main so the empty collection ingests the new file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _page_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "crawled_pages" in data:
        return data["crawled_pages"]
    if "raw_crawled_data" in data:
        return data["raw_crawled_data"]
    raise ValueError(
        "JSON has neither 'crawled_pages' nor 'raw_crawled_data'. "
        "Use *_raw.json or a refined file that still includes raw_crawled_data."
    )


def _build_body_text(content: Dict[str, Any]) -> str:
    parts: List[str] = []
    title = (content.get("title") or "").strip()
    if title:
        parts.append(f"# {title}")
    md = (content.get("meta_description") or "").strip()
    if md:
        parts.append(md)
    tc = (content.get("text_content") or "").strip()
    if tc:
        parts.append(tc)
    for h in content.get("headings") or []:
        if isinstance(h, dict):
            lvl = str(h.get("level") or "h2")
            t = (h.get("text") or "").strip()
            if t:
                parts.append(f"[{lvl}] {t}")
    for para in content.get("paragraphs") or []:
        p = (para or "").strip()
        if p:
            parts.append(p)
    for table in content.get("tables") or []:
        for row in table:
            if isinstance(row, (list, tuple)):
                parts.append(" | ".join(str(c) for c in row))
    return "\n\n".join(parts).strip()


def crawl_json_to_documents(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for page in _page_list(data):
        c = page.get("content") or {}
        if not c.get("crawl_success"):
            continue
        body = _build_body_text(c)
        if not body:
            continue
        sm = page.get("sitemap_item") or {}
        url = (c.get("url") or sm.get("url") or "").strip()
        section = (sm.get("category") or "General").strip()
        out.append(
            {
                "content": body,
                "section": section,
                "metadata": {
                    "source": url or (c.get("title") or "unknown"),
                    "url": url,
                    "structured": False,
                },
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Crawler JSON → merged_rag_data.json for Chroma ingest")
    ap.add_argument("-i", "--input", required=True, type=Path, help="sfu_sitemap_depth*_raw.json or *_refined.json")
    ap.add_argument("-o", "--output", type=Path, default=Path("merged_rag_data.json"), help="Output path")
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    documents = crawl_json_to_documents(data)
    payload = {"documents": documents}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(documents)} documents → {args.output.resolve()}")


if __name__ == "__main__":
    main()
