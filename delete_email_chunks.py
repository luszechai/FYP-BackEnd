from __future__ import annotations

from src.vector_db import ChromaDBManager


def is_email_meta(meta: dict) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("type") == "email":
        return True
    src = meta.get("source") or ""
    if isinstance(src, str) and src.startswith("email:"):
        return True
    if meta.get("email_id"):
        return True
    return False


def main():
    db = ChromaDBManager("./chroma_db", "sfu_admission")
    col = db.collection
    total = col.count()
    print("Total chunks before:", total)

    batch = 2000
    offset = 0
    to_delete: list[str] = []

    while offset < total:
        res = col.get(include=["metadatas"], limit=batch, offset=offset)
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        for _id, meta in zip(ids, metas):
            if is_email_meta(meta):
                to_delete.append(_id)
        offset += batch

    print("Email chunks to delete:", len(to_delete))
    if not to_delete:
        print("No email chunks found; nothing to delete.")
        return

    # Delete in chunks to avoid request size limits
    chunk = 1000
    deleted = 0
    for i in range(0, len(to_delete), chunk):
        part = to_delete[i : i + chunk]
        col.delete(ids=part)
        deleted += len(part)

    print("Deleted:", deleted)
    print("Total chunks after:", col.count())


if __name__ == "__main__":
    main()

