#!/usr/bin/env python3
"""
Fetch emails from an IMAP account, process them (including attachments),
and ingest the resulting text into the chatbot's ChromaDB.

Behaviour:
- Fetch messages from a specific sender since the last successful run.
- Classify emails as scholarship (獎學金), events (活動), or recruitment (招募).
- Use Kimi v2.5 vision to read image attachments (no OCR).
- Extract text from non-image attachments (PDF, TXT, CSV, DOCX, XLSX) with
  existing document loaders.
- Combine everything (body + attachments + image descriptions) into ONE
  document per email with structured metadata, then ingest into ChromaDB.
- Preserve all links found in emails.
"""

import base64
import email
import hashlib
import imaplib
import json
import os
import re
import sqlite3
import sys
import time
import random
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.header import decode_header as _decode_header_parts
from email.message import Message
from typing import Dict, List, Optional, Tuple

from config import Config
from src.llm_provider import LLMProvider
from src.vector_db import ChromaDBManager
from src.document_loader import DocumentLoaderFactory

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
_IMAGE_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_JOBS_EVENTS_PHRASE = "[Jobs & Events]"
_JOBS_EVENTS_TYPES = ["Job Recruitment", "events"]
_VALID_EMAIL_TYPES = {
    "scholarship",
    "events",
    "Member Recruitment",
    "Job Recruitment",
    "workshop",
}

_KIMI_CACHE_VERSION = "kimi_cache_v3"
_KIMI_CACHE_DB_PATH = os.path.join(Config.CHROMA_DB_DIR, "kimi_extract_cache.sqlite3")

_TPD_HINTS = (
    "TPD rate limit",
    "tokens per day",
    "reached organization TPD rate limit",
)


def _is_tpd_limit_error(exc: Exception) -> bool:
    s = str(exc)
    lower = s.lower()
    return any(h.lower() in lower for h in _TPD_HINTS)


@contextmanager
def _kimi_cache_conn():
    os.makedirs(os.path.dirname(_KIMI_CACHE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_KIMI_CACHE_DB_PATH, timeout=30)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kimi_cache (
                fingerprint TEXT PRIMARY KEY,
                created_utc TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA journal_mode=WAL;")
        yield conn
    finally:
        conn.close()


def _kimi_cache_get(fingerprint: str) -> Optional[Dict]:
    try:
        with _kimi_cache_conn() as conn:
            row = conn.execute(
                "SELECT result_json FROM kimi_cache WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            if not row:
                return None
            return json.loads(row[0])
    except Exception:
        print("⚠️ Kimi cache read failed; continuing without cache.")
        return None


def _kimi_cache_put(fingerprint: str, result: Dict) -> None:
    try:
        payload = json.dumps(result, ensure_ascii=False)
        with _kimi_cache_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kimi_cache (fingerprint, created_utc, result_json) VALUES (?, ?, ?)",
                (fingerprint, datetime.now(timezone.utc).isoformat(), payload),
            )
            conn.commit()
    except Exception:
        # cache must never break ingestion
        print("⚠️ Kimi cache write failed; continuing without cache.")
        return


def _normalize_text_for_llm(text: str) -> str:
    if not text:
        return ""
    # Normalize newlines early for stable fingerprints
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Drop common quoted-reply separators and long footers
    stop_markers = (
        "-----Original Message-----",
        "________________________________",
        "Sent from my iPhone",
        "Sent from my Android",
        "Unsubscribe",
        "unsubscribe",
        "To unsubscribe",
    )
    lines: List[str] = []
    all_lines = t.splitlines()
    total = len(all_lines)
    tail_start = max(0, total - 40)
    for idx, line in enumerate(all_lines):
        # Only treat unsubscribe/footer markers as terminators near the end.
        if idx >= tail_start and any(m in line for m in stop_markers):
            break
        # Typical reply marker: "On ... wrote:"
        if re.search(r"^On .{0,120}wrote:\s*$", line):
            break
        # Strip very long base64-ish junk lines
        if len(line) > 4000 and re.fullmatch(r"[A-Za-z0-9+/=]+", line.strip() or "x"):
            continue
        lines.append(line)

    t = "\n".join(lines)
    # Collapse excessive whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{4,}", "\n\n\n", t).strip()
    return t


def _truncate_for_llm(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 60] + "\n\n[TRUNCATED]\n"


def _strip_jsonish_and_urls(text: str) -> str:
    """Remove JSON-ish blobs / urls from extracted text fields."""
    if not text:
        return ""
    t = _normalize_text_for_llm(text)
    # Drop obvious dict/list fragments that sometimes leak from model output
    t = re.sub(r"\[[^\]]*\burl\b[^\]]*\]", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\{[^}]*\burl\b[^}]*\}", " ", t, flags=re.IGNORECASE)
    t = re.sub(r'"\s*,\s*"\w+"\s*:\s*".*?"', " ", t, flags=re.IGNORECASE)
    # Strip raw urls in requirements-like fields
    t = re.sub(r"https?://[^\s)>\]]+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,;\"'")
    return t.strip()


_URL_RE = re.compile(r"\b(https?://[^\s)>\]]+)", re.IGNORECASE)
_DATE_ITEM_RE = re.compile(
    r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b(?:\s*\([^)]+\))?",
    re.IGNORECASE,
)


def _first_url(text: str) -> str:
    if not text:
        return ""
    m = _URL_RE.search(text)
    return (m.group(1) or "").strip() if m else ""


def _tighten_deadline_snippet(s: str) -> str:
    """Keep just the date-ish part plus a nearby time token if present."""
    if not s:
        return ""
    t = _normalize_text_for_llm(s)
    m = _DATEISH_RE.search(t)
    if not m:
        return t.strip().strip(" ,;\"'")
    start, end = m.span()
    date_part = t[start:end].strip()
    tail = t[end : end + 24]
    time_m = re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", tail, flags=re.IGNORECASE)
    if time_m:
        return f"{date_part} {time_m.group(0).strip()}".strip()
    # Chinese time tokens like "下午11時59分"
    zh_time_m = re.search(r"(上午|下午)?\s*\d{1,2}\s*[時点]\s*\d{1,2}\s*分?", tail)
    if zh_time_m:
        return f"{date_part} {zh_time_m.group(0).strip()}".strip()
    return date_part


def _parse_various_deadlines(text: str) -> List[str]:
    """
    Parse lines like:
      Various deadlines: 28 Apr 2026 (X), 29 Apr 2026 (Y) ...
    into a list of items.
    """
    if not text:
        return []
    m = re.search(r"\bVarious deadlines?\s*[:\-]\s*(.+)$", text, flags=re.IGNORECASE)
    if not m:
        return []
    body = m.group(1).strip()
    if not body:
        return []
    parts = re.split(r",\s*(?=\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b)", body)
    out: List[str] = []
    for p in parts:
        p = _normalize_text_for_llm(p).strip().strip(",")
        if p:
            out.append(p)
    return out


def _local_extract_event_fields(text: str, discovered_links: List[str], categorized_links: Optional[List[Dict]] = None) -> Dict[str, str]:
    """
    Rule-first extraction to reduce LLM usage.
    Returns additive fields that won't break existing consumers.
    """
    t = _normalize_text_for_llm(text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    joined = "\n".join(lines)

    # Deadline
    deadlines = _parse_various_deadlines(joined)
    if not deadlines:
        m = re.search(
            r"\b(?:deadline|apply by|application deadline|registration deadline)\s*[:\-]\s*([^\n]{4,160})",
            joined,
            flags=re.IGNORECASE,
        )
        if m:
            deadlines = [m.group(1).strip()]
    if not deadlines:
        # common phrasing without ':' e.g. "open ... until 16 July 23:59"
        m = re.search(r"\buntil\s+([^\n]{3,80})", joined, flags=re.IGNORECASE)
        if m and _DATEISH_RE.search(m.group(1) or ""):
            deadlines = [m.group(1).strip()]
    if not deadlines:
        # "on or before 16 July 2026 (11:59 pm)"
        m = re.search(r"\bon\s+or\s+before\s+([^\n]{3,80})", joined, flags=re.IGNORECASE)
        if m and _DATEISH_RE.search(m.group(1) or ""):
            deadlines = [m.group(1).strip()]
    if not deadlines:
        # Chinese: "...於7月16日下午11時59分截止..."
        m = re.search(r"(?:於|至)\s*([^\n]{0,40}?\d{1,2}\s*月\s*\d{1,2}\s*[日號]?[^\n]{0,20}?)\s*(?:截止|前)", joined)
        if m and _DATEISH_RE.search(m.group(1) or ""):
            deadlines = [m.group(1).strip()]

    # Filter obvious false positives like tail fragments of "accepted"
    filtered: List[str] = []
    for d in deadlines:
        d = _normalize_text_for_llm(d).strip().strip(" ,;\"'")
        if not d:
            continue
        if not _DATEISH_RE.search(d) and re.search(r"\baccepted\b", d, flags=re.IGNORECASE):
            continue
        if not _DATEISH_RE.search(d) and len(d) <= 6:
            continue
        filtered.append(_tighten_deadline_snippet(d))
    deadlines = filtered
    application_deadline = "; ".join(deadlines[:8]).strip()

    # Period (event date range / date&time line)
    event_period = ""
    for key_re in (
        r"\b(?:event|workshop|course)\s*(?:date|dates|time|period|duration)\s*[:\-]\s*([^\n.]{6,160})",
        r"\b(?:date\s*&\s*time|date/time)\s*[:\-]\s*([^\n.]{6,160})",
    ):
        m = re.search(key_re, joined, flags=re.IGNORECASE)
        if m and m.group(1):
            cand = m.group(1).strip()
            if "deadline" not in cand.lower():
                event_period = cand
                break
    if not event_period:
        # Date range "DD Mon YYYY - DD Mon YYYY"
        m = re.search(
            r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}[^.\n]{0,60})\s*(?:to|\-|–|—)\s*([^\n.]{6,80})",
            joined,
            flags=re.IGNORECASE,
        )
        if m:
            cand = f"{m.group(1).strip()} - {m.group(2).strip()}"
            if "deadline" not in cand.lower():
                event_period = cand

    # Location (only explicit keys; avoid false positives)
    location = ""
    m = re.search(r"\b(?:location|venue|where)\s*[:\-]\s*([^\n.]{3,120})", joined, flags=re.IGNORECASE)
    if m:
        location = m.group(1).strip()

    # Requirements: try explicit section; strip urls/jsonish
    requirements = ""
    m = re.search(r"\b(?:requirements?|eligibility|who can apply|criteria)\s*[:\-]\s*([^\n]{6,240})", joined, flags=re.IGNORECASE)
    if m:
        requirements = m.group(1).strip()
    requirements = _strip_jsonish_and_urls(requirements)

    # Application link: prefer categorized enrollment/info, else any discovered url
    application_link = ""
    if categorized_links:
        for lk in categorized_links:
            if isinstance(lk, dict) and lk.get("url"):
                cat = (lk.get("category") or "").lower()
                if cat in ("enrollment", "registration", "apply", "application"):
                    application_link = lk["url"]
                    break
        if not application_link:
            for lk in categorized_links:
                if isinstance(lk, dict) and lk.get("url"):
                    application_link = lk["url"]
                    break
    if not application_link:
        # Score discovered links to avoid picking WhatsApp/mailto etc.
        candidates = list(discovered_links or [])
        if not candidates:
            # fall back to urls in body text
            candidates = [m.group(1) for m in _URL_RE.finditer(joined)]

        def score(u: str) -> int:
            s = u.lower()
            sc = 0
            if "mailto:" in s:
                return -1000
            if "whatsapp" in s:
                sc -= 50
            if any(k in s for k in ("apply", "application", "register", "registration", "enrol", "enroll", "forms", "/student")):
                sc += 30
            if any(k in s for k in ("linkreit.com", "hkcsssol.org.hk", "office.com", "microsoft.com")):
                sc += 10
            return sc

        candidates = sorted({c.strip() for c in candidates if c and c.strip()}, key=score, reverse=True)
        application_link = candidates[0] if candidates else ""

    return {
        "application_deadline": application_deadline,
        "event_period": event_period,
        "location": location,
        "requirements": requirements,
        "application_link": application_link,
    }


def _cheap_type_hint(subject: str, body: str, attachment_text: str) -> List[str]:
    """Ultra-cheap local hinting to avoid LLM on low-value emails."""
    s = f"{subject}\n{body}\n{attachment_text}".lower()
    hits: List[str] = []
    if any(k in s for k in ("scholarship", "scholarships", "獎學金")):
        hits.append("scholarship")
    if any(k in s for k in ("workshop", "seminar", "certificate", "course", "活動", "event")):
        hits.append("workshop")
    if any(k in s for k in ("career", "internship", "job", "recruit", "position", "vacancy", "招聘")):
        hits.append("Job Recruitment")
    if any(k in s for k in ("event", "events", "活動", "join us", "register")):
        hits.append("events")
    # de-dupe while preserving order
    out: List[str] = []
    for t in hits:
        if t not in out:
            out.append(t)
    return out


def _should_call_llm(forced_types: List[str], hinted_types: List[str]) -> bool:
    # Only do expensive extraction for these buckets
    allow = {"events", "workshop", "Job Recruitment", "Member Recruitment", "scholarship"}
    return any(t in allow for t in (forced_types or hinted_types))

_CLASSIFICATION_SYSTEM_MESSAGE = (
    "You are processing university emails for a student information chatbot.\n\n"
    "Given an email (subject, body, attachment text, and optionally images), you must:\n\n"
    "1. CLASSIFY the email as exactly one of these types:\n"
    '   - "scholarship" (獎學金) - scholarship information, financial aid\n'
    '   - "events" (活動) - campus events, activities, workshops, seminars\n'
    '   - "Member Recruitment" (成員招募) - member recruitment, club recruitment\n'
    '   - "Job Recruitment" (職位招募) - job postings, volunteer recruitment\n'
    '   - "workshop" (工作坊) - workshop, seminar, training\n\n'
    f'   IMPORTANT: if the email subject/body contains the exact phrase "{_JOBS_EVENTS_PHRASE}", '
    'it belongs to BOTH "Job Recruitment" and "events". Use "Job Recruitment" as "type" '
    'and include both values in "types".\n\n'
    '2. EXTRACT structured information (leave empty string "" if not found):\n'
    "   - name: The name/title of the scholarship, event, or recruitment\n"
    '   - introduction: Introduction of the scholarship, event, or recruitment(a short summary of the email, or extraction of the main content of the email)\n'
    '   - application_period: Application / registration / enrolment period or deadline '
    '(e.g. "2024-01-01 to 2024-03-31"; if only a deadline exists, copy it exactly)\n'
    '   - event_period: Event period/date, not the registration period '
    '(e.g. "2024-01-01 to 2024-03-31")\n'
    "   - details: Brief description of what this is about\n"
    '   - fees: Any fees or costs involved (e.g. "$100", "Free", "")\n'
    "   - event_time: Specific date/time of event or deadline\n"
    "   - requirements: Eligibility requirements or qualifications needed\n"
    "   - links: List of objects for each URL found, with 'url' and 'category'.\n"
    "     Categories: 'enrollment' (registration/application forms), 'info' (information pages),\n"
    "     'social' (social media like Instagram, Facebook), 'contact' (mailto links, phone),\n"
    "     'other' (anything else).\n\n"
    "3. If images are attached, describe their content and extract any text visible in them.\n\n"
    "4. Produce a combined document that includes ALL the information from the email body, "
    "attachments, and images in a single coherent text. Preserve ALL links. "
    "Keep content in its original language, with an English summary at the end.\n\n"
    "Return your response as a JSON object with these exact keys:\n"
    "{\n"
    '  "type": "scholarship" | "events" | "Member Recruitment"| "Job Recruitment" | "workshop",\n'
    '  "types": ["scholarship" | "events" | "Member Recruitment" | "Job Recruitment" | "workshop"],\n'
    '  "name": "...",\n'
    '  "introduction": "...",\n'
    '  "application_period": "...",\n'
    '  "event_period": "...",\n'
    '  "details": "...",\n'
    '  "fees": "...",\n'
    '  "event_time": "...",\n'
    '  "requirements": "...",\n'
    '  "links": [{"url": "https://...", "category": "enrollment|info|social|contact|other"}, ...],\n'
    '  "full_text": "The complete combined document text with all content..."\n'
    "}\n\n"
    "Return ONLY the JSON object, no markdown code blocks or other text."
)


# ---------------------------------------------------------------------------
# Utility helpers (IMAP, timestamps, body extraction)
# ---------------------------------------------------------------------------

def _decode_mime_header(raw: str) -> str:
    """Decode a MIME-encoded header (e.g. =?utf-8?B?...?=) into a plain string."""
    if not raw:
        return ""
    parts = _decode_header_parts(raw)
    decoded: List[str] = []
    for data, charset in parts:
        if isinstance(data, bytes):
            decoded.append(data.decode(charset or "utf-8", errors="ignore"))
        else:
            decoded.append(data)
    return " ".join(decoded).strip()


def _get_last_run(path: str) -> datetime:
    """Read last run timestamp from file; default to 7 days ago if missing."""
    if not os.path.exists(path):
        return datetime.now(timezone.utc) - timedelta(days=7)
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = f.read().strip()
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.now(timezone.utc) - timedelta(days=7)


def _set_last_run(path: str, when: Optional[datetime] = None) -> None:
    """Write last run timestamp to file."""
    when = when or datetime.now(timezone.utc)
    with open(path, "w", encoding="utf-8") as f:
        f.write(when.isoformat())


def _connect_imap() -> imaplib.IMAP4_SSL:
    """Connect to the IMAP server using Config / environment variables."""
    host = (Config.EMAIL_IMAP_HOST or "").strip()
    user = (Config.EMAIL_IMAP_USER or "").strip()
    password = Config.EMAIL_IMAP_PASSWORD
    port = Config.EMAIL_IMAP_PORT

    if not host or not user or not password:
        raise RuntimeError(
            "EMAIL_IMAP_HOST, EMAIL_IMAP_USER, and EMAIL_IMAP_PASSWORD must be set "
            "in the environment or .env file."
        )

    try:
        client = imaplib.IMAP4_SSL(host, port)
        client.login(user, password)
        return client
    except imaplib.IMAP4.error as e:
        err_bytes = e.args[0] if e.args else b""
        if b"AUTHENTICATIONFAILED" in err_bytes or b"Invalid credentials" in err_bytes:
            raise RuntimeError(
                "IMAP login failed (invalid credentials). For Gmail (imap.gmail.com), "
                "you must use an App Password, not your normal password: Google Account → "
                "Security → 2-Step Verification → App passwords → generate one for Mail."
            ) from e
        raise
    except OSError as e:
        if "getaddrinfo" in str(e).lower() or (hasattr(e, "errno") and e.errno == 11001):
            raise RuntimeError(
                f"The IMAP host '{host}' could not be resolved (DNS failed). "
                "Check that EMAIL_IMAP_HOST is correct. For Microsoft 365 / Office 365 use "
                "outlook.office365.com; if your mail is on-campus only, you may need VPN. "
                "Check .env and try again."
            ) from e
        raise


def _build_search_criteria(last_run: datetime) -> List[bytes]:
    """Build IMAP search criteria for messages since last_run (and optional sender filter).

    Supports comma-separated addresses in EMAIL_FROM_FILTER.
    Multiple addresses are combined with IMAP OR so emails from *any*
    of the listed senders are matched.
    """
    since_date = last_run.strftime("%d-%b-%Y")
    criteria: List[bytes] = [b"SINCE", since_date.encode("ascii")]
    if Config.EMAIL_FROM_FILTER:
        senders = [s.strip() for s in Config.EMAIL_FROM_FILTER.split(",") if s.strip()]
        if len(senders) == 1:
            criteria.extend([b"FROM", senders[0].encode("utf-8")])
        elif len(senders) > 1:
            # IMAP OR is binary: OR <crit1> <crit2>
            # For N senders we nest: OR (OR (FROM a) (FROM b)) (FROM c)
            from_parts: List[bytes] = []
            from_parts.extend([b"FROM", senders[0].encode("utf-8")])
            for sender in senders[1:]:
                from_parts = [b"OR"] + from_parts + [b"FROM", sender.encode("utf-8")]
            criteria.extend(from_parts)
    return criteria


def _get_body_from_message(msg: Message) -> str:
    """Extract a reasonable text body from an email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = (part.get("Content-Disposition") or "").lower()
            if content_type == "text/plain" and "attachment" not in disposition:
                try:
                    return part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", errors="ignore"
                    )
                except Exception:
                    continue

        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = (part.get("Content-Disposition") or "").lower()
            if content_type == "text/html" and "attachment" not in disposition:
                try:
                    html = part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", errors="ignore"
                    )
                    return _html_to_text(html)
                except Exception:
                    continue
    else:
        if msg.get_content_type() == "text/plain":
            try:
                return msg.get_payload(decode=True).decode(
                    msg.get_content_charset() or "utf-8", errors="ignore"
                )
            except Exception:
                return msg.get_payload()

    return ""


def _html_to_text(html: str) -> str:
    """Cheap HTML -> text converter (keeps basic structure + links)."""
    if not html:
        return ""
    h = html
    # Drop scripts/styles
    h = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", h)
    # Convert common block breaks to newlines
    h = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", h)
    h = re.sub(r"(?i)</\s*(p|div|tr|li|h[1-6])\s*>", "\n", h)
    h = re.sub(r"(?i)<\s*(p|div|tr|li|h[1-6])\b[^>]*>", "", h)
    # Table cells -> separator
    h = re.sub(r"(?i)</\s*td\s*>", "\t", h)
    h = re.sub(r"(?i)</\s*th\s*>", "\t", h)
    # Preserve links: "text (url)"
    def _a_repl(m):
        href = (m.group(1) or "").strip()
        text = re.sub(r"<[^>]+>", " ", m.group(2) or "")
        text = _normalize_text_for_llm(text).strip()
        href = href.strip()
        if href and text and href not in text:
            return f"{text} ({href})"
        return text or href

    h = re.sub(r'(?is)<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', _a_repl, h)
    # Strip remaining tags
    h = re.sub(r"(?s)<[^>]+>", " ", h)
    # Decode common entities
    h = (
        h.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    # Normalize whitespace but keep some newlines
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in h.splitlines()]
    lines = [ln for ln in lines if ln]
    text = "\n".join(lines)
    # Collapse excessive newlines
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    return text


def _iter_attachments(msg: Message) -> List[Tuple[str, bytes]]:
    """Return (filename, payload_bytes) for each attachment."""
    attachments: List[Tuple[str, bytes]] = []
    for part in msg.walk():
        disposition = (part.get("Content-Disposition") or "").lower()
        if "attachment" not in disposition:
            continue
        filename = part.get_filename()
        if not filename:
            continue
        payload = part.get_payload(decode=True) or b""
        if payload:
            attachments.append((filename, payload))
    return attachments


def _extract_inline_images(msg: Message) -> List[Tuple[str, str, bytes]]:
    """Extract inline image parts referenced by Content-ID (cid:) in HTML.
    Returns list of (content_id, filename, payload_bytes).
    """
    result: List[Tuple[str, str, bytes]] = []
    for part in msg.walk():
        content_type = part.get_content_type() or ""
        if not content_type.startswith("image/"):
            continue
        disposition = (part.get("Content-Disposition") or "").lower()
        if "attachment" in disposition:
            continue
        content_id = part.get("Content-ID", "")
        if not content_id:
            continue
        cid = content_id.strip("<>").strip()
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        orig_filename = part.get_filename()
        ext = content_type.split("/")[-1].replace("jpeg", "jpg")
        if orig_filename:
            file_ext = os.path.splitext(orig_filename)[1].lstrip(".")
            if file_ext:
                ext = file_ext
        filename = f"{cid}.{ext}"
        result.append((cid, filename, payload))
    return result


def _create_kimi_llm() -> Optional[LLMProvider]:
    """Create an LLMProvider (with automatic failover), if configured.

    Priority:
      1) Kimi (Moonshot) if configured
      2) DeepSeek if configured
    """

    class _FailoverLLM:
        is_failover = True

        def __init__(self, providers: List[LLMProvider]):
            self.providers = providers

        def _should_failover(self, exc: Exception) -> bool:
            msg = str(exc).lower()
            return _is_tpd_limit_error(exc) or ("429" in msg) or ("rate_limit" in msg)

        def _call(self, fn_name: str, *args, **kwargs) -> str:
            last_exc: Optional[Exception] = None
            for idx, p in enumerate(self.providers):
                try:
                    fn = getattr(p, fn_name)
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    # Only fail over on quota/rate-limit style errors or unsupported feature.
                    msg = str(e).lower()
                    unsupported = "not supported" in msg or "unsupported" in msg
                    if idx < len(self.providers) - 1 and (self._should_failover(e) or unsupported):
                        continue
                    raise
            if last_exc:
                raise last_exc
            raise RuntimeError("No LLM providers configured")

        def generate_response(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
            return self._call("generate_response", prompt=prompt, system_message=system_message, **kwargs)

        def generate_response_with_images(
            self,
            prompt: str,
            images: List[Tuple[bytes, str]],
            system_message: Optional[str] = None,
            **kwargs,
        ) -> str:
            return self._call(
                "generate_response_with_images",
                prompt=prompt,
                images=images,
                system_message=system_message,
                **kwargs,
            )

    providers: List[LLMProvider] = []

    if Config.KIMI_API_KEY:
        providers.append(
            LLMProvider(
                provider="kimi",
                api_key=Config.KIMI_API_KEY,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=4096,
                enable_cache=Config.LLM_ENABLE_CACHE,
                base_url=Config.KIMI_BASE_URL,
                model=Config.KIMI_MODEL,
                kimi_disable_thinking=Config.KIMI_DISABLE_THINKING,
                request_timeout=Config.LLM_REQUEST_TIMEOUT,
            )
        )

    if Config.DEEPSEEK_API_KEY:
        providers.append(
            LLMProvider(
                provider="deepseek",
                api_key=Config.DEEPSEEK_API_KEY,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=4096,
                enable_cache=Config.LLM_ENABLE_CACHE,
                base_url=Config.DEEPSEEK_BASE_URL,
                model=Config.DEEPSEEK_MODEL,
                request_timeout=Config.LLM_REQUEST_TIMEOUT,
            )
        )

    if not providers:
        print("⚠️ No LLM API key set (KIMI_API_KEY/DEEPSEEK_API_KEY) – ingesting without LLM extraction.")
        return None

    return providers[0] if len(providers) == 1 else _FailoverLLM(providers)


@contextmanager
def _temp_attachment_file(filename: str, data: bytes):
    """Write attachment bytes to a temporary file and yield its path."""
    import tempfile

    suffix = os.path.splitext(filename)[1] or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# New helpers: link extraction, attachment splitting, Kimi processing
# ---------------------------------------------------------------------------

def _extract_links_from_message(msg: Message) -> List[str]:
    """Extract all URLs from the email (HTML hrefs and plain-text URLs)."""
    urls: set = set()
    url_pattern = re.compile(r"https?://[^\s<>\"')\]]+")

    for part in msg.walk():
        content_type = part.get_content_type()
        disposition = (part.get("Content-Disposition") or "").lower()
        if "attachment" in disposition:
            continue

        try:
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            text = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
        except Exception:
            continue

        if content_type == "text/html":
            for match in re.finditer(r'href=["\']([^"\']+)["\']', text, re.IGNORECASE):
                href = match.group(1)
                if href.startswith(("http://", "https://")):
                    urls.add(href)

        for match in url_pattern.finditer(text):
            urls.add(match.group())

    return sorted(urls)


def _email_id(subject: str, date_header: str) -> str:
    """Generate a stable, URL-safe ID for an email based on subject + date."""
    return hashlib.md5(f"{subject}:{date_header}".encode()).hexdigest()[:12]


def _extract_html_body(msg: Message) -> Optional[str]:
    """Extract the text/html body from an email message (decoded string)."""
    for part in msg.walk():
        content_type = part.get_content_type()
        disposition = (part.get("Content-Disposition") or "").lower()
        if content_type == "text/html" and "attachment" not in disposition:
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode(
                        part.get_content_charset() or "utf-8", errors="ignore"
                    )
            except Exception:
                continue
    return None


def _save_email_assets(
    eid: str,
    html_body: Optional[str],
    image_attachments: List[Tuple[str, bytes]],
    inline_images: Optional[List[Tuple[str, str, bytes]]] = None,
) -> Tuple[bool, List[str]]:
    """
    Save email HTML and image attachments to disk under EMAIL_ASSETS_DIR.
    Inline images (cid: referenced) are also saved and their references in
    the HTML are rewritten to servable URLs.
    Returns (has_html, list_of_saved_image_filenames).
    """
    assets_dir = Config.EMAIL_ASSETS_DIR
    email_dir = os.path.join(assets_dir, eid)
    os.makedirs(email_dir, exist_ok=True)

    saved_images: List[str] = []

    if image_attachments:
        images_dir = os.path.join(email_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for filename, payload in image_attachments:
            safe_name = re.sub(r'[<>:"/\\|?*]', "_", filename)
            img_path = os.path.join(images_dir, safe_name)
            with open(img_path, "wb") as f:
                f.write(payload)
            saved_images.append(safe_name)
            print(f"🖼️ Saved image → {img_path}")

    cid_map: Dict[str, str] = {}
    if inline_images:
        images_dir = os.path.join(email_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for cid, filename, payload in inline_images:
            safe_name = re.sub(r'[<>:"/\\|?*]', "_", filename)
            img_path = os.path.join(images_dir, safe_name)
            with open(img_path, "wb") as f:
                f.write(payload)
            saved_images.append(safe_name)
            cid_map[cid] = f"/email_assets/{eid}/images/{safe_name}"
            print(f"🖼️ Saved inline image (cid:{cid}) → {img_path}")

    has_html = False
    if html_body:
        if cid_map:
            for cid, url in cid_map.items():
                html_body = html_body.replace(f"cid:{cid}", url)
        html_path = os.path.join(email_dir, "original.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_body)
        has_html = True
        print(f"💾 Saved original HTML → {html_path}")

    return has_html, saved_images


def _split_attachments(
    msg: Message,
) -> Tuple[List[Tuple[str, bytes]], List[Tuple[str, bytes]]]:
    """Split email attachments into (image_list, non_image_list)."""
    images: List[Tuple[str, bytes]] = []
    others: List[Tuple[str, bytes]] = []

    for filename, payload in _iter_attachments(msg):
        ext = os.path.splitext(filename)[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            images.append((filename, payload))
        else:
            others.append((filename, payload))

    return images, others


def _extract_non_image_text(attachments: List[Tuple[str, bytes]]) -> str:
    """Extract text from non-image attachments using DocumentLoaderFactory."""
    if not attachments:
        return ""

    factory = DocumentLoaderFactory(
        min_text_length=Config.MIN_TEXT_LENGTH_FOR_OCR,
        ocr_language=Config.OCR_LANGUAGE,
        tesseract_path=Config.TESSERACT_PATH,
    )
    texts: List[str] = []

    for filename, payload in attachments:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in Config.SUPPORTED_EXTENSIONS or ext in _IMAGE_EXTENSIONS:
            print(f"ℹ️ Skipping unsupported attachment: {filename}")
            continue

        print(f"📎 Extracting text from attachment: {filename}")
        try:
            with _temp_attachment_file(filename, payload) as path:
                docs = factory.load(path)
                for d in docs:
                    content = d.get("content", "").strip()
                    if content:
                        texts.append(f"[Attachment: {filename}]\n{content}")
        except Exception as e:
            print(f"⚠️ Failed to process attachment {filename}: {e}")

    return "\n\n".join(texts)


def _detect_forced_email_types(*parts: str) -> List[str]:
    """Return deterministic categories for exact subject/body markers."""
    if any(_JOBS_EVENTS_PHRASE in (part or "") for part in parts):
        return list(_JOBS_EVENTS_TYPES)
    return []


def _resolve_email_types(result: Dict, forced_types: List[str]) -> List[str]:
    """Normalize model output to the categories supported by the app."""
    if forced_types:
        return list(forced_types)

    raw_types = result.get("types") or result.get("type") or []
    if isinstance(raw_types, str):
        raw_types = [raw_types]

    normalized: List[str] = []
    for value in raw_types:
        if value in _VALID_EMAIL_TYPES and value not in normalized:
            normalized.append(value)
    return normalized


_DATEISH_RE = re.compile(
    r"("
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r"|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?"
    r"|\d{1,2}\s*(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|"
    r"Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\.?\s*,?\s*\d{0,4}"
    r"|(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|"
    r"Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\.?\s+\d{1,2}(?:,\s*\d{4})?"
    r"|\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*[日號]?"
    r"|\d{1,2}\s*月\s*\d{1,2}\s*[日號]?"
    r")",
    re.IGNORECASE,
)

_APPLICATION_PERIOD_KEYWORDS = (
    "application period",
    "application deadline",
    "application due",
    "application:",
    "application：",
    "registration period",
    "registration deadline",
    "registration:",
    "registration：",
    "enrolment period",
    "enrolment deadline",
    "enrolment:",
    "enrolment：",
    "enrollment period",
    "enrollment deadline",
    "enrollment:",
    "enrollment：",
    "apply by",
    "submission deadline",
    "deadline",
    "報名期",
    "報名期間",
    "報名截止",
    "申請期",
    "申請期間",
    "申請截止",
    "截止日期",
    "截止",
)

_EVENT_PERIOD_KEYWORDS = (
    "event period",
    "event date",
    "event time",
    "date and time",
    "date:",
    "date：",
    "activity period",
    "活動日期",
    "活動時間",
    "舉行日期",
    "日期及時間",
    "日期：",
)


def _extract_json_string_field(text: str, keys: Tuple[str, ...]) -> str:
    """Extract a string field from a JSON-ish model response without parsing all JSON."""
    if not text:
        return ""

    for key in keys:
        match = re.search(
            rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
            text,
            re.DOTALL,
        )
        if not match:
            continue
        raw_value = match.group(1)
        try:
            value = json.loads(f'"{raw_value}"')
        except json.JSONDecodeError:
            value = raw_value
        return str(value).strip()
    return ""


def _clean_period_fragment(fragment: str) -> str:
    """Remove accidental JSON key wrappers from fallback period snippets."""
    if fragment is None or isinstance(fragment, (dict, list)):
        return ""

    cleaned = str(fragment)[:320].strip(" -:;：，。")
    if not cleaned or cleaned.startswith(("{", "[")):
        return ""

    match = re.match(
        r'^["\']?(?P<key>[A-Za-z_]+)["\']?\s*:\s*(?P<value>.+?)(?:,)?$',
        cleaned,
        re.DOTALL,
    )
    if not match:
        return cleaned

    key = match.group("key")
    if key in {"full_text", "details", "introduction", "links", "name"}:
        return ""

    value = match.group("value").strip().rstrip(",").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = value[1:-1]
    return str(value).strip(" -:;：，。")[:320]


def _extract_period_by_keywords(text: str, keywords: Tuple[str, ...]) -> str:
    """Best-effort fallback for dates Kimi leaves out of structured JSON."""
    if not text:
        return ""

    normalized = re.sub(r"[ \t]+", " ", text.replace("\r", "\n"))
    fragments: List[str] = []
    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue
        fragments.extend(
            part.strip()
            for part in re.split(r"(?<=[。；;])\s+| {2,}", line)
            if part.strip()
        )

    for index, fragment in enumerate(fragments):
        lower = fragment.lower()
        if not any(keyword.lower() in lower for keyword in keywords):
            continue
        candidates = [
            " ".join(fragments[index:index + width])
            for width in range(1, 4)
        ]
        for candidate in candidates:
            if _DATEISH_RE.search(candidate):
                cleaned = _clean_period_fragment(candidate)
                if cleaned:
                    return cleaned

    return ""


def _process_email_with_kimi(
    llm: Optional[LLMProvider],
    subject: str,
    body: str,
    attachment_text: str,
    html_body: Optional[str],
    image_attachments: List[Tuple[str, bytes]],
    discovered_links: List[str],
) -> Dict:
    """
    Use Kimi to classify the email, extract structured fields, and produce
    a single combined document.  Falls back to raw concatenation when Kimi
    is unavailable.
    """
    normalized_body_full = _normalize_text_for_llm(body)
    normalized_html_full = _normalize_text_for_llm(_html_to_text(html_body or ""))
    normalized_attachment_full = _normalize_text_for_llm(attachment_text)
    # If plain text is too short (common for HTML-only/table emails), fall back to HTML-derived text.
    effective_body = normalized_body_full
    if len(effective_body.strip()) < 200:
        effective_body = normalized_html_full or effective_body
    cleaned_body = _truncate_for_llm(effective_body, max_chars=12000)
    cleaned_attachment = _truncate_for_llm(normalized_attachment_full, max_chars=6000)

    parts: List[str] = [f"Subject: {subject}"]
    if cleaned_body.strip():
        parts.append(f"Body:\n{cleaned_body}")
    if normalized_html_full.strip() and len(normalized_body_full.strip()) < 200:
        # Preserve that this was HTML-derived for debugging; keep it short.
        parts.append("HTML (converted):\n" + _truncate_for_llm(normalized_html_full, max_chars=6000))
    if cleaned_attachment.strip():
        parts.append(f"Attachment text:\n{cleaned_attachment}")
    if discovered_links:
        parts.append("Links found in the email:\n" + "\n".join(discovered_links))
    prompt = "\n\n".join(parts)

    local_fields = _local_extract_event_fields(prompt, discovered_links)

    fallback_links = [{"url": u, "category": "other"} for u in discovered_links]
    forced_types = _detect_forced_email_types(subject, body, attachment_text)
    hinted_types = _cheap_type_hint(subject, cleaned_body, cleaned_attachment)
    should_call_llm = _should_call_llm(forced_types, hinted_types)

    image_sig = "|".join(
        f"{name}:{len(payload)}:{hashlib.sha256(payload[:2048]).hexdigest()}"
        for name, payload in (image_attachments or [])
    )

    fingerprint_src = "\n".join(
        [
            _KIMI_CACHE_VERSION,
            subject.strip(),
            effective_body.strip(),
            normalized_attachment_full.strip(),
            "\n".join(sorted(discovered_links or [])),
            ",".join(forced_types or []),
            image_sig,
        ]
    ).encode("utf-8", errors="ignore")
    fingerprint = hashlib.sha256(fingerprint_src).hexdigest()

    cached = _kimi_cache_get(fingerprint)
    if cached:
        cached_links = cached.get("links") or []
        seen = set()
        merged: List[Dict] = []
        for lk in cached_links:
            if isinstance(lk, dict) and "url" in lk:
                u = lk["url"]
                if u not in seen:
                    merged.append({"url": u, "category": lk.get("category", "other")})
                    seen.add(u)
            elif isinstance(lk, str):
                if lk not in seen:
                    merged.append({"url": lk, "category": "other"})
                    seen.add(lk)
        for u in discovered_links or []:
            if u not in seen:
                merged.append({"url": u, "category": "other"})
                seen.add(u)
        cached["links"] = merged
        if forced_types:
            cached["types"] = forced_types
            cached["type"] = forced_types[0]
        return cached
    fallback_application_period = _extract_period_by_keywords(
        prompt, _APPLICATION_PERIOD_KEYWORDS,
    )
    fallback_event_period = _extract_period_by_keywords(prompt, _EVENT_PERIOD_KEYWORDS)

    if llm is None:
        email_type = forced_types[0] if forced_types else ""
        return {
            "type": email_type,
            "types": forced_types,
            "name": subject,
            "introduction": "",
            "application_period": fallback_application_period,
            "event_period": "" if forced_types == _JOBS_EVENTS_TYPES else (local_fields.get("event_period") or fallback_event_period),
            "details": "",
            "fees": "",
            "event_time": "",
            "requirements": local_fields.get("requirements", ""),
            "application_deadline": local_fields.get("application_deadline", ""),
            "location": local_fields.get("location", ""),
            "application_link": local_fields.get("application_link", ""),
            "links": fallback_links,
            "full_text": prompt,
        }

    if not should_call_llm:
        picked_types = forced_types or hinted_types
        email_type = picked_types[0] if picked_types else ""
        intro = ""
        if cleaned_body.strip():
            intro = cleaned_body.split("\n", 1)[0].strip()[:240]
        result = {
            "type": email_type,
            "types": picked_types,
            "name": subject,
            "introduction": intro,
            "application_period": fallback_application_period,
            "event_period": "" if picked_types == _JOBS_EVENTS_TYPES else (local_fields.get("event_period") or fallback_event_period),
            "details": "",
            "fees": "",
            "event_time": "",
            "requirements": local_fields.get("requirements", ""),
            "application_deadline": local_fields.get("application_deadline", ""),
            "location": local_fields.get("location", ""),
            "application_link": local_fields.get("application_link", ""),
            "links": fallback_links,
            "full_text": prompt,
        }
        _kimi_cache_put(fingerprint, result)
        return result

    try:
        image_data: List[Tuple[bytes, str]] = []
        for filename, payload in image_attachments:
            ext = os.path.splitext(filename)[1].lower()
            mime = _IMAGE_MIME_MAP.get(ext, "image/png")
            image_data.append((payload, mime))
            print(f"🖼️ Attaching image for Kimi vision: {filename}")

        def _call_llm_once():
            if image_data:
                return llm.generate_response_with_images(
                    prompt=prompt,
                    images=image_data,
                    system_message=_CLASSIFICATION_SYSTEM_MESSAGE,
                )
            return llm.generate_response(
                prompt=prompt,
                system_message=_CLASSIFICATION_SYSTEM_MESSAGE,
            )

        raw = None
        last_exc: Optional[Exception] = None
        for attempt in range(6):
            try:
                raw = _call_llm_once()
                break
            except Exception as e:
                last_exc = e
                if _is_tpd_limit_error(e) and not getattr(llm, "is_failover", False):
                    raise SystemExit(f"TPD exceeded; stop and rerun after reset.\n{e}")
                msg = str(e).lower()
                if "429" in msg or "rate_limit" in msg:
                    sleep_s = min(60.0, (2.0 ** attempt) + random.random())
                    time.sleep(sleep_s)
                    continue
                raise
        if raw is None and last_exc is not None:
            raise last_exc

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        result = json.loads(cleaned)

        raw_links = result.get("links") or []
        categorized: List[Dict] = []
        if isinstance(raw_links, str):
            raw_links = [l.strip() for l in raw_links.split("\n") if l.strip()]
        for lk in raw_links:
            if isinstance(lk, dict) and "url" in lk:
                categorized.append({
                    "url": lk["url"],
                    "category": lk.get("category", "other"),
                })
            elif isinstance(lk, str):
                categorized.append({"url": lk, "category": "other"})

        seen_urls = {lk["url"] for lk in categorized}
        for url in discovered_links:
            if url not in seen_urls:
                categorized.append({"url": url, "category": "other"})

        # Prompt uses application_period, event_period, event_time; support legacy period/time
        full_text = result.get("full_text", prompt)
        application_period = (
            _clean_period_fragment(result.get("application_period", ""))
            or _extract_period_by_keywords(full_text, _APPLICATION_PERIOD_KEYWORDS)
            or fallback_application_period
            or _clean_period_fragment(result.get("period", ""))
        )
        event_period = _clean_period_fragment(result.get("event_period", ""))
        if not event_period and forced_types != _JOBS_EVENTS_TYPES:
            event_period = (
                _extract_period_by_keywords(full_text, _EVENT_PERIOD_KEYWORDS)
                or fallback_event_period
                or _clean_period_fragment(result.get("period", ""))
            )
        event_time = result.get("event_time", "") or result.get("time", "")
        email_types = _resolve_email_types(result, forced_types)
        local_from_links = _local_extract_event_fields(full_text, discovered_links, categorized_links=categorized)
        requirements = _strip_jsonish_and_urls(result.get("requirements", "")) or local_from_links.get("requirements") or local_fields.get("requirements", "")
        application_link = local_from_links.get("application_link") or local_fields.get("application_link", "")
        final = {
            "type": email_types[0] if email_types else result.get("type", ""),
            "types": email_types,
            "name": result.get("name", ""),
            "introduction": result.get("introduction", ""),
            "application_period": application_period,
            "event_period": event_period,
            "details": result.get("details", ""),
            "fees": result.get("fees", ""),
            "event_time": event_time,
            "requirements": requirements,
            "application_deadline": local_from_links.get("application_deadline") or local_fields.get("application_deadline", ""),
            "location": local_from_links.get("location") or local_fields.get("location", ""),
            "application_link": application_link,
            "links": categorized,
            "full_text": full_text,
        }
        _kimi_cache_put(fingerprint, final)
        return final

    except json.JSONDecodeError:
        print("⚠️ Kimi did not return valid JSON; using raw response as full_text.")
        email_type = forced_types[0] if forced_types else ""
        raw_full_text = _extract_json_string_field(raw, ("full_text",)) or raw
        final = {
            "type": email_type,
            "types": forced_types,
            "name": subject,
            "introduction": "",
            "application_period": (
                _clean_period_fragment(
                    _extract_json_string_field(raw, ("application_period", "period"))
                )
                or _extract_period_by_keywords(raw_full_text, _APPLICATION_PERIOD_KEYWORDS)
                or fallback_application_period
            ),
            "event_period": (
                _clean_period_fragment(_extract_json_string_field(raw, ("event_period",)))
                or (
                    ""
                    if forced_types == _JOBS_EVENTS_TYPES
                    else (
                        _extract_period_by_keywords(raw_full_text, _EVENT_PERIOD_KEYWORDS)
                        or fallback_event_period
                    )
                )
            ),
            "details": "",
            "fees": "",
            "event_time": "",
            "requirements": local_fields.get("requirements", ""),
            "application_deadline": local_fields.get("application_deadline", ""),
            "location": local_fields.get("location", ""),
            "application_link": local_fields.get("application_link", ""),
            "links": fallback_links,
            "full_text": raw,
        }
        _kimi_cache_put(fingerprint, final)
        return final
    except Exception as e:
        if _is_tpd_limit_error(e):
            raise SystemExit(f"TPD exceeded; stop and rerun after reset.\n{e}")
        print(f"⚠️ Kimi processing failed: {e}")
        email_type = forced_types[0] if forced_types else ""
        final = {
            "type": email_type,
            "types": forced_types,
            "name": subject,
            "introduction": "",
            "application_period": fallback_application_period,
            "event_period": "" if forced_types == _JOBS_EVENTS_TYPES else (local_fields.get("event_period") or fallback_event_period),
            "details": "",
            "fees": "",
            "event_time": "",
            "requirements": local_fields.get("requirements", ""),
            "application_deadline": local_fields.get("application_deadline", ""),
            "location": local_fields.get("location", ""),
            "application_link": local_fields.get("application_link", ""),
            "links": fallback_links,
            "full_text": prompt,
        }
        _kimi_cache_put(fingerprint, final)
        return final


def _build_document_content(info: Dict) -> str:
    """Build the final searchable document string from structured info."""
    type_labels = {
        "scholarship": "scholarship (獎學金)",
        "events": "events (活動)",
        "Member Recruitment": "Member Recruitment (成員招募)",
        "Job Recruitment": "Job Recruitment (職位招募)",
        "workshop": "workshop (工作坊)",
        "recruitment": "recruitment (招募)",
    }
    lines: List[str] = []

    email_types = info.get("types") or [info.get("type", "")]
    email_types = [email_type for email_type in email_types if email_type]
    if len(email_types) > 1:
        labels = [type_labels.get(email_type, email_type) for email_type in email_types]
        lines.append(f"Email Types: {', '.join(labels)}")
    elif email_types:
        email_type = email_types[0]
        lines.append(f"Email Type: {type_labels.get(email_type, email_type)}")
    if info.get("name"):
        lines.append(f"Name: {info['name']}")
    if info.get("introduction"):
        lines.append(f"Introduction: {info['introduction']}")
    for label, key in [
        ("Application Period", "application_period"),
        ("Event Period", "event_period"),
        ("Event Time", "event_time"),
    ]:
        val = info.get(key)
        if val:
            lines.append(f"{label}: {val}")
    if info.get("details"):
        lines.append(f"Details: {info['details']}")
    if info.get("fees"):
        lines.append(f"Fees: {info['fees']}")
    if info.get("requirements"):
        lines.append(f"Requirements: {info['requirements']}")
    if info.get("links"):
        link_urls = [
            lk["url"] if isinstance(lk, dict) else lk for lk in info["links"]
        ]
        lines.append("Links:")
        for u in link_urls:
            lines.append(f"  {u}")

    header = "\n".join(lines)
    full_text = info.get("full_text", "")

    if header and full_text:
        return f"{header}\n\n---\n\n{full_text}"
    return header or full_text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_and_ingest_emails():
    """Main entry point: fetch emails, process, and ingest into ChromaDB."""
    last_run_path = Config.EMAIL_LAST_RUN_FILE
    last_run = _get_last_run(last_run_path)
    now = datetime.now(timezone.utc)

    print(f"📬 Fetching emails since {last_run.isoformat()} (UTC)")

    llm = _create_kimi_llm()

    db = ChromaDBManager(
        persist_directory=Config.CHROMA_DB_DIR,
        collection_name=Config.CHROMA_COLLECTION_NAME,
    )

    try:
        client = _connect_imap()
    except Exception as e:
        raise RuntimeError(f"Failed to connect to IMAP: {e}") from e

    try:
        status, _ = client.select("INBOX")
        if status != "OK":
            raise RuntimeError("Failed to select INBOX")

        criteria = _build_search_criteria(last_run)
        status, data = client.search(None, *criteria)
        if status != "OK":
            raise RuntimeError(f"IMAP search failed with status: {status}")

        ids = data[0].split()
        print(f"📨 Found {len(ids)} email(s) to process")

        all_docs: List[Dict] = []

        for msg_id in ids:
            status, msg_data = client.fetch(msg_id, "(RFC822)")
            if status != "OK" or not msg_data:
                print(f"⚠️ Failed to fetch message {msg_id}")
                continue

            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            subject = _decode_mime_header(msg.get("Subject", ""))
            date_header = msg.get("Date", "")
            print(f"\n📧 Processing: {subject}")

            eid = _email_id(subject, date_header)

            body_text = _get_body_from_message(msg)
            html_body = _extract_html_body(msg)

            discovered_links = _extract_links_from_message(msg)

            image_attachments, non_image_attachments = _split_attachments(msg)
            inline_images = _extract_inline_images(msg)

            has_html, saved_images = _save_email_assets(
                eid, html_body, image_attachments, inline_images,
            )

            attachment_text = _extract_non_image_text(non_image_attachments)

            info = _process_email_with_kimi(
                llm, subject, body_text, attachment_text,
                html_body, image_attachments, discovered_links,
            )

            category_types = info.get("types") or []
            if not category_types and info.get("type"):
                category_types = [info["type"]]
            category_types = category_types or [""]
            email_types_json = json.dumps(category_types, ensure_ascii=False)

            for category_type in category_types:
                doc_info = {**info, "type": category_type}
                doc_content = _build_document_content(doc_info)
                if not doc_content.strip():
                    continue

                cat_links = info.get("links", [])
                if cat_links and isinstance(cat_links[0], str):
                    cat_links = [{"url": u, "category": "other"} for u in cat_links]
                flat_links_str = "\n".join(
                    lk["url"] if isinstance(lk, dict) else lk for lk in info.get("links", [])
                )
                application_period = _clean_period_fragment(info.get("application_period", ""))
                event_period = _clean_period_fragment(info.get("event_period", ""))
                event_time = info.get("event_time", "")
                period = application_period or event_period
                all_docs.append({
                    "content": doc_content,
                    "metadata": {
                        "source": f"email:{eid}",
                        "type": "email",
                        "section": "email",
                        "email_type": category_type,
                        "email_types": email_types_json,
                        "email_name": info.get("name", ""),
                        "email_introduction": info.get("introduction", ""),
                        "email_application_period": application_period,
                        "email_event_period": event_period,
                        "email_period": period,
                        "email_details": info.get("details", ""),
                        "email_fees": info.get("fees", ""),
                        "email_event_time": event_time,
                        "email_time": event_time,
                        "email_requirements": info.get("requirements", ""),
                        "email_application_deadline": info.get("application_deadline", ""),
                        "email_location": info.get("location", ""),
                        "email_application_link": info.get("application_link", ""),
                        "email_links": flat_links_str,
                        "email_subject": subject,
                        "email_date": date_header,
                        "email_id": eid,
                        "email_categorized_links": json.dumps(cat_links, ensure_ascii=False),
                        "email_images": json.dumps(saved_images, ensure_ascii=False),
                        "email_has_html": "true" if has_html else "false",
                        "ingested_at": now.isoformat(),
                    },
                })

        if not all_docs:
            print("ℹ️ No new email content to ingest.")
        else:
            print(f"\n🚀 Ingesting {len(all_docs)} document(s) into ChromaDB...")
            db._add_loaded_documents(all_docs, extra_metadata=None)

        _set_last_run(last_run_path, now)
        print(f"✅ Email fetch and ingestion complete. Last run set to {now.isoformat()}")
    finally:
        try:
            client.logout()
        except Exception:
            pass


if __name__ == "__main__":
    fetch_and_ingest_emails()
