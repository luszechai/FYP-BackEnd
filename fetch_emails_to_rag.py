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
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.header import decode_header as _decode_header_parts
from email.message import Message
from typing import Dict, List, Optional, Tuple

from config import Config
from src.llm_provider import LLMProvider
from src.vector_db import ChromaDBManager
from src.document_loader import DocumentLoaderFactory

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

_CLASSIFICATION_SYSTEM_MESSAGE = (
    "You are processing university emails for a student information chatbot.\n\n"
    "Given an email (subject, body, attachment text, and optionally images), you must:\n\n"
    "1. CLASSIFY the email as exactly one of these types:\n"
    '   - "scholarship" (獎學金) - scholarship information, financial aid\n'
    '   - "events" (活動) - campus events, activities, workshops, seminars\n'
    '   - "Member Recruitment" (成員招募) - member recruitment, club recruitment\n'
    '   - "Job Recruitment" (職位招募) - job postings, volunteer recruitment\n'
    '   - "workshop" (工作坊) - workshop, seminar, training\n\n'
    '2. EXTRACT structured information (leave empty string "" if not found):\n'
    "   - name: The name/title of the scholarship, event, or recruitment\n"
    '   - introduction: Introduction of the scholarship, event, or recruitment(a short summary of the email, or extraction of the main content of the email)\n'
    '   - application_period: Application period (e.g. "2024-01-01 to 2024-03-31")\n'
    '   - event_period: Event period (e.g. "2024-01-01 to 2024-03-31")\n'
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
                    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
                    text = re.sub(r"<[^>]+>", " ", text)
                    return re.sub(r"\s+", " ", text).strip()
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
    """Create an LLMProvider instance for Kimi, if configured."""
    if not Config.KIMI_API_KEY:
        print("⚠️ KIMI_API_KEY not set – email bodies will be ingested without LLM processing.")
        return None

    return LLMProvider(
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


def _process_email_with_kimi(
    llm: Optional[LLMProvider],
    subject: str,
    body: str,
    attachment_text: str,
    image_attachments: List[Tuple[str, bytes]],
    discovered_links: List[str],
) -> Dict:
    """
    Use Kimi to classify the email, extract structured fields, and produce
    a single combined document.  Falls back to raw concatenation when Kimi
    is unavailable.
    """
    parts: List[str] = [f"Subject: {subject}"]
    if body.strip():
        parts.append(f"Body:\n{body}")
    if attachment_text.strip():
        parts.append(f"Attachment text:\n{attachment_text}")
    if discovered_links:
        parts.append("Links found in the email:\n" + "\n".join(discovered_links))
    prompt = "\n\n".join(parts)

    fallback_links = [{"url": u, "category": "other"} for u in discovered_links]

    if llm is None:
        return {
            "type": "",
            "name": subject,
            "introduction": "",
            "application_period": "",
            "event_period": "",
            "details": "",
            "fees": "",
            "event_time": "",
            "requirements": "",
            "links": fallback_links,
            "full_text": prompt,
        }

    try:
        image_data: List[Tuple[bytes, str]] = []
        for filename, payload in image_attachments:
            ext = os.path.splitext(filename)[1].lower()
            mime = _IMAGE_MIME_MAP.get(ext, "image/png")
            image_data.append((payload, mime))
            print(f"🖼️ Attaching image for Kimi vision: {filename}")

        if image_data:
            raw = llm.generate_response_with_images(
                prompt=prompt,
                images=image_data,
                system_message=_CLASSIFICATION_SYSTEM_MESSAGE,
            )
        else:
            raw = llm.generate_response(
                prompt=prompt,
                system_message=_CLASSIFICATION_SYSTEM_MESSAGE,
            )

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
        application_period = result.get("application_period", "") or result.get("period", "")
        event_period = result.get("event_period", "") or result.get("period", "")
        event_time = result.get("event_time", "") or result.get("time", "")
        return {
            "type": result.get("type", ""),
            "name": result.get("name", ""),
            "introduction": result.get("introduction", ""),
            "application_period": application_period,
            "event_period": event_period,
            "details": result.get("details", ""),
            "fees": result.get("fees", ""),
            "event_time": event_time,
            "requirements": result.get("requirements", ""),
            "links": categorized,
            "full_text": result.get("full_text", prompt),
        }

    except json.JSONDecodeError:
        print("⚠️ Kimi did not return valid JSON; using raw response as full_text.")
        return {
            "type": "",
            "name": subject,
            "introduction": "",
            "application_period": "",
            "event_period": "",
            "details": "",
            "fees": "",
            "event_time": "",
            "requirements": "",
            "links": fallback_links,
            "full_text": raw,
        }
    except Exception as e:
        print(f"⚠️ Kimi processing failed: {e}")
        return {
            "type": "",
            "name": subject,
            "introduction": "",
            "application_period": "",
            "event_period": "",
            "details": "",
            "fees": "",
            "event_time": "",
            "requirements": "",
            "links": fallback_links,
            "full_text": prompt,
        }


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

    email_type = info.get("type", "")
    if email_type:
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
                image_attachments, discovered_links,
            )

            doc_content = _build_document_content(info)

            if doc_content.strip():
                cat_links = info.get("links", [])
                if cat_links and isinstance(cat_links[0], str):
                    cat_links = [{"url": u, "category": "other"} for u in cat_links]
                flat_links_str = "\n".join(
                    lk["url"] if isinstance(lk, dict) else lk for lk in info.get("links", [])
                )
                application_period = info.get("application_period", "")
                event_period = info.get("event_period", "")
                event_time = info.get("event_time", "")
                period = application_period or event_period
                all_docs.append({
                    "content": doc_content,
                    "metadata": {
                        "source": f"email:{subject}",
                        "type": "email",
                        "section": "email",
                        "email_type": info.get("type", ""),
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
