"""Text cleaning utilities for removing boilerplate content from extracted documents."""
import re
from collections import Counter
from typing import List, Optional


def remove_boilerplate(pages: List[str], min_pages_for_detection: int = 3,
                       repeat_threshold: float = 0.5) -> List[str]:
    """Remove repeated headers, footers, and navigation lines across pages.

    Analyses the first and last few lines of each page.  Lines that appear
    in more than *repeat_threshold* fraction of pages are treated as
    boilerplate and stripped.

    Args:
        pages: List of page texts (one string per page).
        min_pages_for_detection: Minimum pages required to attempt detection.
        repeat_threshold: Fraction of pages a line must appear in to be
            considered boilerplate (0.0-1.0).

    Returns:
        Cleaned page texts with boilerplate lines removed.
    """
    if len(pages) < min_pages_for_detection:
        return pages

    n_pages = len(pages)
    header_lines: Counter = Counter()
    footer_lines: Counter = Counter()

    # Collect first / last N lines from every page
    n_sample_lines = 4
    for text in pages:
        lines = text.splitlines()
        for line in lines[:n_sample_lines]:
            normalized = _normalize(line)
            if normalized:
                header_lines[normalized] += 1
        for line in lines[-n_sample_lines:]:
            normalized = _normalize(line)
            if normalized:
                footer_lines[normalized] += 1

    threshold_count = max(2, int(n_pages * repeat_threshold))
    boilerplate = set()
    for line, count in header_lines.items():
        if count >= threshold_count:
            boilerplate.add(line)
    for line, count in footer_lines.items():
        if count >= threshold_count:
            boilerplate.add(line)

    if not boilerplate:
        return pages

    cleaned = []
    for text in pages:
        lines = text.splitlines()
        kept = [l for l in lines if _normalize(l) not in boilerplate]
        cleaned.append("\n".join(kept).strip())

    removed = len(boilerplate)
    if removed:
        print(f"  🧹 Boilerplate removal: stripped {removed} repeated line pattern(s)")

    return cleaned


def clean_text(text: str) -> str:
    """General-purpose text cleaning applied after extraction.

    - Collapses 3+ consecutive blank lines into 2
    - Strips trailing whitespace per line
    - Removes common web navigation fragments
    """
    # Remove common web navigation / cookie-banner fragments
    nav_patterns = [
        r"(?i)^(skip to (main )?content|cookie policy|accept all cookies|privacy policy|terms of use)$",
        r"(?i)^(home\s*[|>»/]\s*)+.*$",
        r"(?i)^\s*(menu|navigation|breadcrumb)\s*$",
    ]
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.rstrip()
        if any(re.match(p, stripped) for p in nav_patterns):
            continue
        cleaned_lines.append(stripped)

    result = "\n".join(cleaned_lines)
    # Collapse excessive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _normalize(line: str) -> str:
    """Lowercase, strip whitespace and page numbers for comparison."""
    s = line.strip().lower()
    # Remove standalone page numbers like "Page 3", "- 12 -", "3"
    s = re.sub(r"^(page\s*)?\d+(\s*of\s*\d+)?$", "", s)
    s = re.sub(r"^-\s*\d+\s*-$", "", s)
    return s.strip()
