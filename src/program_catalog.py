"""Static SFU programme name/code reference used as a retrieval aid."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple


PROGRAMME_CODE_REFERENCE_ID = "programme_code_reference"

PROGRAMMES: List[Dict[str, str]] = [
    {"level": "Postgraduate Programmes", "name": "Master in Nursing and Allied Health (Part-time)", "code": "MNAH"},
    {"level": "Postgraduate Programmes", "name": "Postgraduate Certificate in Nursing and Allied Health (Part-time)", "code": "Intermediate Exit Award of MNAH"},
    {"level": "Postgraduate Programmes", "name": "Postgraduate Diploma in Nursing and Allied Health (Part-time)", "code": "Intermediate Exit Award of MNAH"},
    {"level": "Postgraduate Programmes", "name": "Master of Corporate Governance (Part-time)", "code": "MCG"},
    {"level": "Postgraduate Programmes", "name": "Postgraduate Diploma in Private Banking and Family Office (Part-time)", "code": "PGDPBFO"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Arts (Honours) in Language and Culture", "code": "BALC"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Arts (Honours) in Language and Liberal Studies", "code": "BALLS"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Arts (Honours) in Translation Technology", "code": "BATT"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Business Administration (Honours)", "code": "BBA"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Business Administration (Honours) in Applied Hotel and Tourism Management", "code": "BBA-AHTM"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Crime and Security Science (Honours)", "code": "BCSS"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Education (Honours) in Early Childhood Education", "code": "BEDECE"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Entrepreneurial Management (Honours) in Design Business", "code": "BEM"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Health Sciences (Honours)", "code": "BHS"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Nursing (Honours)", "code": "BN"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Science (Honours) in Artificial Intelligence", "code": "BSAI"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Science (Honours) in Digital Entertainment Technology", "code": "BSDET"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Science (Honours) in Physiotherapy", "code": "BSPT"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Social Sciences (Honours)", "code": "BSS"},
    {"level": "Undergraduate Programmes", "name": "Bachelor of Social Work (Honours)", "code": "BSW"},
    {"level": "Sub-degree Programmes", "name": "Associate Degree in Business", "code": "ADB"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Artificial Intelligence and Information and Communication Technology", "code": "HDAI"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Crime and Security Science", "code": "HDCSS"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Design", "code": "HDDE"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Early Childhood Education", "code": "HDECE"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Enrolled Nursing (General)", "code": "HDEN"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Enrolled Nursing (General) (Social Welfare Department scheme)", "code": "HDEN-SWD"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Film and Media Production", "code": "HDFMP"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Health Care", "code": "HDHC"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Hospitality Management", "code": "HDHM"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Human Services", "code": "HDHS"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Music Studies", "code": "HDMS"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Pharmaceutical Dispensing", "code": "HDPD"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Social Work", "code": "HDSW"},
    {"level": "Sub-degree Programmes", "name": "Higher Diploma in Translation Technology and Modern Languages", "code": "HDTTML"},
    {"level": "Professional Diploma / Diploma Programmes", "name": "Professional Diploma in Property Management", "code": "PDPM"},
    {"level": "Professional Diploma / Diploma Programmes", "name": "Diploma in Foundation Studies of Higher Education", "code": "DFS"},
    {"level": "Professional Diploma / Diploma Programmes", "name": "Diploma in Health Sciences", "code": "DHS"},
]


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _name_variants(name: str) -> List[str]:
    variants = {name.lower()}
    stripped_parentheticals = re.sub(r"\s*\([^)]*\)", "", name).strip().lower()
    variants.add(stripped_parentheticals)
    variants.add(stripped_parentheticals.replace("honours", "").strip())
    return [variant for variant in variants if variant]


def _code_match_span(code: str, query: str) -> Optional[Tuple[int, int]]:
    """Return the matched query span for a standalone programme code."""
    if code.lower().startswith("intermediate exit award"):
        if "mnah" in query.lower() and "exit" in query.lower():
            match = re.search(r"(?<![a-z0-9])mnah(?![a-z0-9-])", query.lower())
            if match:
                return match.span()
        return None

    parts = [re.escape(part) for part in re.findall(r"[a-z0-9]+", code.lower())]
    if not parts:
        return None

    separator = r"[-\s]?"
    code_pattern = separator.join(parts)
    # Do not let BATT match battery/battle, or BBA match BBA-AHTM.
    pattern = rf"(?<![a-z0-9]){code_pattern}(?![a-z0-9-])"
    match = re.search(pattern, query.lower())
    return match.span() if match else None


def _code_matches_query(code: str, query: str) -> bool:
    """Match programme codes as standalone codes, not substrings of words."""
    return _code_match_span(code, query) is not None


def _spans_overlap(first: Tuple[int, int], second: Tuple[int, int]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def programme_aliases() -> Dict[str, List[str]]:
    """Return code/name aliases for query expansion."""
    aliases: Dict[str, List[str]] = {}
    for programme in PROGRAMMES:
        code = programme["code"]
        name = programme["name"]
        aliases[name.lower()] = [code]
        if code.lower().startswith("intermediate exit award"):
            continue
        aliases[code.lower()] = [name]
        aliases[_normalise(code)] = [name]
    return aliases


def iter_programme_search_terms() -> Iterable[str]:
    for programme in PROGRAMMES:
        yield programme["name"]
        yield programme["code"]


def find_programmes_in_query(query: str) -> List[Dict[str, str]]:
    """Return catalogue entries explicitly mentioned by code or name in a query."""
    query_lower = query.lower()
    normalised_query = _normalise(query)
    code_matches: List[Tuple[Dict[str, str], Tuple[int, int]]] = []
    name_matches: List[Dict[str, str]] = []

    for programme in PROGRAMMES:
        code = programme["code"]
        name = programme["name"]
        code_span = _code_match_span(code, query)

        if code_span is not None:
            code_matches.append((programme, code_span))
            continue

        if any(
            name_variant in query_lower or _normalise(name_variant) in normalised_query
            for name_variant in _name_variants(name)
        ):
            name_matches.append(programme)

    matches: List[Dict[str, str]] = []
    used_spans: List[Tuple[int, int]] = []
    for programme, span in sorted(code_matches, key=lambda item: item[1][1] - item[1][0], reverse=True):
        if any(_spans_overlap(span, used) for used in used_spans):
            continue
        matches.append(programme)
        used_spans.append(span)

    matched_codes = {programme["code"] for programme in matches}
    for programme in name_matches:
        if programme["code"] not in matched_codes:
            matches.append(programme)
            matched_codes.add(programme["code"])

    return matches


def format_programme_code_reference() -> str:
    """Format the whole catalogue as one compact source document."""
    lines = [
        "SFU programme code reference.",
        "Use these mappings when a user asks for programme codes, programme names, or programme abbreviations.",
        "",
    ]
    current_level: Optional[str] = None
    for programme in PROGRAMMES:
        level = programme["level"]
        if level != current_level:
            current_level = level
            lines.append(f"{level}:")
        lines.append(f"- {programme['name']}: {programme['code']}")
    return "\n".join(lines)


def is_programme_code_query(query: str) -> bool:
    """Detect whether the static programme code reference should be injected."""
    query_lower = query.lower()
    normalised_query = _normalise(query)

    code_intent = (
        ("programme" in query_lower or "program" in query_lower or "course" in query_lower)
        and ("code" in query_lower or "abbr" in query_lower or "acronym" in query_lower)
    )
    if code_intent:
        return True

    for programme in PROGRAMMES:
        code = programme["code"]
        name = programme["name"]
        for name_variant in _name_variants(name):
            if name_variant in query_lower or _normalise(name_variant) in normalised_query:
                return True

        if _code_matches_query(code, query):
            return True

    return False


def build_programme_code_reference_doc(query: str) -> Optional[Dict]:
    if not is_programme_code_query(query):
        return None

    return {
        "id": PROGRAMME_CODE_REFERENCE_ID,
        "document": format_programme_code_reference(),
        "metadata": {
            "section": "Programme Code Reference",
            "source": "SFU programme code registry",
            "structured": True,
            "type": "programme_code_reference",
        },
        "similarity": 1.0,
        "retrieval_score": 1.0,
        "rank": 0,
    }


def get_programme_code_reference_source(source_id: str) -> Optional[Dict]:
    if source_id not in {PROGRAMME_CODE_REFERENCE_ID, f"doc_{PROGRAMME_CODE_REFERENCE_ID}"}:
        return None

    metadata = {
        "section": "Programme Code Reference",
        "source": "SFU programme code registry",
        "structured": True,
        "type": "programme_code_reference",
    }
    return {
        "source_id": source_id,
        "section": metadata["section"],
        "source_file": metadata["source"],
        "content": format_programme_code_reference(),
        "metadata": metadata,
        "total_chunks": 1,
    }
