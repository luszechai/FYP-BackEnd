"""Static SFU scholarship and financial aid reference used as a retrieval aid."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


SCHOLARSHIP_REFERENCE_ID = "scholarship_reference"

SCHOLARSHIPS: List[Dict[str, str]] = [
    {
        "category": "Specialized & Regional Scholarships",
        "name": "Hong Kong Future Talents Scholarship Scheme for Advanced Studies",
        "identifier": "FTSS",
        "note": "For local Master's students in priority areas.",
    },
    {
        "category": "Specialized & Regional Scholarships",
        "name": "Saint Francis Scholarship for Southeast Asian Catholic Youth",
        "identifier": "SEACY",
        "note": "Full tuition coverage for Catholic students from Southeast Asian countries.",
    },
    {
        "category": "Admission & JUPAS Scholarships",
        "name": "Academic Achievement Scholarship",
        "identifier": "",
        "note": "JUPAS Programme Scholarship tied to entry or academic performance.",
    },
    {
        "category": "Admission & JUPAS Scholarships",
        "name": "Entrance Scholarship",
        "identifier": "",
        "note": "JUPAS Programme Scholarship tied to entry performance.",
    },
    {
        "category": "Admission & JUPAS Scholarships",
        "name": "Outstanding Performance Scholarship",
        "identifier": "OPS",
        "note": "JUPAS Programme Scholarship.",
    },
    {
        "category": "Admission & JUPAS Scholarships",
        "name": "Best Progress Awards",
        "identifier": "BPA",
        "note": "JUPAS Programme Scholarship.",
    },
    {
        "category": "Admission & JUPAS Scholarships",
        "name": "Best Clinical Skills Awards",
        "identifier": "BCSA",
        "note": "JUPAS Programme Scholarship.",
    },
    {
        "category": "Admission & JUPAS Scholarships",
        "name": "Dean's List",
        "identifier": "",
        "note": "JUPAS Programme academic award.",
    },
    {
        "category": "Admission & JUPAS Scholarships",
        "name": "Best Clinical Practicum Award",
        "identifier": "BCPA",
        "note": "JUPAS Programme Scholarship.",
    },
    {
        "category": "Admission & JUPAS Scholarships",
        "name": "Voluntary Services Awards",
        "identifier": "VSA",
        "note": "JUPAS Programme Scholarship.",
    },
    {
        "category": "Admission & JUPAS Scholarships",
        "name": "President's Scholarship",
        "identifier": "",
        "note": "For exceptionally high-achieving applicants.",
    },
    {
        "category": "Admission & JUPAS Scholarships",
        "name": "Vice-President's Scholarship",
        "identifier": "",
        "note": "For high-achieving applicants.",
    },
    {
        "category": "Internal Articulation Scholarships",
        "name": "Articulation to Postgraduate",
        "identifier": "",
        "note": "For graduates entering Master's or Postgraduate Diploma programmes.",
    },
    {
        "category": "Internal Articulation Scholarships",
        "name": "Articulation to Undergraduate",
        "identifier": "",
        "note": "For graduates with sub-degrees entering Year 3 of Bachelor's programmes.",
    },
    {
        "category": "Internal Articulation Scholarships",
        "name": "Articulation to Sub-degree",
        "identifier": "",
        "note": "For graduates with diplomas entering Associate Degree or Higher Diploma programmes.",
    },
    {
        "category": "Programme-Specific Scholarships",
        "name": "Nursing Programme Scholarships",
        "identifier": "",
        "note": "Specific awards for BN or HDEN students, such as clinical excellence awards.",
    },
    {
        "category": "Programme-Specific Scholarships",
        "name": "Artificial Intelligence / DET Scholarships",
        "identifier": "",
        "note": "For students in tech-heavy disciplines such as AI or Digital Entertainment Technology.",
    },
    {
        "category": "Programme-Specific Scholarships",
        "name": "Social Work / Social Science Awards",
        "identifier": "",
        "note": "Often focused on community service and placement performance.",
    },
    {
        "category": "Government & External Financial Assistance",
        "name": "Tertiary Student Finance Scheme - Publicly-funded Programmes",
        "identifier": "TSFS",
        "note": "Means-tested government financial assistance.",
    },
    {
        "category": "Government & External Financial Assistance",
        "name": "Financial Assistance Scheme for Post-secondary Students",
        "identifier": "FASP",
        "note": "Means-tested assistance for self-financing programmes.",
    },
    {
        "category": "Government & External Financial Assistance",
        "name": "Non-means-tested Loan Scheme for Full-time Tertiary Students",
        "identifier": "NLSFT",
        "note": "Non-means-tested loan scheme.",
    },
    {
        "category": "Government & External Financial Assistance",
        "name": "Extended Non-means-tested Loan Scheme",
        "identifier": "ENLS",
        "note": "For part-time or specific professional programmes.",
    },
    {
        "category": "Government & External Financial Assistance",
        "name": "Study Subsidy Scheme for Designated Professions/Sectors",
        "identifier": "SSSDP",
        "note": "Tuition subsidy for designated programmes, applied to specific programme codes such as JSSA01 and JSSA02.",
    },
    {
        "category": "Bursaries & Hardship Funds",
        "name": "Student Emergency Relief Fund",
        "identifier": "",
        "note": "For students facing sudden, unforeseen financial crises.",
    },
    {
        "category": "Bursaries & Hardship Funds",
        "name": "Maintenance Bursary",
        "identifier": "",
        "note": "For students with demonstrated financial need to cover daily living expenses.",
    },
    {
        "category": "Bursaries & Hardship Funds",
        "name": "Caritas Bursary",
        "identifier": "",
        "note": "Hardship fund provided through the Caritas network.",
    },
]


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _identifier_match_span(identifier: str, query: str) -> Optional[Tuple[int, int]]:
    if not identifier:
        return None
    parts = [re.escape(part) for part in re.findall(r"[a-z0-9]+", identifier.lower())]
    if not parts:
        return None
    separator = r"[-\s]?"
    identifier_pattern = separator.join(parts)
    pattern = rf"(?<![a-z0-9-/]){identifier_pattern}(?![a-z0-9-/])"
    match = re.search(pattern, query.lower())
    if match and identifier.lower() == "ops":
        before = query.lower()[:match.start()]
        after = query.lower()[match.end():]
        if re.search(r"\bco\s+$", before) or re.match(r"\s+options?\b", after):
            return None
    return match.span() if match else None


def scholarship_aliases() -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    for item in SCHOLARSHIPS:
        name = item["name"]
        identifier = item["identifier"]
        aliases[name.lower()] = [identifier] if identifier else []
        if identifier:
            aliases[identifier.lower()] = [name]
            aliases[_normalise(identifier)] = [name]
    return {key: value for key, value in aliases.items() if value}


def find_scholarships_in_query(query: str) -> List[Dict[str, str]]:
    query_lower = query.lower()
    normalised_query = _normalise(query)
    matches: List[Dict[str, str]] = []

    for item in SCHOLARSHIPS:
        name = item["name"]
        identifier = item["identifier"]
        matched = _identifier_match_span(identifier, query) is not None
        if not matched:
            matched = name.lower() in query_lower or _normalise(name) in normalised_query
        if matched:
            matches.append(item)

    return matches


def is_scholarship_reference_query(query: str) -> bool:
    query_lower = query.lower()
    if any(
        term in query_lower
        for term in [
            "scholarship",
            "scholarships",
            "bursary",
            "bursaries",
            "financial aid",
            "financial assistance",
            "financial support",
            "student finance",
            "student loan",
            "student loans",
            "subsidy",
            "subsidies",
            "loan scheme",
            "loan schemes",
            "grant",
            "grants",
            "hardship fund",
            "hardship funds",
        ]
    ):
        return True
    return bool(find_scholarships_in_query(query))


def format_scholarship_reference() -> str:
    lines = [
        "SFU scholarship, bursary, subsidy, and financial assistance reference.",
        "Use these mappings when a user asks about scholarship names, identifiers, bursaries, subsidies, loans, or financial aid.",
        "",
    ]
    current_category: Optional[str] = None
    for item in SCHOLARSHIPS:
        category = item["category"]
        if category != current_category:
            current_category = category
            lines.append(f"{category}:")

        identifier = f" ({item['identifier']})" if item["identifier"] else ""
        note = f" - {item['note']}" if item["note"] else ""
        lines.append(f"- {item['name']}{identifier}{note}")

    return "\n".join(lines)


def build_scholarship_reference_doc(query: str) -> Optional[Dict]:
    if not is_scholarship_reference_query(query):
        return None

    return {
        "id": SCHOLARSHIP_REFERENCE_ID,
        "document": format_scholarship_reference(),
        "metadata": {
            "section": "Scholarship Reference",
            "source": "SFU scholarship and financial aid registry",
            "structured": True,
            "type": "scholarship_reference",
        },
        "similarity": 1.0,
        "retrieval_score": 1.0,
        "rank": 0,
    }


def get_scholarship_reference_source(source_id: str) -> Optional[Dict]:
    if source_id not in {SCHOLARSHIP_REFERENCE_ID, f"doc_{SCHOLARSHIP_REFERENCE_ID}"}:
        return None

    metadata = {
        "section": "Scholarship Reference",
        "source": "SFU scholarship and financial aid registry",
        "structured": True,
        "type": "scholarship_reference",
    }
    return {
        "source_id": source_id,
        "section": metadata["section"],
        "source_file": metadata["source"],
        "content": format_scholarship_reference(),
        "metadata": metadata,
        "total_chunks": 1,
    }
