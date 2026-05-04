"""Small text utilities used by the Project 1 RAG pipeline.

The implementation intentionally stays dependency-light so that the final
`main.py` can run on the grader even if optional RAG packages are unavailable.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "you", "your", "yours", "yourself", "yourselves",
    # Domain words that occur almost everywhere in this corpus.
    "rapidfire", "rapidfireai", "ai", "oss", "documentation", "docs",
}

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?")
IDENTIFIER_RE = re.compile(r"`([^`]+)`|:code:`([^`]+)`|:func:`([^`]+)`|:class:`([^`]+)`|[A-Za-z_][A-Za-z0-9_]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`])")
NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def _split_identifier(token: str) -> List[str]:
    token = token.strip("_").lower()
    if not token:
        return []
    pieces = re.split(r"[_\W]+", token)
    out: List[str] = []
    for piece in pieces:
        if not piece:
            continue
        # Split camelCase/PascalCase after lowering fallback has not helped; keep
        # the full identifier as well because API names are strong retrieval cues.
        camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", piece).lower().split()
        out.extend(camel or [piece])
    return out


def _simple_stems(token: str) -> List[str]:
    """Return conservative stem-like variants for matching inflections."""
    stems: List[str] = []
    suffixes = (
        ("difference", "differ"),
        ("different", "differ"),
        ("configs", "config"),
        ("generators", "generator"),
        ("retrievers", "retriever"),
        ("rerankers", "reranker"),
        ("metrics", "metric"),
        ("ies", "y"),
        ("ing", ""),
        ("ed", ""),
        ("ions", "ion"),
        ("es", ""),
        ("s", ""),
    )
    for suffix, replacement in suffixes:
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            stems.append(token[: -len(suffix)] + replacement)
    return stems


def tokenize(text: str, *, keep_stopwords: bool = False) -> List[str]:
    """Tokenize text for lexical retrieval.

    API identifiers are kept as complete lower-case tokens and also decomposed
    into useful parts, e.g. ``RFGridSearch`` contributes ``rfgridsearch`` plus
    ``rf``, ``grid``, and ``search`` when those parts are visible.
    """
    tokens: List[str] = []
    for raw in TOKEN_RE.findall(text):
        lowered = raw.lower()
        candidates = [lowered]
        candidates.extend(_split_identifier(raw))
        candidates.extend(_simple_stems(lowered))
        if lowered in NUMBER_WORDS:
            candidates.append(NUMBER_WORDS[lowered])
        for tok in candidates:
            if len(tok) <= 1 and not tok.isdigit():
                continue
            if not keep_stopwords and tok in STOPWORDS:
                continue
            tokens.append(tok)
    return tokens


def token_counts(text: str) -> Counter:
    return Counter(tokenize(text))


def approx_token_count(text: str) -> int:
    """A conservative, dependency-free token estimate.

    It overestimates slightly for plain English, which is useful for respecting
    the 2,000-token project context budget.
    """
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def extract_special_terms(text: str) -> List[str]:
    """Return exact-match terms such as API names and code identifiers."""
    terms: List[str] = []
    for match in IDENTIFIER_RE.finditer(text):
        grouped = next((g for g in match.groups() if g), None)
        value = grouped or match.group(0)
        value = value.strip("`'\" ")
        code_like = bool(grouped) or "_" in value or bool(re.search(r"[A-Z]{2,}|[a-z][A-Z]|[0-9]", value))
        title_like = value[:1].isupper() and value[1:].islower() and len(value) >= 5
        if code_like and len(value) >= 3 and value.lower() not in STOPWORDS:
            terms.append(value.lower())
        elif title_like and value.lower() not in STOPWORDS:
            terms.append(value.lower())
    # Keep order while deduplicating.
    return list(dict.fromkeys(terms))


def split_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    return [s.strip() for s in SENTENCE_RE.split(cleaned) if s.strip()]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
