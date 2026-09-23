"""
src/news/relevance.py

Classifies ONE article (already fetched, from any of the three tiered
queries - see src/news/news_analyzer.py) into exactly one of:

    COMPANY | SECTOR | MACRO | IRRELEVANT

and a 0-100 relevance_score, using the SAME company metadata every
other News & Risk Layer module already has (src/news/company_sector_map.py)
- NOT "which query fetched it". This matters because a query is only a
best-effort filter (the News API's own relevance ranking); the project
spec's own example makes this explicit: a sector-tier article about
"Government changes regulations for private hospitals" never mentions
Apollo Hospitals by name, so a naive "does it mention the company"
check would call it IRRELEVANT, but it should be SECTOR. Conversely, a
company-tier query can surface an article that only shares a word with
the company's name and isn't really about it at all.

DELIBERATELY RULE-BASED, not a second ML classifier: keyword/alias
matching against COMPANY_METADATA's own aliases/sector_terms/macro_terms
is transparent (spec #12's "do not simply average everything" ethos
applies here too - a human can see exactly why an article landed in a
given bucket) and needs no extra model dependency. It is intentionally
a supporting signal, same caveat as src/news/event_detector.py.

SCORING (matches project spec's suggested bands exactly):
    COMPANY:    90-100  (company name/ticker/alias found)
    SECTOR:     60-89   (no company mention, but sector terms found)
    MACRO:      30-59   (no company/sector mention, but a genuinely
                         important macro term found)
    IRRELEVANT: 0-29    (none of the above - rejected by the analyzer)
"""

import re
from dataclasses import dataclass
from typing import List

COMPANY = "COMPANY"
SECTOR = "SECTOR"
MACRO = "MACRO"
IRRELEVANT = "IRRELEVANT"

# Score bands (spec-mandated) - kept as named constants, not buried
# magic numbers, per the project's "don't hide thresholds" convention.
COMPANY_SCORE_MIN, COMPANY_SCORE_MAX = 90, 100
SECTOR_SCORE_MIN, SECTOR_SCORE_MAX = 60, 89
MACRO_SCORE_MIN, MACRO_SCORE_MAX = 30, 59
IRRELEVANT_SCORE_MAX = 29

_STOPWORDS = {"of", "the", "in", "for", "to", "and", "a", "an", "by", "on", "at", "is", "are"}


@dataclass
class RelevanceResult:
    relevance_type: str      # COMPANY | SECTOR | MACRO | IRRELEVANT
    relevance_score: int     # 0-100
    matched_terms: List[str]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _phrase_in_text(phrase: str, text_lower: str) -> bool:
    """Whole-phrase, case-insensitive substring match. Short (<=3 char)
    aliases like an all-caps ticker fragment are matched as a whole
    word only, to avoid e.g. ticker 'ITC' matching inside an unrelated
    word - every other alias/term is matched as a plain substring,
    which is intentionally permissive (spec favors recall for company
    mentions: "Apollo Hospitals Enterprise" should still match text
    that only says "Apollo Hospitals")."""
    phrase_lower = phrase.lower()
    if len(phrase) <= 3:
        return re.search(rf"\b{re.escape(phrase_lower)}\b", text_lower) is not None
    return phrase_lower in text_lower


def _content_words(term: str) -> List[str]:
    return [w for w in re.findall(r"[a-z]+", term.lower()) if w not in _STOPWORDS and len(w) > 2]


def _term_matches(term: str, text_lower: str) -> bool:
    """
    True if `term` is relevant to `text_lower` either as an exact
    phrase, OR - since curated terms are necessarily a fixed phrasing
    ("RBI interest rate decision") while real headlines paraphrase
    freely ("RBI cuts interest rate by 25 bps") - if MOST of the
    term's significant words all appear somewhere in the text
    (order-independent). Requires at least 2 significant words and at
    least 70% of them present, so a single incidental word overlap
    (e.g. just "rate") never counts as a match on its own - this is
    still a transparent, debuggable rule (matched_terms shows exactly
    which curated term fired), just tolerant of real-world phrasing.
    """
    if _phrase_in_text(term, text_lower):
        return True
    words = _content_words(term)
    if len(words) < 2:
        return False
    hits = sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", text_lower))
    return hits / len(words) >= 0.7


def classify_article(text: str, metadata: dict) -> RelevanceResult:
    """
    `text` should be headline (+ description if available). `metadata`
    is a src.news.company_sector_map.get_metadata(symbol) dict.
    """
    text_lower = _normalize(text)
    if not text_lower:
        return RelevanceResult(IRRELEVANT, 0, [])

    # --- Tier 1: COMPANY - any alias or ticker mentioned ---
    company_matches = [a for a in ([metadata["ticker"]] + metadata["aliases"])
                        if _phrase_in_text(a, text_lower)]
    if company_matches:
        # More independent mentions (e.g. both the full name AND the
        # ticker appear) -> slightly higher confidence this article is
        # really about the company, not a passing mention.
        score = min(COMPANY_SCORE_MAX, COMPANY_SCORE_MIN + 2 * (len(set(company_matches)) - 1))
        return RelevanceResult(COMPANY, score, sorted(set(company_matches)))

    # --- Tier 2: SECTOR - sector terms, no company mention needed ---
    sector_matches = [t for t in metadata["sector_terms"] if _term_matches(t, text_lower)]
    # A generic sector-name mention (e.g. just "healthcare", "hospital",
    # "IT sector", "banking") also counts even if the fuller phrase
    # ("Indian healthcare sector") doesn't appear verbatim.
    # BUG FIX (2026-09): metadata["sector"] can legitimately be "" (the
    # single-tier analyze_company_news() back-compat path builds an
    # ad-hoc metadata dict with no real sector - see news_analyzer.py).
    # The old `.split("/")[0].split()[0]` crashed with IndexError the
    # moment an empty string reached it, for ANY article that didn't
    # directly mention the company by name - the most common case, not
    # an edge case. first_word_lower() below returns "" instead of
    # raising when there's nothing to split.
    def _first_word_lower(text: str) -> str:
        parts = text.split("/")[0].split()
        return parts[0].lower() if parts else ""

    sector_word = _first_word_lower(metadata["sector"])  # e.g. "Healthcare" -> "healthcare"
    industry_words = [w.lower() for w in metadata["industry"].replace("&", " ").split() if len(w) > 3]
    generic_sector_hit = (sector_word and _phrase_in_text(sector_word, text_lower)) or any(
        _phrase_in_text(w, text_lower) for w in industry_words
    )
    if sector_matches or generic_sector_hit:
        matched = sector_matches or [metadata["sector"]]
        score = min(SECTOR_SCORE_MAX, SECTOR_SCORE_MIN + 5 * (len(sector_matches) if sector_matches else 0))
        return RelevanceResult(SECTOR, score, matched)

    # --- Tier 3: MACRO - only genuinely important macro terms count ---
    macro_matches = [t for t in metadata["macro_terms"] if _term_matches(t, text_lower)]
    if macro_matches:
        score = min(MACRO_SCORE_MAX, MACRO_SCORE_MIN + 5 * (len(macro_matches) - 1))
        return RelevanceResult(MACRO, score, sorted(set(macro_matches)))

    # --- Nothing matched: this is generic/unrelated noise ---
    return RelevanceResult(IRRELEVANT, 0, [])
