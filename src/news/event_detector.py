"""
src/news/event_detector.py

Detects potentially high-impact company events in a news headline/
description and assigns a severity 0-4 (None/Low/Medium/High/Critical).

NOT keyword-matching alone (per project spec #6): a keyword match only
proposes a CANDIDATE event + a base severity. The final severity is
then adjusted using that same article's sentiment score (already
computed by src/news/sentiment.py) - e.g. a headline matching "lawsuit"
but with a clearly POSITIVE sentiment (like "Company wins major
lawsuit") has its severity reduced, since the keyword alone would have
overstated the risk. This combination is intentionally simple and
transparent (a human can read the rule and see why a score came out
the way it did) rather than a second opaque model.

The keyword lists are necessarily incomplete - this is a supporting
signal for the Risk Engine, not a certified news-classification
system. Add more phrases to EVENT_RULES as needed; nothing else in
this module needs to change to pick them up.
"""

from dataclasses import dataclass
from typing import List

NEGATIVE = "negative"
POSITIVE = "positive"

SEVERITY_NONE = 0
SEVERITY_LOW = 1
SEVERITY_MEDIUM = 2
SEVERITY_HIGH = 3
SEVERITY_CRITICAL = 4

SEVERITY_LABELS = {
    SEVERITY_NONE: "None", SEVERITY_LOW: "Low", SEVERITY_MEDIUM: "Medium",
    SEVERITY_HIGH: "High", SEVERITY_CRITICAL: "Critical",
}

# (event_type, direction, base_severity, keyword phrases - matched
# case-insensitively as substrings of the headline+description).
EVENT_RULES = [
    # --- Negative, high-impact ---
    ("fraud", NEGATIVE, SEVERITY_CRITICAL, ["fraud", "scam", "embezzle"]),
    ("bankruptcy", NEGATIVE, SEVERITY_CRITICAL, ["bankruptcy", "insolvency", "insolvent", "files for chapter 11", "liquidation"]),
    ("accounting_issue", NEGATIVE, SEVERITY_CRITICAL, ["accounting fraud", "accounting irregularit", "restated earnings", "restate its earnings", "financial irregularit"]),
    ("regulatory_investigation", NEGATIVE, SEVERITY_HIGH, ["investigation", "probe launched", "under scrutiny", "regulatory scrutiny", "sebi action", "sec investigation"]),
    ("government_action", NEGATIVE, SEVERITY_HIGH, ["government action", "raided", "license revoked", "banned by", "seized by authorities"]),
    ("major_lawsuit", NEGATIVE, SEVERITY_HIGH, ["lawsuit", "sued for", "sues", "litigation against", "class action"]),
    ("credit_downgrade", NEGATIVE, SEVERITY_HIGH, ["credit downgrade", "rating downgrade", "downgraded to", "downgrades rating"]),
    ("major_accident", NEGATIVE, SEVERITY_HIGH, ["explosion at", "fire at", "plant accident", "accident kills", "factory collapse"]),
    ("cyberattack", NEGATIVE, SEVERITY_HIGH, ["cyberattack", "cyber attack", "data breach", "hacked", "ransomware"]),
    ("regulatory_penalty", NEGATIVE, SEVERITY_HIGH, ["fined", "penalty imposed", "regulatory fine", "sebi fine", "sec fine"]),
    ("ceo_resignation", NEGATIVE, SEVERITY_MEDIUM, ["ceo resigns", "ceo steps down", "ceo quits", "chief executive resigns"]),
    ("management_change", NEGATIVE, SEVERITY_MEDIUM, ["management shakeup", "board resigns", "cfo resigns", "management overhaul"]),
    ("product_failure", NEGATIVE, SEVERITY_MEDIUM, ["product recall", "recalls its", "recalled over", "product defect"]),
    ("earnings_miss", NEGATIVE, SEVERITY_MEDIUM, ["misses estimates", "earnings miss", "profit falls", "loss widens", "profit plunges"]),
    ("negative_announcement", NEGATIVE, SEVERITY_LOW, ["profit warning", "guidance cut", "cuts outlook", "warns of lower"]),

    # --- Positive, high-impact ---
    ("major_acquisition", POSITIVE, SEVERITY_HIGH, ["to acquire", "acquisition of", "completes acquisition"]),
    ("earnings_surprise", POSITIVE, SEVERITY_HIGH, ["beats estimates", "profit surges", "record profit", "earnings beat", "beats expectations"]),
    ("large_contract", POSITIVE, SEVERITY_MEDIUM, ["wins contract", "secures order", "bags order", "major deal", "wins deal"]),
    ("major_expansion", POSITIVE, SEVERITY_MEDIUM, ["expansion plan", "new plant", "expands capacity", "to expand"]),
    ("regulatory_approval", POSITIVE, SEVERITY_MEDIUM, ["gets approval", "regulatory approval", "cleared by regulator", "receives approval"]),
    ("major_partnership", POSITIVE, SEVERITY_MEDIUM, ["partnership with", "strategic alliance", "joint venture", "signs pact"]),
]

# Sentiment magnitude thresholds used to adjust a keyword-matched severity.
_STRONG_SENTIMENT = 0.5
_CONFLICTING_SENTIMENT = 0.3


@dataclass
class EventMatch:
    event_type: str
    direction: str        # "negative" | "positive"
    severity: int          # 0-4, AFTER sentiment adjustment
    base_severity: int     # 0-4, BEFORE sentiment adjustment (keyword-only)
    matched_phrase: str


def detect_events(text: str, sentiment_score: float) -> List[EventMatch]:
    """
    `text` should be the article's headline (+ description if available).
    `sentiment_score` is that SAME article's -1..+1 sentiment (see
    src/news/sentiment.py) - used only to adjust severity, never to
    invent an event that no keyword actually matched.
    """
    text_lower = (text or "").lower()
    if not text_lower:
        return []

    matches = []
    for event_type, direction, base_severity, phrases in EVENT_RULES:
        for phrase in phrases:
            if phrase in text_lower:
                severity = _adjust_severity(base_severity, direction, sentiment_score)
                matches.append(EventMatch(
                    event_type=event_type, direction=direction, severity=severity,
                    base_severity=base_severity, matched_phrase=phrase,
                ))
                break  # one match per event_type is enough - don't double count synonyms
    return matches


def _adjust_severity(base_severity: int, direction: str, sentiment_score: float) -> int:
    """
    Sentiment-based adjustment (see module docstring):
    - Sentiment CONFLICTS with the keyword's expected direction (e.g. a
      "lawsuit" keyword but clearly positive sentiment - likely "wins
      lawsuit" or a resolved dispute) -> severity reduced by 1.
    - Sentiment STRONGLY REINFORCES the expected direction -> severity
      increased by 1 (capped at Critical/4).
    - Otherwise the keyword-derived base_severity stands as-is.
    """
    expected_sign = 1.0 if direction == POSITIVE else -1.0
    aligned = sentiment_score * expected_sign  # positive if sentiment agrees with direction

    if aligned <= -_CONFLICTING_SENTIMENT:
        return max(SEVERITY_NONE, base_severity - 1)
    if aligned >= _STRONG_SENTIMENT:
        return min(SEVERITY_CRITICAL, base_severity + 1)
    return base_severity


def top_severity(matches: List[EventMatch]) -> int:
    """The single worst (highest) severity across all matches, or SEVERITY_NONE if none."""
    if not matches:
        return SEVERITY_NONE
    return max(m.severity for m in matches)
