"""
src/news/sentiment.py

Local (no-API) sentiment scoring for news headlines/descriptions.
Never calls an external service and never randomly generates a score -
every score comes from an actual model running on the actual text.

TWO MODELS, GRACEFUL FALLBACK
-------------------------------
1. DEFAULT - VADER (via the `vaderSentiment` pip package). A lexicon/
   rule-based sentiment analyzer. It ships its lexicon INSIDE the pip
   package (no separate download/internet access needed at runtime,
   unlike NLTK's vader_lexicon corpus), is fast, and is a reasonable
   general-purpose baseline for short text like headlines. It is NOT
   finance-specific - "tanks" or "crashes" score negative correctly,
   but finance jargon like "beats estimates" or "misses guidance" may
   not land the way a domain model would.
   Install: pip install vaderSentiment (already in requirements.txt)

2. OPTIONAL ENHANCEMENT - FinBERT (ProsusAI/finbert on HuggingFace), a
   BERT model fine-tuned specifically on financial text. This is
   PREFERRED when available per the project spec, but is NOT a hard
   dependency here: it requires `transformers` + `torch` (a large
   install, GPU recommended for speed) which most environments running
   this dashboard won't have. This module tries to load it ONCE at
   first use; if the packages aren't installed, or the model can't be
   downloaded/loaded for any reason, it silently and permanently falls
   back to VADER for the rest of the process - this is the "gracefully
   handle model-loading failures" requirement.
   Opt in with:
       pip install transformers torch
   (first use then downloads the ~440MB model from HuggingFace and
   caches it locally under ~/.cache/huggingface/).

Every result reports which model actually produced it (`model_name`),
so the UI/tests can be honest about what generated a given score
instead of silently mixing two different models under one label.
"""

from dataclasses import dataclass
from typing import Optional

POSITIVE = "POSITIVE"
NEUTRAL = "NEUTRAL"
NEGATIVE = "NEGATIVE"

VADER_MODEL_NAME = "VADER (vaderSentiment, lexicon-based, general-purpose)"
FINBERT_MODEL_NAME = "FinBERT (ProsusAI/finbert, finance-specific)"

# Score -> label thresholds, matching the project spec's -1..+1 scale.
_POSITIVE_THRESHOLD = 0.25
_NEGATIVE_THRESHOLD = -0.25


@dataclass
class SentimentResult:
    label: str        # POSITIVE | NEUTRAL | NEGATIVE
    score: float       # -1.0 (very negative) .. +1.0 (very positive)
    model_name: str    # which model actually produced this result


def label_for_score(score: float) -> str:
    """Public helper: -1..+1 score -> POSITIVE/NEUTRAL/NEGATIVE label,
    using the same thresholds analyze_sentiment() itself uses."""
    return _label_for_score(score)


def _label_for_score(score: float) -> str:
    if score >= _POSITIVE_THRESHOLD:
        return POSITIVE
    if score <= _NEGATIVE_THRESHOLD:
        return NEGATIVE
    return NEUTRAL


# ---------------------------------------------------------------------------
# VADER (always available - hard dependency)
# ---------------------------------------------------------------------------

_vader_analyzer = None


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def _analyze_vader(text: str) -> SentimentResult:
    scores = _get_vader().polarity_scores(text)
    compound = float(scores["compound"])  # already -1..+1
    return SentimentResult(label=_label_for_score(compound), score=compound, model_name=VADER_MODEL_NAME)


# ---------------------------------------------------------------------------
# FinBERT (optional - loaded lazily, disabled permanently on first failure)
# ---------------------------------------------------------------------------

_finbert_pipeline = None
_finbert_unavailable = False  # sticky - don't retry a slow failing import/download every call


def _get_finbert():
    global _finbert_pipeline, _finbert_unavailable
    if _finbert_unavailable:
        return None
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline
        _finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        return _finbert_pipeline
    except Exception:
        # Covers: transformers/torch not installed, no internet to download
        # the model, out of memory, corrupted cache, etc. - all treated the
        # same way: fall back to VADER for the rest of this process.
        _finbert_unavailable = True
        return None


def _analyze_finbert(text: str) -> Optional[SentimentResult]:
    model = _get_finbert()
    if model is None:
        return None
    try:
        result = model(text[:512])[0]  # FinBERT truncates around 512 tokens anyway
        label_raw = result["label"].upper()  # "positive" | "negative" | "neutral"
        confidence = float(result["score"])  # 0..1 confidence in that label
        signed = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0}.get(label_raw, 0.0)
        score = signed * confidence
        label = {"POSITIVE": POSITIVE, "NEGATIVE": NEGATIVE, "NEUTRAL": NEUTRAL}.get(label_raw, NEUTRAL)
        return SentimentResult(label=label, score=score, model_name=FINBERT_MODEL_NAME)
    except Exception:
        return None  # a single bad inference call shouldn't take down the whole request


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_sentiment(text: str, prefer_finbert: bool = True) -> SentimentResult:
    """
    Score `text` (a headline, or headline+description). Tries FinBERT
    first (if installed and prefer_finbert=True), falls back to VADER
    otherwise - VADER is a hard dependency so this function always
    returns a real result, never raises for a missing model.
    """
    text = (text or "").strip()
    if not text:
        return SentimentResult(label=NEUTRAL, score=0.0, model_name=VADER_MODEL_NAME)

    if prefer_finbert:
        finbert_result = _analyze_finbert(text)
        if finbert_result is not None:
            return finbert_result

    return _analyze_vader(text)
