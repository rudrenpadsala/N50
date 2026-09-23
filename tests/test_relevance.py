"""
tests/test_relevance.py

Tests for src/news/relevance.py, focused on the fixed IndexError bug:
classify_article() must not crash when metadata["sector"] is an empty
string (the shape used by news_analyzer.analyze_company_news()'s
single-tier back-compat pipeline for any article that doesn't mention
the company by name).
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.news import relevance  # noqa: E402


def _minimal_metadata(company_name="Reliance Industries"):
    return dict(ticker=company_name, sector="", industry="", aliases=[company_name],
                sector_terms=[], macro_terms=[])


def test_empty_sector_does_not_crash_on_non_matching_article():
    """Regression test: this used to raise IndexError because
    "".split("/")[0].split()[0] indexes into an empty list."""
    result = relevance.classify_article("Company reports strong earnings", _minimal_metadata())
    assert result.relevance_type in (relevance.IRRELEVANT, relevance.MACRO)


def test_company_mention_still_classified_as_company_with_empty_sector():
    result = relevance.classify_article("Reliance Industries reports strong earnings",
                                         _minimal_metadata())
    assert result.relevance_type == relevance.COMPANY


def test_normal_sector_metadata_still_matches_generic_sector_mention():
    metadata = dict(ticker="Apollo Hospitals", sector="Healthcare/Hospitals", industry="Hospitals",
                     aliases=["Apollo Hospitals"], sector_terms=[], macro_terms=[])
    result = relevance.classify_article("Government changes regulations for private hospitals",
                                         metadata)
    assert result.relevance_type == relevance.SECTOR
