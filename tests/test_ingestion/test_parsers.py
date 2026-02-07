"""
Tests for scraper parsing functions.
Validates date parsing and data extraction without network calls.
"""

from datetime import datetime

from ingestion.fed_scraper import parse_fed_date


class TestParseFedDate:
    """Test Fed date parsing from different formats."""

    def test_mm_dd_yyyy_format(self):
        result = parse_fed_date("01/15/2025")
        assert result == datetime(2025, 1, 15)

    def test_month_dd_yyyy_format(self):
        result = parse_fed_date("January 15, 2025")
        assert result == datetime(2025, 1, 15)

    def test_whitespace_handling(self):
        result = parse_fed_date("  01/15/2025  ")
        assert result == datetime(2025, 1, 15)

    def test_invalid_date_returns_now(self):
        result = parse_fed_date("not-a-date")
        # Should return datetime.now() — just check it's recent
        assert isinstance(result, datetime)
        assert (datetime.now() - result).total_seconds() < 5
