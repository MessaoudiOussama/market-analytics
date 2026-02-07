"""
Tests for market data utilities.
Validates calculation logic without requiring API calls.
"""

from ingestion.market_data import _calc_pct_change


class TestCalcPctChange:
    """Test the percentage change calculation."""

    def test_positive_change(self):
        result = _calc_pct_change(100.0, 110.0)
        assert result == 10.0

    def test_negative_change(self):
        result = _calc_pct_change(100.0, 90.0)
        assert result == -10.0

    def test_no_change(self):
        result = _calc_pct_change(100.0, 100.0)
        assert result == 0.0

    def test_none_before(self):
        result = _calc_pct_change(None, 100.0)
        assert result is None

    def test_none_after(self):
        result = _calc_pct_change(100.0, None)
        assert result is None

    def test_zero_before(self):
        result = _calc_pct_change(0, 100.0)
        assert result is None

    def test_small_change(self):
        result = _calc_pct_change(1.0850, 1.0870)
        assert abs(result - 0.1843) < 0.01  # ~0.18% change for EUR/USD
