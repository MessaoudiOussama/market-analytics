"""
Market Data Fetcher
--------------------
Fetches market data (prices, changes) around speech timestamps
using Yahoo Finance (yfinance).

Tracks key instruments that are most affected by central bank communications:
    - EUR/USD (EURUSD=X)
    - S&P 500 (^GSPC)
    - US 10Y Treasury Yield (^TNX)
    - Gold (GC=F)
    - Euro Stoxx 50 (^STOXX50E)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

# Key symbols to track
DEFAULT_SYMBOLS = {
    "EURUSD=X": "EUR/USD",
    "^GSPC": "S&P 500",
    "^TNX": "US 10Y Treasury",
    "GC=F": "Gold",
    "^STOXX50E": "Euro Stoxx 50",
}


@dataclass
class MarketSnapshot:
    """Market data around a specific timestamp."""

    symbol: str
    symbol_name: str
    price_at_speech: float | None
    price_1d_after: float | None
    price_1w_after: float | None
    change_1d_pct: float | None
    change_1w_pct: float | None
    speech_date: datetime


def fetch_market_data_for_date(
    speech_date: datetime,
    symbols: dict[str, str] | None = None,
) -> list[MarketSnapshot]:
    """
    Fetch market data around a speech date.

    Gets the closing price on the speech date and calculates
    percentage changes for 1 day and 1 week after.

    Args:
        speech_date: Date of the speech.
        symbols: Dict mapping Yahoo Finance symbols to display names.

    Returns:
        List of MarketSnapshot objects for each symbol.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    # Fetch data: from 2 days before to 10 days after (to account for weekends)
    start_date = speech_date - timedelta(days=2)
    end_date = speech_date + timedelta(days=10)

    snapshots = []

    for symbol, name in symbols.items():
        try:
            snapshot = _fetch_single_symbol(symbol, name, speech_date, start_date, end_date)
            if snapshot:
                snapshots.append(snapshot)
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")

    return snapshots


def _fetch_single_symbol(
    symbol: str,
    name: str,
    speech_date: datetime,
    start_date: datetime,
    end_date: datetime,
) -> MarketSnapshot | None:
    """
    Fetch market data for a single symbol around a speech date.

    Args:
        symbol: Yahoo Finance ticker symbol.
        name: Human-readable name.
        speech_date: Date of the speech.
        start_date: Start of data window.
        end_date: End of data window.

    Returns:
        MarketSnapshot or None if data is unavailable.
    """
    logger.info(f"Fetching {name} ({symbol}) for date {speech_date.date()}")

    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

    if hist.empty:
        logger.warning(f"No data for {symbol} in range {start_date.date()} to {end_date.date()}")
        return None

    # Find the closest trading day to the speech date
    price_at_speech = _get_closest_price(hist, speech_date)

    if price_at_speech is None:
        logger.warning(f"Could not find price for {symbol} near {speech_date.date()}")
        return None

    # Get price 1 day after
    price_1d = _get_closest_price(hist, speech_date + timedelta(days=1))

    # Get price 1 week after
    price_1w = _get_closest_price(hist, speech_date + timedelta(days=7))

    # Calculate percentage changes
    change_1d = _calc_pct_change(price_at_speech, price_1d)
    change_1w = _calc_pct_change(price_at_speech, price_1w)

    return MarketSnapshot(
        symbol=name,
        symbol_name=name,
        price_at_speech=round(price_at_speech, 4),
        price_1d_after=round(price_1d, 4) if price_1d else None,
        price_1w_after=round(price_1w, 4) if price_1w else None,
        change_1d_pct=round(change_1d, 4) if change_1d else None,
        change_1w_pct=round(change_1w, 4) if change_1w else None,
        speech_date=speech_date,
    )


def _get_closest_price(hist, target_date: datetime) -> float | None:
    """
    Get the closing price on or closest to the target date.

    If the target date is not a trading day (weekend/holiday),
    uses the most recent previous trading day.

    Args:
        hist: yfinance history DataFrame.
        target_date: Target datetime.

    Returns:
        Closing price or None.
    """
    if hist.empty:
        return None

    target = target_date.strftime("%Y-%m-%d")

    # Exact match
    if target in hist.index.strftime("%Y-%m-%d"):
        mask = hist.index.strftime("%Y-%m-%d") == target
        return float(hist.loc[mask, "Close"].iloc[0])

    # Find closest previous date
    prior = hist[hist.index <= target_date.strftime("%Y-%m-%d")]
    if not prior.empty:
        return float(prior["Close"].iloc[-1])

    # Find closest next date
    after = hist[hist.index >= target_date.strftime("%Y-%m-%d")]
    if not after.empty:
        return float(after["Close"].iloc[0])

    return None


def _calc_pct_change(price_before: float | None, price_after: float | None) -> float | None:
    """Calculate percentage change between two prices."""
    if price_before is None or price_after is None or price_before == 0:
        return None
    return ((price_after - price_before) / price_before) * 100


def fetch_bulk_market_data(
    start_date: str,
    end_date: str,
    symbols: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Fetch bulk historical market data for a date range.

    Useful for initial data loading or backfilling.

    Args:
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        symbols: Dict mapping Yahoo Finance symbols to display names.

    Returns:
        Dict mapping symbol names to their historical DataFrames.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    results = {}
    for symbol, name in symbols.items():
        logger.info(f"Fetching bulk data for {name} ({symbol})")
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                results[name] = hist
                logger.info(f"  Got {len(hist)} rows for {name}")
            else:
                logger.warning(f"  No data for {name}")
        except Exception as e:
            logger.error(f"  Failed: {e}")

    return results


if __name__ == "__main__":
    # Quick test: fetch market data for a specific date
    logging.basicConfig(level=logging.INFO)

    test_date = datetime(2025, 1, 15)
    print(f"Fetching market data around {test_date.date()}")

    snapshots = fetch_market_data_for_date(test_date)
    for s in snapshots:
        print(f"\n{s.symbol_name}:")
        print(f"  Price at speech: {s.price_at_speech}")
        print(f"  Price 1d after:  {s.price_1d_after} ({s.change_1d_pct}%)")
        print(f"  Price 1w after:  {s.price_1w_after} ({s.change_1w_pct}%)")
