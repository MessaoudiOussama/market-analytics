"""
Historical Backfill Runner
----------------------------
Runs the historical backfill for ECB and Fed speeches,
then fetches market data and runs sentiment analysis for all new entries.

Usage:
    python -m ingestion.run_backfill
    python -m ingestion.run_backfill --ecb-max 100 --fed-years 2023 2024 2025
"""

import argparse
import logging

from database.connection import init_db
from database.repository import get_speeches, store_market_snapshot, store_speech
from ingestion.ecb_backfill import fetch_ecb_archive
from ingestion.fed_backfill import fetch_fed_archive
from ingestion.market_data import fetch_market_data_for_date
from nlp.run_analysis import run_sentiment_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ecb_backfill(max_speeches: int = 200) -> int:
    """Backfill ECB speeches from the archive."""
    logger.info("=" * 60)
    logger.info(f"Starting ECB backfill (max {max_speeches} speeches)")
    logger.info("=" * 60)

    speeches = fetch_ecb_archive(max_speeches=max_speeches, delay=1.0)
    stored = 0

    for speech in speeches:
        result = store_speech(
            source=speech.source,
            speaker=speech.speaker,
            title=speech.title,
            content=speech.content,
            published_at=speech.published_at,
            url=speech.url,
        )
        if result:
            stored += 1

    logger.info(f"ECB backfill: {stored}/{len(speeches)} new speeches stored")
    return stored


def run_fed_backfill(years: list[int] | None = None) -> int:
    """Backfill Fed speeches from yearly archive pages."""
    if years is None:
        years = [2024, 2025]

    logger.info("=" * 60)
    logger.info(f"Starting Fed backfill (years: {years})")
    logger.info("=" * 60)

    speeches = fetch_fed_archive(years=years, delay=1.0)
    stored = 0

    for speech in speeches:
        result = store_speech(
            source=speech.source,
            speaker=speech.speaker,
            title=speech.title,
            content=speech.content,
            published_at=speech.published_at,
            url=speech.url,
        )
        if result:
            stored += 1

    logger.info(f"Fed backfill: {stored}/{len(speeches)} new speeches stored")
    return stored


def run_market_backfill() -> int:
    """Fetch market data for all speech dates that don't have it yet."""
    logger.info("=" * 60)
    logger.info("Starting market data backfill")
    logger.info("=" * 60)

    speeches = get_speeches(limit=5000)
    stored = 0
    processed_dates = set()

    for speech in speeches:
        date_key = speech.published_at.date()
        if date_key in processed_dates:
            continue
        processed_dates.add(date_key)

        logger.info(f"Fetching market data for {date_key}")
        snapshots = fetch_market_data_for_date(speech.published_at)

        for snapshot in snapshots:
            store_market_snapshot(
                symbol=snapshot.symbol,
                speech_date=snapshot.speech_date,
                price_at_speech=snapshot.price_at_speech,
                price_1d_after=snapshot.price_1d_after,
                price_1w_after=snapshot.price_1w_after,
                change_1d_pct=snapshot.change_1d_pct,
                change_1w_pct=snapshot.change_1w_pct,
            )
            stored += 1

    logger.info(f"Market data backfill: {stored} snapshots stored")
    return stored


def main():
    parser = argparse.ArgumentParser(description="Historical backfill for market-analytics")
    parser.add_argument("--ecb-max", type=int, default=200, help="Max ECB speeches to backfill")
    parser.add_argument(
        "--fed-years", type=int, nargs="+", default=[2024, 2025],
        help="Years to backfill for Fed speeches",
    )
    parser.add_argument("--skip-ecb", action="store_true", help="Skip ECB backfill")
    parser.add_argument("--skip-fed", action="store_true", help="Skip Fed backfill")
    parser.add_argument("--skip-market", action="store_true", help="Skip market data")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip NLP analysis")
    args = parser.parse_args()

    logger.info("Starting historical backfill pipeline")
    init_db()

    ecb_count = 0
    fed_count = 0
    market_count = 0
    nlp_count = 0

    # Step 1: Backfill speeches
    if not args.skip_ecb:
        ecb_count = run_ecb_backfill(max_speeches=args.ecb_max)

    if not args.skip_fed:
        fed_count = run_fed_backfill(years=args.fed_years)

    # Step 2: Fetch market data for all dates
    if not args.skip_market:
        market_count = run_market_backfill()

    # Step 3: Run NLP on all unanalyzed speeches
    if not args.skip_nlp:
        nlp_count = run_sentiment_analysis()

    # Summary
    logger.info("=" * 60)
    logger.info("Backfill complete!")
    logger.info(f"  ECB speeches stored:     {ecb_count}")
    logger.info(f"  Fed speeches stored:     {fed_count}")
    logger.info(f"  Market snapshots stored:  {market_count}")
    logger.info(f"  Speeches analyzed (NLP):  {nlp_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
