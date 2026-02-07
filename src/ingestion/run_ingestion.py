"""
Ingestion Pipeline Runner
--------------------------
Orchestrates the full ingestion process:
    1. Scrape ECB speeches
    2. Scrape Fed speeches
    3. Fetch market data for each speech date
    4. Store everything in PostgreSQL

Usage:
    python -m ingestion.run_ingestion
"""

import logging

from database.connection import init_db
from database.repository import store_market_snapshot, store_speech
from ingestion.ecb_scraper import fetch_ecb_speeches
from ingestion.fed_scraper import fetch_fed_speeches
from ingestion.market_data import fetch_market_data_for_date
from nlp.run_analysis import run_sentiment_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ecb_ingestion() -> int:
    """Scrape ECB speeches and store in database."""
    logger.info("=" * 60)
    logger.info("Starting ECB ingestion")
    logger.info("=" * 60)

    speeches = fetch_ecb_speeches()
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

    logger.info(f"ECB ingestion complete: {stored}/{len(speeches)} new speeches stored")
    return stored


def run_fed_ingestion() -> int:
    """Scrape Fed speeches and store in database."""
    logger.info("=" * 60)
    logger.info("Starting Fed ingestion")
    logger.info("=" * 60)

    speeches = fetch_fed_speeches()
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

    logger.info(f"Fed ingestion complete: {stored}/{len(speeches)} new speeches stored")
    return stored


def run_market_data_ingestion() -> int:
    """Fetch market data for all speech dates in the database."""
    logger.info("=" * 60)
    logger.info("Starting market data ingestion")
    logger.info("=" * 60)

    from database.repository import get_speeches

    speeches = get_speeches(limit=500)
    stored = 0

    # Get unique dates to avoid fetching market data multiple times for the same date
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

    logger.info(f"Market data ingestion complete: {stored} snapshots stored")
    return stored


def main():
    """Run the full ingestion pipeline."""
    logger.info("🚀 Starting full ingestion pipeline")

    # Initialize database tables (safe to run multiple times)
    init_db()

    # Step 1: Scrape speeches
    ecb_count = run_ecb_ingestion()
    fed_count = run_fed_ingestion()

    # Step 2: Fetch market data
    market_count = run_market_data_ingestion()

    # Step 3: Run NLP sentiment analysis
    nlp_count = run_sentiment_analysis()

    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  ECB speeches stored:     {ecb_count}")
    logger.info(f"  Fed speeches stored:     {fed_count}")
    logger.info(f"  Market snapshots stored:  {market_count}")
    logger.info(f"  Speeches analyzed (NLP):  {nlp_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
