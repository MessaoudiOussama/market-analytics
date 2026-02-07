"""
Parquet Data Exporter
----------------------
Exports data from PostgreSQL into Parquet files for Streamlit Cloud deployment.

Since Streamlit Cloud can't connect to your local PostgreSQL database,
we export the data as Parquet files that are committed to the repository.
The dashboard reads from these files when no database is available.

Usage:
    python -m ingestion.export_parquet
"""

import logging
import os

import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://airflow:airflow@localhost:5432/market_analytics",
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def export_all():
    """Export all dashboard datasets to Parquet files."""
    engine = create_engine(DATABASE_URL)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    datasets = {
        "speeches.parquet": """
            SELECT
                s.id,
                s.source,
                s.speaker,
                s.title,
                s.published_at,
                s.url,
                ss.sentiment,
                ss.score AS sentiment_score,
                ss.model_name
            FROM speeches s
            LEFT JOIN sentiment_scores ss ON s.id = ss.speech_id
            ORDER BY s.published_at DESC
        """,
        "market_data.parquet": """
            SELECT
                id,
                symbol,
                speech_date,
                price_at_speech,
                price_1d_after,
                price_1w_after,
                change_1d_pct,
                change_1w_pct,
                recorded_at
            FROM market_data
            ORDER BY speech_date DESC
        """,
        "speeches_with_market.parquet": """
            SELECT
                s.id AS speech_id,
                s.source,
                s.speaker,
                s.title,
                s.published_at,
                ss.sentiment,
                ss.score AS sentiment_score,
                md.symbol,
                md.price_at_speech,
                md.price_1d_after,
                md.price_1w_after,
                md.change_1d_pct,
                md.change_1w_pct
            FROM speeches s
            JOIN sentiment_scores ss ON s.id = ss.speech_id
            JOIN market_data md ON DATE(s.published_at) = DATE(md.speech_date)
            ORDER BY s.published_at DESC
        """,
        "sentiment_summary.parquet": """
            SELECT
                s.source,
                s.speaker,
                ss.sentiment,
                COUNT(*) AS count,
                AVG(ss.score) AS avg_score
            FROM speeches s
            JOIN sentiment_scores ss ON s.id = ss.speech_id
            GROUP BY s.source, s.speaker, ss.sentiment
            ORDER BY s.source, s.speaker, ss.sentiment
        """,
    }

    for filename, query in datasets.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        logger.info(f"Exporting {filename}...")

        df = pd.read_sql(query, engine)
        df.to_parquet(filepath, index=False)

        logger.info(f"  Exported {len(df)} rows to {filepath}")

    logger.info("=" * 60)
    logger.info("All datasets exported successfully!")
    logger.info(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    export_all()
