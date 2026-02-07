"""
Data Loader for Streamlit Dashboard
-------------------------------------
Loads data from Parquet files (Streamlit Cloud) or PostgreSQL (local development).

Priority:
    1. Parquet files in data/ folder (for Streamlit Cloud deployment)
    2. PostgreSQL database (for local development with Docker)
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Data source configuration ─────────────────────────────
# Look for Parquet files relative to the project root
_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# Check if Parquet files exist (Streamlit Cloud mode)
_USE_PARQUET = (_DATA_DIR / "speeches.parquet").exists()

# PostgreSQL fallback (local Docker mode)
_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://airflow:airflow@localhost:5432/market_analytics",
)


def _get_engine():
    """Lazy-load SQLAlchemy engine (only when needed)."""
    from sqlalchemy import create_engine

    return create_engine(_DATABASE_URL)


@st.cache_data(ttl=600)
def load_speeches() -> pd.DataFrame:
    """Load all speeches with their sentiment scores."""
    if _USE_PARQUET:
        return pd.read_parquet(_DATA_DIR / "speeches.parquet")

    query = """
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
    """
    return pd.read_sql(query, _get_engine())


@st.cache_data(ttl=600)
def load_market_data() -> pd.DataFrame:
    """Load all market data snapshots."""
    if _USE_PARQUET:
        return pd.read_parquet(_DATA_DIR / "market_data.parquet")

    query = """
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
    """
    return pd.read_sql(query, _get_engine())


@st.cache_data(ttl=600)
def load_speeches_with_market() -> pd.DataFrame:
    """Load speeches joined with market data for correlation analysis."""
    if _USE_PARQUET:
        return pd.read_parquet(_DATA_DIR / "speeches_with_market.parquet")

    query = """
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
    """
    return pd.read_sql(query, _get_engine())


@st.cache_data(ttl=600)
def load_sentiment_summary() -> pd.DataFrame:
    """Load aggregated sentiment stats by source and speaker."""
    if _USE_PARQUET:
        return pd.read_parquet(_DATA_DIR / "sentiment_summary.parquet")

    query = """
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
    """
    return pd.read_sql(query, _get_engine())
