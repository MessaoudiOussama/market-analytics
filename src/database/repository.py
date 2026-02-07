"""
Data Repository
----------------
Handles all database operations: inserting, querying, and deduplication.

Provides a clean interface between the scrapers/processors and the database.
"""

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.connection import get_session
from database.models import MarketData, SentimentScore, Speech, SpeechMarketCorrelation

logger = logging.getLogger(__name__)


# ── Speech Operations ─────────────────────────────────────


def store_speech(
    source: str,
    speaker: str,
    title: str,
    content: str,
    published_at: datetime,
    url: str,
    session: Session | None = None,
) -> Speech | None:
    """
    Store a speech in the database. Skips duplicates based on URL.

    Args:
        source: Source identifier ('ecb', 'fed', etc.)
        speaker: Speaker name.
        title: Speech title.
        content: Full text content.
        published_at: Publication datetime.
        url: Original URL (used for deduplication).
        session: Optional existing session. If None, creates a new one.

    Returns:
        The Speech object if inserted, None if duplicate.
    """

    def _store(s: Session) -> Speech | None:
        # Check for duplicate
        existing = s.execute(select(Speech).where(Speech.url == url)).scalar_one_or_none()

        if existing:
            logger.info(f"Skipping duplicate speech: {title}")
            return None

        speech = Speech(
            source=source,
            speaker=speaker,
            title=title,
            content=content,
            published_at=published_at,
            url=url,
        )
        s.add(speech)
        s.flush()  # Get the ID without committing
        logger.info(f"Stored speech: {title} by {speaker}")
        return speech

    if session:
        return _store(session)

    with get_session() as s:
        return _store(s)


def store_speeches_bulk(speeches: list[dict]) -> int:
    """
    Store multiple speeches at once. Skips duplicates.

    Args:
        speeches: List of dicts with keys: source, speaker, title, content, published_at, url.

    Returns:
        Number of new speeches inserted.
    """
    inserted = 0

    with get_session() as session:
        for speech_data in speeches:
            result = store_speech(session=session, **speech_data)
            if result:
                inserted += 1

    logger.info(f"Bulk insert: {inserted}/{len(speeches)} new speeches stored")
    return inserted


def get_speeches(
    source: str | None = None,
    speaker: str | None = None,
    limit: int = 100,
) -> list[Speech]:
    """
    Query speeches with optional filters.

    Args:
        source: Filter by source ('ecb', 'fed', etc.)
        speaker: Filter by speaker name (partial match).
        limit: Maximum number of results.

    Returns:
        List of Speech objects.
    """
    with get_session() as session:
        query = select(Speech).order_by(Speech.published_at.desc()).limit(limit)

        if source:
            query = query.where(Speech.source == source)
        if speaker:
            query = query.where(Speech.speaker.ilike(f"%{speaker}%"))

        result = session.execute(query).scalars().all()
        return result


def get_unanalyzed_speeches(model_name: str) -> list[Speech]:
    """
    Get speeches that haven't been analyzed by a specific NLP model yet.

    Args:
        model_name: The NLP model name (e.g., 'finbert').

    Returns:
        List of Speech objects without sentiment scores for the given model.
    """
    with get_session() as session:
        analyzed_ids = (
            select(SentimentScore.speech_id).where(SentimentScore.model_name == model_name)
        )

        query = select(Speech).where(Speech.id.not_in(analyzed_ids))
        result = session.execute(query).scalars().all()
        return result


# ── Sentiment Operations ──────────────────────────────────


def store_sentiment(
    speech_id: int,
    model_name: str,
    sentiment: str,
    score: float,
    session: Session | None = None,
) -> SentimentScore:
    """
    Store a sentiment analysis result.

    Args:
        speech_id: ID of the analyzed speech.
        model_name: NLP model used (e.g., 'finbert').
        sentiment: Sentiment label ('positive', 'negative', 'neutral').
        score: Confidence score (0-1).
        session: Optional existing session.

    Returns:
        The SentimentScore object.
    """

    def _store(s: Session) -> SentimentScore:
        sentiment_score = SentimentScore(
            speech_id=speech_id,
            model_name=model_name,
            sentiment=sentiment,
            score=score,
        )
        s.add(sentiment_score)
        s.flush()
        return sentiment_score

    if session:
        return _store(session)

    with get_session() as s:
        return _store(s)


# ── Market Data Operations ────────────────────────────────


def store_market_snapshot(
    symbol: str,
    speech_date: datetime,
    price_at_speech: float | None,
    price_1d_after: float | None = None,
    price_1w_after: float | None = None,
    change_1d_pct: float | None = None,
    change_1w_pct: float | None = None,
    session: Session | None = None,
) -> MarketData:
    """
    Store a market data snapshot.

    Args:
        symbol: Market symbol name (e.g., 'EUR/USD').
        speech_date: Date of the related speech.
        price_at_speech: Price at speech time.
        price_1d_after: Price 1 day after.
        price_1w_after: Price 1 week after.
        change_1d_pct: Percentage change after 1 day.
        change_1w_pct: Percentage change after 1 week.
        session: Optional existing session.

    Returns:
        The MarketData object.
    """

    def _store(s: Session) -> MarketData:
        market = MarketData(
            symbol=symbol,
            speech_date=speech_date,
            price_at_speech=price_at_speech,
            price_1d_after=price_1d_after,
            price_1w_after=price_1w_after,
            change_1d_pct=change_1d_pct,
            change_1w_pct=change_1w_pct,
        )
        s.add(market)
        s.flush()
        return market

    if session:
        return _store(session)

    with get_session() as s:
        return _store(s)


# ── Correlation Operations ────────────────────────────────


def store_correlation(
    speech_id: int,
    market_data_id: int,
    correlation_score: float,
    session: Session | None = None,
) -> SpeechMarketCorrelation:
    """
    Store a speech-market correlation result.

    Args:
        speech_id: ID of the speech.
        market_data_id: ID of the market data snapshot.
        correlation_score: Computed correlation score.
        session: Optional existing session.

    Returns:
        The SpeechMarketCorrelation object.
    """

    def _store(s: Session) -> SpeechMarketCorrelation:
        correlation = SpeechMarketCorrelation(
            speech_id=speech_id,
            market_data_id=market_data_id,
            correlation_score=correlation_score,
        )
        s.add(correlation)
        s.flush()
        return correlation

    if session:
        return _store(session)

    with get_session() as s:
        return _store(s)
