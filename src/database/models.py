"""
SQLAlchemy Models
------------------
Defines all database tables for the market-analytics project.

Tables:
    - speeches: Raw declarations/statements from central banks and public figures
    - sentiment_scores: NLP analysis results for each speech
    - market_data: Price snapshots around speech timestamps
    - speech_market_correlation: Computed correlations between speeches and market movements
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Speech(Base):
    """
    A speech, press release, or statement from a central bank or public figure.

    The `url` field is used as a natural deduplication key — if a speech with the
    same URL already exists, it won't be inserted again.
    """

    __tablename__ = "speeches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)  # 'ecb', 'fed', 'public_figure'
    speaker = Column(String(200), nullable=False)
    title = Column(String(500))
    content = Column(Text, nullable=False)
    published_at = Column(DateTime, nullable=False)
    url = Column(String(1000))
    ingested_at = Column(DateTime, default=datetime.now)

    # Relationships
    sentiment_scores = relationship("SentimentScore", back_populates="speech", cascade="all, delete")
    correlations = relationship(
        "SpeechMarketCorrelation", back_populates="speech", cascade="all, delete"
    )

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("url", name="uq_speeches_url"),
        Index("idx_speeches_source", "source"),
        Index("idx_speeches_published", "published_at"),
        Index("idx_speeches_speaker", "speaker"),
    )

    def __repr__(self):
        return f"<Speech(id={self.id}, speaker='{self.speaker}', title='{self.title[:50]}...')>"


class SentimentScore(Base):
    """
    Sentiment analysis result for a speech.

    Multiple models can analyze the same speech, so we store
    the model_name to distinguish results (e.g., 'finbert', 'mistral').
    """

    __tablename__ = "sentiment_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    speech_id = Column(Integer, ForeignKey("speeches.id"), nullable=False)
    model_name = Column(String(100), nullable=False)  # 'finbert', 'mistral', etc.
    sentiment = Column(String(20), nullable=False)  # 'positive', 'negative', 'neutral'
    score = Column(Float, nullable=False)  # confidence score (0-1)
    analyzed_at = Column(DateTime, default=datetime.now)

    # Relationships
    speech = relationship("Speech", back_populates="sentiment_scores")

    # Indexes
    __table_args__ = (
        Index("idx_sentiment_speech", "speech_id"),
        Index("idx_sentiment_model", "model_name"),
    )

    def __repr__(self):
        return (
            f"<SentimentScore(speech_id={self.speech_id}, "
            f"model='{self.model_name}', sentiment='{self.sentiment}', score={self.score})>"
        )


class MarketData(Base):
    """
    Market data snapshot around a speech timestamp.

    Captures prices at the time of the speech and at intervals after,
    along with percentage changes.
    """

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)  # 'EUR/USD', 'S&P 500', etc.
    speech_date = Column(DateTime, nullable=False)
    price_at_speech = Column(Float)
    price_1d_after = Column(Float)
    price_1w_after = Column(Float)
    change_1d_pct = Column(Float)
    change_1w_pct = Column(Float)
    recorded_at = Column(DateTime, default=datetime.now)

    # Relationships
    correlations = relationship(
        "SpeechMarketCorrelation", back_populates="market_data", cascade="all, delete"
    )

    # Indexes
    __table_args__ = (
        Index("idx_market_symbol", "symbol"),
        Index("idx_market_speech_date", "speech_date"),
    )

    def __repr__(self):
        return (
            f"<MarketData(symbol='{self.symbol}', "
            f"price={self.price_at_speech}, date={self.speech_date})>"
        )


class SpeechMarketCorrelation(Base):
    """
    Computed correlation between a speech and a market movement.

    Links a speech to a market data snapshot with a correlation score.
    """

    __tablename__ = "speech_market_correlation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    speech_id = Column(Integer, ForeignKey("speeches.id"), nullable=False)
    market_data_id = Column(Integer, ForeignKey("market_data.id"), nullable=False)
    correlation_score = Column(Float)
    computed_at = Column(DateTime, default=datetime.now)

    # Relationships
    speech = relationship("Speech", back_populates="correlations")
    market_data = relationship("MarketData", back_populates="correlations")

    def __repr__(self):
        return (
            f"<SpeechMarketCorrelation(speech_id={self.speech_id}, "
            f"market_data_id={self.market_data_id}, score={self.correlation_score})>"
        )
