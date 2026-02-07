"""
Database Connection Manager
----------------------------
Handles database connection, session management, and table creation.

Usage:
    from database.connection import get_session, init_db

    # Create all tables
    init_db()

    # Use a session
    with get_session() as session:
        session.add(some_object)
"""

import logging
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base

logger = logging.getLogger(__name__)

# Database URL — defaults to local Docker PostgreSQL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://airflow:airflow@localhost:5432/market_analytics",
)

# Create engine
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# Session factory
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session() -> Session:
    """
    Context manager that provides a database session.

    Automatically commits on success and rolls back on error.

    Usage:
        with get_session() as session:
            session.add(speech)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """
    Create all tables defined in models.py.

    Safe to call multiple times — only creates tables that don't exist yet.
    """
    logger.info(f"Initializing database at {DATABASE_URL}")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def drop_db():
    """
    Drop all tables. Use with caution — only for development/testing.
    """
    logger.warning("Dropping all database tables!")
    Base.metadata.drop_all(bind=engine)
    logger.info("All tables dropped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Database initialized successfully!")
