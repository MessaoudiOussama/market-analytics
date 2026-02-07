"""
NLP Analysis Pipeline Runner
------------------------------
Processes all speeches that haven't been analyzed yet:
    1. Fetches unanalyzed speeches from the database
    2. Runs FinBERT sentiment analysis on each
    3. Stores results back in the database

Usage:
    python -m nlp.run_analysis
"""

import logging
import sys

from database.connection import init_db
from database.repository import get_unanalyzed_speeches, store_sentiment
from nlp.sentiment import FinBERTAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "finbert"


def run_sentiment_analysis() -> int:
    """
    Run FinBERT sentiment analysis on all unanalyzed speeches.

    Returns:
        Number of speeches analyzed.
    """
    logger.info("=" * 60)
    logger.info("Starting sentiment analysis pipeline")
    logger.info("=" * 60)

    # Get speeches not yet analyzed by FinBERT
    speeches = get_unanalyzed_speeches(MODEL_NAME)
    logger.info(f"Found {len(speeches)} speeches to analyze")

    if not speeches:
        logger.info("No new speeches to analyze")
        return 0

    # Load model once
    analyzer = FinBERTAnalyzer()
    analyzer.load_model()

    analyzed = 0
    for i, speech in enumerate(speeches, 1):
        logger.info(f"[{i}/{len(speeches)}] Analyzing: {speech.title[:60]}...")

        try:
            result = analyzer.analyze(speech.content)

            store_sentiment(
                speech_id=speech.id,
                model_name=MODEL_NAME,
                sentiment=result.sentiment,
                score=result.score,
            )

            logger.info(
                f"  Result: {result.sentiment} "
                f"(pos={result.positive_score:.3f}, "
                f"neg={result.negative_score:.3f}, "
                f"neu={result.neutral_score:.3f}) "
                f"[{result.num_chunks} chunks]"
            )
            analyzed += 1

        except Exception as e:
            logger.error(f"  Failed to analyze speech {speech.id}: {e}")
            continue

    logger.info("=" * 60)
    logger.info(f"Analysis complete: {analyzed}/{len(speeches)} speeches analyzed")
    logger.info("=" * 60)
    return analyzed


def main():
    """Run the full NLP analysis pipeline."""
    init_db()
    run_sentiment_analysis()


if __name__ == "__main__":
    main()
