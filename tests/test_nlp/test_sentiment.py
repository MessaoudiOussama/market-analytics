"""
Tests for the sentiment analysis module.
Validates chunking logic and result structure without requiring the full model.
"""

from nlp.sentiment import FinBERTAnalyzer, SentimentResult


class TestSentimentResult:
    """Test the SentimentResult dataclass."""

    def test_sentiment_result_creation(self):
        result = SentimentResult(
            sentiment="positive",
            score=0.85,
            positive_score=0.85,
            negative_score=0.10,
            neutral_score=0.05,
            num_chunks=3,
        )
        assert result.sentiment == "positive"
        assert result.score == 0.85
        assert result.num_chunks == 3

    def test_sentiment_result_neutral(self):
        result = SentimentResult(
            sentiment="neutral",
            score=0.0,
            positive_score=0.0,
            negative_score=0.0,
            neutral_score=0.0,
            num_chunks=0,
        )
        assert result.sentiment == "neutral"
        assert result.score == 0.0


class TestFinBERTAnalyzerInit:
    """Test FinBERTAnalyzer initialization (no model loading)."""

    def test_analyzer_creation(self):
        analyzer = FinBERTAnalyzer()
        assert analyzer.model_name == "ProsusAI/finbert"
        assert analyzer._loaded is False
        assert analyzer.tokenizer is None
        assert analyzer.model is None
        assert analyzer.pipe is None
