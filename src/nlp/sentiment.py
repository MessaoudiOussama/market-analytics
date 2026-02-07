"""
Financial Sentiment Analyzer
------------------------------
Analyzes the sentiment of financial texts using FinBERT,
a BERT model fine-tuned on financial communications.

FinBERT classifies text as: positive, negative, or neutral
with a confidence score (0-1).

Model: ProsusAI/finbert (downloaded automatically on first use)

Since speeches can be very long and BERT models have a 512-token limit,
we split the text into chunks, analyze each chunk, and aggregate results.
"""

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
MAX_CHUNK_LENGTH = 512  # FinBERT max token length


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a text."""

    sentiment: str  # 'positive', 'negative', 'neutral'
    score: float  # confidence score (0-1)
    positive_score: float
    negative_score: float
    neutral_score: float
    num_chunks: int  # how many chunks the text was split into


class FinBERTAnalyzer:
    """
    Financial sentiment analyzer using FinBERT.

    The model is loaded once and reused for all analyses.
    Handles long texts by chunking and aggregating results.
    """

    def __init__(self):
        self.model_name = MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self._loaded = False

    def load_model(self):
        """Load the FinBERT model and tokenizer. Downloads on first use (~400MB)."""
        if self._loaded:
            return

        logger.info(f"Loading FinBERT model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # Use GPU if available, otherwise CPU
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            top_k=None,  # Return all label scores
            truncation=True,  # Safety net for rare edge cases (e.g., "U.S." split issues)
            max_length=MAX_CHUNK_LENGTH,
        )

        self._loaded = True
        device_name = "GPU" if device == 0 else "CPU"
        logger.info(f"FinBERT loaded successfully on {device_name}")

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze the sentiment of a text.

        For long texts, splits into chunks and aggregates scores
        by averaging across all chunks.

        Args:
            text: The text to analyze.

        Returns:
            SentimentResult with aggregated sentiment and scores.
        """
        self.load_model()

        chunks = self._split_into_chunks(text)
        logger.info(f"Analyzing {len(chunks)} chunk(s)")

        # Analyze each chunk
        all_scores = {"positive": [], "negative": [], "neutral": []}

        for chunk in chunks:
            try:
                results = self.pipe(chunk)[0]  # top_k=None returns list of all labels
                for result in results:
                    label = result["label"].lower()
                    if label in all_scores:
                        all_scores[label].append(result["score"])
            except Exception as e:
                logger.warning(f"Failed to analyze chunk: {e}")
                continue

        # Aggregate scores by averaging
        if not any(all_scores.values()):
            return SentimentResult(
                sentiment="neutral",
                score=0.0,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=0.0,
                num_chunks=len(chunks),
            )

        avg_positive = sum(all_scores["positive"]) / max(len(all_scores["positive"]), 1)
        avg_negative = sum(all_scores["negative"]) / max(len(all_scores["negative"]), 1)
        avg_neutral = sum(all_scores["neutral"]) / max(len(all_scores["neutral"]), 1)

        # Determine overall sentiment
        scores = {
            "positive": avg_positive,
            "negative": avg_negative,
            "neutral": avg_neutral,
        }
        overall_sentiment = max(scores, key=scores.get)

        return SentimentResult(
            sentiment=overall_sentiment,
            score=scores[overall_sentiment],
            positive_score=round(avg_positive, 4),
            negative_score=round(avg_negative, 4),
            neutral_score=round(avg_neutral, 4),
            num_chunks=len(chunks),
        )

    def _split_into_chunks(self, text: str) -> list[str]:
        """
        Split text into chunks that fit within FinBERT's token limit (512 tokens).

        Uses exact token counting on the final joined text to prevent any overflow.
        Splits on sentence boundaries to preserve context. If a single sentence
        exceeds the limit, it's split by token position.

        Args:
            text: The full text to split.

        Returns:
            List of text chunks, each guaranteed to be <= 510 tokens.
        """
        max_tokens = MAX_CHUNK_LENGTH - 2  # Reserve 2 for [CLS] and [SEP]

        # Check if text fits in a single chunk
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return [text]

        # Split by sentences
        sentences = text.replace("\n", " ").split(". ")
        chunks = []
        current_chunk = []

        for sentence in sentences:
            # Test if adding this sentence would exceed the limit
            candidate = ". ".join(current_chunk + [sentence])
            candidate_tokens = self.tokenizer.encode(candidate, add_special_tokens=False)

            if len(candidate_tokens) <= max_tokens:
                # Fits — add sentence to current chunk
                current_chunk.append(sentence)
            else:
                # Doesn't fit — save current chunk and start new one
                if current_chunk:
                    chunks.append(". ".join(current_chunk) + ".")

                # Check if this single sentence fits on its own
                sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
                if len(sentence_tokens) <= max_tokens:
                    current_chunk = [sentence]
                else:
                    # Single sentence too long — split by tokens directly
                    for i in range(0, len(sentence_tokens), max_tokens):
                        token_slice = sentence_tokens[i:i + max_tokens]
                        chunk_text = self.tokenizer.decode(token_slice, skip_special_tokens=True)
                        chunks.append(chunk_text)
                    current_chunk = []

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(". ".join(current_chunk))

        return chunks if chunks else [text[:2000]]


# Module-level instance for convenience
_analyzer = None


def get_analyzer() -> FinBERTAnalyzer:
    """Get or create the singleton FinBERT analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FinBERTAnalyzer()
    return _analyzer


def analyze_text(text: str) -> SentimentResult:
    """
    Convenience function to analyze text sentiment.

    Args:
        text: The text to analyze.

    Returns:
        SentimentResult with sentiment and scores.
    """
    return get_analyzer().analyze(text)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    test_texts = [
        "The ECB has decided to raise interest rates by 25 basis points to combat inflation.",
        "Economic growth remains strong and unemployment is at historic lows.",
        "We are concerned about the rising risks of a recession in the eurozone.",
        "The committee decided to maintain the current policy stance unchanged.",
    ]

    analyzer = FinBERTAnalyzer()
    analyzer.load_model()

    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text[:80]}...")
        print(f"  Sentiment: {result.sentiment} (score: {result.score:.4f})")
        print(f"  Positive: {result.positive_score:.4f}")
        print(f"  Negative: {result.negative_score:.4f}")
        print(f"  Neutral:  {result.neutral_score:.4f}")
