"""
Market Voice Analytics — Streamlit Dashboard
----------------------------------------------
Main entry point for the dashboard.
Displays an overview of all collected data and sentiment analysis results.

Run with:
    streamlit run streamlit_app/app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import load_market_data, load_sentiment_summary, load_speeches

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Market Voice Analytics - V1.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("Market Voice Analytics")
st.sidebar.markdown(
    "Analyzing the market impact of central bank communications using NLP."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources**")
st.sidebar.markdown("- ECB (European Central Bank)")
st.sidebar.markdown("- Federal Reserve")
st.sidebar.markdown("---")
st.sidebar.markdown("**NLP Model:** FinBERT")

# ── Load Data ─────────────────────────────────────────────
speeches = load_speeches()
market_data = load_market_data()
sentiment_summary = load_sentiment_summary()

# ── Header ────────────────────────────────────────────────
st.title("Market Voice Analytics")
st.markdown("*How central bank communications correlate with market movements*")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Speeches", value=len(speeches))

with col2:
    sources = speeches["source"].nunique() if not speeches.empty else 0
    st.metric(label="Data Sources", value=sources)

with col3:
    speakers = speeches["speaker"].nunique() if not speeches.empty else 0
    st.metric(label="Unique Speakers", value=speakers)

with col4:
    market_symbols = market_data["symbol"].nunique() if not market_data.empty else 0
    st.metric(label="Markets Tracked", value=market_symbols)

st.markdown("---")

# ── Sentiment Distribution ────────────────────────────────
st.subheader("Sentiment Distribution")

if not speeches.empty and "sentiment" in speeches.columns:
    col_left, col_right = st.columns(2)

    with col_left:
        # Pie chart of sentiment distribution
        sentiment_counts = speeches["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]

        color_map = {
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#95a5a6",
        }

        fig_pie = px.pie(
            sentiment_counts,
            values="count",
            names="sentiment",
            color="sentiment",
            color_discrete_map=color_map,
            title="Overall Sentiment Breakdown",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        # Sentiment by source
        source_sentiment = (
            speeches.groupby(["source", "sentiment"]).size().reset_index(name="count")
        )

        fig_bar = px.bar(
            source_sentiment,
            x="source",
            y="count",
            color="sentiment",
            color_discrete_map=color_map,
            barmode="group",
            title="Sentiment by Source",
            labels={"source": "Source", "count": "Number of Speeches"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("No sentiment data available yet. Run the NLP pipeline first.")

st.markdown("---")

# ── Sentiment Over Time ──────────────────────────────────
st.subheader("Sentiment Over Time")

if not speeches.empty and "sentiment" in speeches.columns and "published_at" in speeches.columns:
    speeches_with_date = speeches.dropna(subset=["sentiment", "published_at"]).copy()
    speeches_with_date["published_at"] = pd.to_datetime(speeches_with_date["published_at"])

    # Map sentiment to numeric for timeline
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    speeches_with_date["sentiment_numeric"] = speeches_with_date["sentiment"].map(sentiment_map)

    fig_timeline = px.scatter(
        speeches_with_date,
        x="published_at",
        y="sentiment_score",
        color="sentiment",
        color_discrete_map=color_map,
        size="sentiment_score",
        hover_data=["speaker", "title", "source"],
        title="Speech Sentiment Scores Over Time",
        labels={
            "published_at": "Date",
            "sentiment_score": "Confidence Score",
            "sentiment": "Sentiment",
        },
    )
    fig_timeline.update_layout(xaxis_title="Date", yaxis_title="Confidence Score")
    st.plotly_chart(fig_timeline, use_container_width=True)

st.markdown("---")

# ── Recent Speeches Table ─────────────────────────────────
st.subheader("Recent Speeches")

if not speeches.empty:
    # Display filters
    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:
        source_filter = st.multiselect(
            "Filter by source",
            options=speeches["source"].unique(),
            default=speeches["source"].unique(),
        )

    with col_filter2:
        sentiment_filter = st.multiselect(
            "Filter by sentiment",
            options=[s for s in speeches["sentiment"].unique() if pd.notna(s)],
            default=[s for s in speeches["sentiment"].unique() if pd.notna(s)],
        )

    # Apply filters
    filtered = speeches[
        (speeches["source"].isin(source_filter))
        & (speeches["sentiment"].isin(sentiment_filter))
    ]

    # Display table
    display_cols = ["published_at", "source", "speaker", "title", "sentiment", "sentiment_score"]
    available_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[available_cols].rename(
            columns={
                "published_at": "Date",
                "source": "Source",
                "speaker": "Speaker",
                "title": "Title",
                "sentiment": "Sentiment",
                "sentiment_score": "Score",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No speeches collected yet. Run the ingestion pipeline first.")
