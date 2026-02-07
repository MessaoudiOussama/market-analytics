"""
Speaker Analysis Page
----------------------
Drill down into individual speakers: sentiment patterns,
speech frequency, and how their tone has evolved over time.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_speeches

st.set_page_config(page_title="Speaker Analysis", page_icon="🎤", layout="wide")

st.title("Speaker Analysis")
st.markdown("*Analyzing sentiment patterns by speaker*")
st.markdown("---")

# ── Load Data ─────────────────────────────────────────────
speeches = load_speeches()

if speeches.empty:
    st.warning("No data available. Run the pipeline first.")
    st.stop()

color_map = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
}

# ── Speaker Selector ──────────────────────────────────────
speakers = sorted(speeches["speaker"].unique())
selected_speaker = st.selectbox("Select a speaker", options=["All Speakers"] + speakers)

if selected_speaker != "All Speakers":
    filtered = speeches[speeches["speaker"] == selected_speaker]
else:
    filtered = speeches

# ── Speaker KPIs ──────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Speeches", len(filtered))

with col2:
    if "sentiment" in filtered.columns:
        most_common = filtered["sentiment"].mode()
        dominant = most_common.iloc[0] if not most_common.empty else "N/A"
        st.metric("Dominant Sentiment", dominant.capitalize())

with col3:
    if "sentiment_score" in filtered.columns:
        avg_score = filtered["sentiment_score"].mean()
        st.metric("Avg Confidence", f"{avg_score:.3f}" if pd.notna(avg_score) else "N/A")

with col4:
    if "source" in filtered.columns:
        sources = filtered["source"].nunique()
        st.metric("Sources", sources)

st.markdown("---")

# ── Sentiment Breakdown per Speaker ───────────────────────
st.subheader("Sentiment Breakdown by Speaker")

if "sentiment" in speeches.columns:
    speaker_sentiment = (
        speeches.groupby(["speaker", "sentiment"]).size().reset_index(name="count")
    )

    # Calculate total per speaker for sorting
    speaker_totals = speaker_sentiment.groupby("speaker")["count"].sum().reset_index()
    speaker_totals.columns = ["speaker", "total"]
    speaker_sentiment = speaker_sentiment.merge(speaker_totals, on="speaker")
    speaker_sentiment = speaker_sentiment.sort_values("total", ascending=True)

    fig = px.bar(
        speaker_sentiment,
        x="count",
        y="speaker",
        color="sentiment",
        color_discrete_map=color_map,
        orientation="h",
        title="Speech Count & Sentiment by Speaker",
        labels={"count": "Number of Speeches", "speaker": "Speaker"},
    )
    fig.update_layout(height=max(400, len(speakers) * 40))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Speaker Sentiment Over Time ───────────────────────────
st.subheader("Sentiment Over Time")

if selected_speaker != "All Speakers" and not filtered.empty:
    filtered_with_date = filtered.dropna(subset=["sentiment", "published_at"]).copy()
    filtered_with_date["published_at"] = pd.to_datetime(filtered_with_date["published_at"])

    if not filtered_with_date.empty:
        fig_time = px.scatter(
            filtered_with_date,
            x="published_at",
            y="sentiment_score",
            color="sentiment",
            color_discrete_map=color_map,
            size="sentiment_score",
            hover_data=["title"],
            title=f"Sentiment Timeline — {selected_speaker}",
            labels={
                "published_at": "Date",
                "sentiment_score": "Confidence Score",
            },
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Not enough data points for this speaker.")
else:
    # Show all speakers over time
    all_with_date = speeches.dropna(subset=["sentiment", "published_at"]).copy()
    all_with_date["published_at"] = pd.to_datetime(all_with_date["published_at"])

    if not all_with_date.empty:
        fig_all = px.scatter(
            all_with_date,
            x="published_at",
            y="sentiment_score",
            color="speaker",
            symbol="sentiment",
            hover_data=["title", "source"],
            title="All Speakers — Sentiment Timeline",
            labels={
                "published_at": "Date",
                "sentiment_score": "Confidence Score",
            },
        )
        st.plotly_chart(fig_all, use_container_width=True)

st.markdown("---")

# ── Speaker Comparison Radar ──────────────────────────────
st.subheader("Speaker Sentiment Comparison")

if "sentiment" in speeches.columns and "sentiment_score" in speeches.columns:
    # Compute avg scores per speaker per sentiment
    pivot = (
        speeches.dropna(subset=["sentiment", "sentiment_score"])
        .groupby(["speaker", "sentiment"])["sentiment_score"]
        .mean()
        .reset_index()
        .pivot(index="speaker", columns="sentiment", values="sentiment_score")
        .fillna(0)
    )

    if not pivot.empty:
        st.dataframe(
            pivot.style.format("{:.3f}").background_gradient(cmap="RdYlGn", axis=None),
            use_container_width=True,
        )

st.markdown("---")

# ── Speeches Detail Table ─────────────────────────────────
st.subheader("Speech Details")

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
