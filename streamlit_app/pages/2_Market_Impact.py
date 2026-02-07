"""
Market Impact Page
-------------------
Analyzes the correlation between speech sentiment and market movements.
Shows how markets react to positive, negative, and neutral communications.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_speeches_with_market, load_market_data

st.set_page_config(page_title="Market Impact", page_icon="📈", layout="wide")

st.title("Market Impact Analysis")
st.markdown("*Correlating speech sentiment with market reactions*")
st.markdown("---")

# ── Load Data ─────────────────────────────────────────────
try:
    combined = load_speeches_with_market()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    combined = pd.DataFrame()

market_data = load_market_data()

if combined.empty:
    st.warning("No combined speech + market data available. Make sure both pipelines have run.")
    st.stop()

color_map = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
}

# ── Filters ───────────────────────────────────────────────
col_f1, col_f2 = st.columns(2)

with col_f1:
    symbols = sorted(combined["symbol"].unique())
    selected_symbol = st.selectbox("Select market", options=symbols)

with col_f2:
    sources = sorted(combined["source"].unique())
    selected_sources = st.multiselect("Filter by source", options=sources, default=sources)

# Apply filters
df = combined[
    (combined["symbol"] == selected_symbol) & (combined["source"].isin(selected_sources))
].copy()

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Speeches Analyzed", len(df))

with col2:
    avg_1d = df["change_1d_pct"].mean()
    st.metric(
        "Avg 1-Day Change",
        f"{avg_1d:+.3f}%" if pd.notna(avg_1d) else "N/A",
    )

with col3:
    avg_1w = df["change_1w_pct"].mean()
    st.metric(
        "Avg 1-Week Change",
        f"{avg_1w:+.3f}%" if pd.notna(avg_1w) else "N/A",
    )

with col4:
    st.metric("Market", selected_symbol)

st.markdown("---")

# ── Market Reaction by Sentiment ──────────────────────────
st.subheader("Market Reaction by Sentiment")

col_left, col_right = st.columns(2)

with col_left:
    # 1-Day change by sentiment
    sentiment_1d = (
        df.dropna(subset=["change_1d_pct"])
        .groupby("sentiment")["change_1d_pct"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    if not sentiment_1d.empty:
        fig_1d = px.bar(
            sentiment_1d,
            x="sentiment",
            y="mean",
            color="sentiment",
            color_discrete_map=color_map,
            title=f"{selected_symbol} — Avg 1-Day Change by Sentiment",
            labels={"mean": "Avg Change (%)", "sentiment": "Sentiment"},
            text="mean",
        )
        fig_1d.update_traces(texttemplate="%{text:.3f}%", textposition="outside")
        st.plotly_chart(fig_1d, use_container_width=True)

with col_right:
    # 1-Week change by sentiment
    sentiment_1w = (
        df.dropna(subset=["change_1w_pct"])
        .groupby("sentiment")["change_1w_pct"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    if not sentiment_1w.empty:
        fig_1w = px.bar(
            sentiment_1w,
            x="sentiment",
            y="mean",
            color="sentiment",
            color_discrete_map=color_map,
            title=f"{selected_symbol} — Avg 1-Week Change by Sentiment",
            labels={"mean": "Avg Change (%)", "sentiment": "Sentiment"},
            text="mean",
        )
        fig_1w.update_traces(texttemplate="%{text:.3f}%", textposition="outside")
        st.plotly_chart(fig_1w, use_container_width=True)

st.markdown("---")

# ── Scatter: Sentiment Score vs Market Change ─────────────
st.subheader("Sentiment Score vs Market Change")

df_scatter = df.dropna(subset=["sentiment_score", "change_1d_pct"])

if not df_scatter.empty:
    fig_scatter = px.scatter(
        df_scatter,
        x="sentiment_score",
        y="change_1d_pct",
        color="sentiment",
        color_discrete_map=color_map,
        size="sentiment_score",
        hover_data=["speaker", "title", "published_at"],
        title=f"{selected_symbol} — Sentiment Confidence vs 1-Day Price Change",
        labels={
            "sentiment_score": "Sentiment Confidence Score",
            "change_1d_pct": "1-Day Price Change (%)",
        },
        trendline="ols",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Not enough data for scatter plot.")

st.markdown("---")

# ── Box Plot: Distribution of Changes by Sentiment ───────
st.subheader("Distribution of Market Changes by Sentiment")

df_box = df.dropna(subset=["change_1d_pct"])

if not df_box.empty:
    fig_box = px.box(
        df_box,
        x="sentiment",
        y="change_1d_pct",
        color="sentiment",
        color_discrete_map=color_map,
        title=f"{selected_symbol} — 1-Day Change Distribution by Sentiment",
        labels={
            "sentiment": "Speech Sentiment",
            "change_1d_pct": "1-Day Price Change (%)",
        },
        points="all",
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# ── Speaker Impact Ranking ────────────────────────────────
st.subheader("Speaker Impact Ranking")
st.markdown("*Which speakers move markets the most?*")

speaker_impact = (
    df.dropna(subset=["change_1d_pct"])
    .groupby("speaker")
    .agg(
        speeches=("change_1d_pct", "count"),
        avg_1d_change=("change_1d_pct", "mean"),
        avg_1w_change=("change_1w_pct", "mean"),
        avg_abs_1d_change=("change_1d_pct", lambda x: x.abs().mean()),
    )
    .reset_index()
    .sort_values("avg_abs_1d_change", ascending=False)
)

if not speaker_impact.empty:
    fig_impact = px.bar(
        speaker_impact,
        x="speaker",
        y="avg_abs_1d_change",
        color="avg_1d_change",
        color_continuous_scale="RdYlGn",
        title=f"{selected_symbol} — Average Absolute 1-Day Change by Speaker",
        labels={
            "speaker": "Speaker",
            "avg_abs_1d_change": "Avg Absolute 1-Day Change (%)",
            "avg_1d_change": "Direction",
        },
    )
    st.plotly_chart(fig_impact, use_container_width=True)

    # Table view
    st.dataframe(
        speaker_impact.rename(
            columns={
                "speaker": "Speaker",
                "speeches": "Speeches",
                "avg_1d_change": "Avg 1D Change (%)",
                "avg_1w_change": "Avg 1W Change (%)",
                "avg_abs_1d_change": "Avg |1D Change| (%)",
            }
        ).style.format(
            {
                "Avg 1D Change (%)": "{:+.3f}",
                "Avg 1W Change (%)": "{:+.3f}",
                "Avg |1D Change| (%)": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")

# ── Raw Data ──────────────────────────────────────────────
with st.expander("View raw data"):
    display_cols = [
        "published_at", "source", "speaker", "title", "sentiment",
        "sentiment_score", "symbol", "price_at_speech", "change_1d_pct", "change_1w_pct",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True, hide_index=True)
