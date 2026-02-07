# Market Voice Analytics

**Analyzing the market impact of central bank communications using NLP**

An end-to-end data pipeline that scrapes speeches from the **ECB** and **Federal Reserve**, performs **financial sentiment analysis** using FinBERT, and correlates the results with **market movements** (EUR/USD, S&P 500, Gold, Treasuries, Euro Stoxx 50).

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Airflow](https://img.shields.io/badge/Apache_Airflow-2.9-017CEE?logo=apacheairflow)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FinBERT-FFD21E?logo=huggingface)

---

## Architecture

```
                    ECB RSS Feed              Fed JSON / RSS
                         |                          |
                         v                          v
                  +--------------+          +--------------+
                  | ECB Scraper  |          | Fed Scraper  |
                  +--------------+          +--------------+
                         |                          |
                         +----------+  +------------+
                                    |  |
                                    v  v
                            +----------------+
                            |  PostgreSQL     |
                            |  (Docker)       |
                            +----------------+
                                    |
                    +---------------+----------------+
                    |                                |
                    v                                v
            +---------------+              +-----------------+
            | FinBERT NLP   |              | Market Data     |
            | (Sentiment)   |              | (yfinance)      |
            +---------------+              +-----------------+
                    |                                |
                    +---------------+----------------+
                                    |
                                    v
                          +------------------+
                          | Streamlit        |
                          | Dashboard        |
                          +------------------+
```

## Features

- **Data Ingestion**: Automated scrapers for ECB and Federal Reserve speeches, press releases, and statements
- **Historical Backfill**: Archive scrapers that can fetch thousands of historical speeches (ECB foedb JSON database + Fed yearly archives)
- **Financial NLP**: Sentiment analysis using [FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned on financial text. Handles long documents via intelligent sentence-based chunking
- **Market Data**: Automatic fetching of price data (EUR/USD, S&P 500, US 10Y Treasury, Gold, Euro Stoxx 50) around speech dates using Yahoo Finance
- **Correlation Analysis**: Measures how speech sentiment correlates with market movements (1-day and 1-week changes)
- **Interactive Dashboard**: 3-page Streamlit app with sentiment distribution, speaker analysis, and market impact visualizations
- **Deduplication**: URL-based deduplication ensures no duplicate speeches are stored
- **Dockerized**: Full stack runs with `docker compose up`

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **Orchestration** | Apache Airflow 2.9 |
| **Database** | PostgreSQL 16 |
| **ORM** | SQLAlchemy 2.0 |
| **NLP** | HuggingFace Transformers, FinBERT |
| **Market Data** | yfinance |
| **Dashboard** | Streamlit, Plotly |
| **Infrastructure** | Docker, Docker Compose |
| **Dependency Management** | Poetry |
| **Linting** | Ruff |
| **Testing** | pytest |

## Project Structure

```
market-analytics/
├── docker-compose.yml          # Airflow + PostgreSQL
├── pyproject.toml              # Poetry config + tool settings
│
├── dags/                       # Airflow DAGs
│
├── src/
│   ├── ingestion/              # Data collection
│   │   ├── ecb_scraper.py      # ECB RSS feed scraper
│   │   ├── fed_scraper.py      # Fed JSON/RSS scraper
│   │   ├── ecb_backfill.py     # ECB historical archive (foedb)
│   │   ├── fed_backfill.py     # Fed yearly archive scraper
│   │   ├── market_data.py      # Yahoo Finance market data
│   │   ├── run_ingestion.py    # Daily pipeline runner
│   │   └── run_backfill.py     # Historical backfill runner
│   │
│   ├── nlp/                    # NLP processing
│   │   ├── sentiment.py        # FinBERT analyzer with chunking
│   │   └── run_analysis.py     # Sentiment pipeline runner
│   │
│   ├── analytics/              # Correlation engine
│   │
│   └── database/               # Data layer
│       ├── models.py           # SQLAlchemy models
│       ├── connection.py       # DB connection manager
│       └── repository.py       # CRUD operations
│
├── streamlit_app/              # Dashboard
│   ├── app.py                  # Overview page
│   ├── data_loader.py          # Cached data queries
│   └── pages/
│       ├── 1_Speaker_Analysis.py
│       └── 2_Market_Impact.py
│
├── tests/                      # Unit tests
├── notebooks/                  # Exploratory analysis
└── config/
    └── init_db.sql             # Database initialization
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Poetry

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/market-analytics.git
cd market-analytics
poetry install
```

### 2. Start infrastructure

```bash
docker compose up -d
```

This starts PostgreSQL and Apache Airflow. Wait ~15 seconds for initialization.

- **Airflow UI**: http://localhost:8080 (admin/admin)
- **PostgreSQL**: localhost:5432

### 3. Run the pipeline

```bash
cd src

# Daily ingestion (latest speeches + market data + NLP)
python -m ingestion.run_ingestion

# Historical backfill (200 ECB + 2 years Fed)
python -m ingestion.run_backfill --ecb-max 200 --fed-years 2024 2025
```

### 4. Launch the dashboard

```bash
streamlit run streamlit_app/app.py
```

Open http://localhost:8501

## Dashboard Pages

### Overview
KPI cards, sentiment distribution, sentiment over time, filterable speech table.

### Speaker Analysis
Per-speaker drill down, sentiment breakdown, speaker comparison heatmap.

### Market Impact
Sentiment vs market changes, scatter plots with trendlines, box plots, speaker impact ranking — showing which speakers move markets the most.

## Data Sources

| Source | Type | Method |
|---|---|---|
| [ECB](https://www.ecb.europa.eu) | Speeches, press releases | RSS feed + foedb JSON archive |
| [Federal Reserve](https://www.federalreserve.gov) | Speeches, statements | JSON endpoint + yearly HTML archives |
| [Yahoo Finance](https://finance.yahoo.com) | Market prices | yfinance API |

## NLP Approach

**Model**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) — a BERT model fine-tuned on financial text (10K filings, analyst reports, financial news).

**Challenge**: Central bank speeches are typically 2,000-5,000 words, but BERT models have a 512-token limit.

**Solution**: Intelligent sentence-based chunking that:
1. Splits text on sentence boundaries
2. Verifies exact token count for each chunk
3. Handles edge cases (abbreviations, long sentences)
4. Aggregates sentiment scores across all chunks

## License

MIT
