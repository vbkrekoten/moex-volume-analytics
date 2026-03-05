# MOEX Volume Analytics Dashboard

Interactive dashboard for analyzing dependencies between MOEX trading volume across instrument classes and market/macro factors.

## Features

- **Trading volume by instrument class**: stocks, bonds, futures, options, currency, repo
- **9 analytical factors**: volatility (RVI), trend strength (ADX), trend direction, IMOEX level, market cap, CPI inflation, USD/RUB, CNY/RUB, M2 money supply
- **Correlation analysis**: Pearson/Spearman heatmaps, rolling correlations, scatter plots
- **Regression analysis**: OLS factor decomposition with significance testing
- **Interactive filters**: select instruments, periods, show/hide factors
- **Weekly granularity** from 2018 to present

## Data Sources

| Data | Source | Authentication |
|------|--------|---------------|
| Trading volumes | [MOEX ISS API](https://iss.moex.com/iss/reference/) | Free, no auth |
| Indices (IMOEX, RGBI, RVI) | MOEX ISS API | Free, no auth |
| Market capitalization | MOEX ISS API | Free, no auth |
| USD/RUB, CNY/RUB | [CBR XML API](https://www.cbr.ru/development/) | Free, no auth |
| Key rate, M2 | CBR SOAP API | Free, no auth |
| CPI inflation | Bundled CSV (Rosstat) | Local file |

## Tech Stack

- **Python 3.11+** with Streamlit
- **Plotly** for interactive charts
- **Supabase** (PostgreSQL) for data storage
- **pandas / scipy / statsmodels** for analytics
- **Streamlit Community Cloud** for deployment

## Quick Start

```bash
# Clone
git clone https://github.com/vbkrekoten/moex-volume-analytics.git
cd moex-volume-analytics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Supabase credentials

# Run
streamlit run app.py
```

## Docker

```bash
docker-compose up --build
```

Open http://localhost:8501

## Data Pipeline

On first launch, click "Обновить данные" on the "Данные" tab to load historical data (~10-15 min for 2018-present).

## License

MIT
