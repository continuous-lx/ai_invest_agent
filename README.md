# AI Invest Agent

AI Invest Agent is a Python-based investment research workflow that combines
market data, financial news, local knowledge retrieval, AI analysis, portfolio
risk guidance, and basic backtesting into one command-line tool.

The project is designed for single-stock research. Given a ticker symbol, it
collects market context, retrieves relevant local documents, analyzes recent
news and price action, runs a buy-and-hold backtest, calculates risk metrics,
and generates a Markdown investment report.

## Core Workflow

```text
Ticker input
  -> Market data and news collection
  -> Local RAG knowledge retrieval
  -> Researcher agent
  -> Analyst agent
  -> Portfolio manager agent
  -> Backtest metrics
  -> Markdown report
```

## Features

- Interactive ticker input from the command line
- Market data and price history via `yfinance`
- Recent financial news retrieval
- Local RAG knowledge base with strict ticker metadata filtering
- Researcher, analyst, and portfolio manager agents using CrewAI
- Rules-based investment scoring with `BUY`, `HOLD`, or `SELL` guidance
- Buy-and-hold backtesting
- Sharpe Ratio, annualized volatility, cumulative return, and max drawdown
- Markdown report generation under `reports/`

## Project Structure

```text
agents/
  researcher.py          # Collects market, news, and RAG context
  analyst.py             # Investment scoring and AI analyst task
  portfolio_manager.py   # Position sizing and portfolio risk guidance

backtests/
  metrics.py             # Sharpe Ratio, max drawdown, returns
  runner.py              # Buy-and-hold backtest runner

rag/
  document_loader.py     # Loads local .md/.txt documents
  vector_store.py        # Lightweight local retrieval
  retriever.py           # Safe ticker-aware RAG interface
  README.md              # Knowledge base document format

tools/
  market_data.py         # Market data helpers
  news_tool.py           # News retrieval helpers
  sec_tool.py            # Reserved for SEC filing tools

stock_analysis.py        # Main command-line entry point
```

## Setup

Install dependencies with Poetry:

```bash
poetry install
```

Create your local `.env` file from the example:

```bash
cp .env.example .env
```

Then fill in your DeepSeek settings:

```text
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek/deepseek-v4-pro
```

## Usage

Run the app and enter a ticker when prompted:

```bash
poetry run python stock_analysis.py
```

Or pass the ticker directly:

```bash
poetry run python stock_analysis.py AAPL
```

Run with less CrewAI output:

```bash
poetry run python stock_analysis.py AAPL --quiet
```

Skip report generation:

```bash
poetry run python stock_analysis.py AAPL --no-report
```

Use a custom report directory:

```bash
poetry run python stock_analysis.py AAPL --report-dir reports
```

## Local RAG Knowledge Base

Place local knowledge documents under:

```text
data/documents/
```

Company-specific documents must include a matching ticker symbol:

```markdown
---
title: NVIDIA 2025 10-K Notes
symbol: NVDA
doc_type: 10-K
---

Notes about revenue, margin, cash flow, risks, and management commentary.
```

General documents can be used for any ticker:

```markdown
---
title: Position Sizing Rules
scope: general
doc_type: investment_framework
---

Portfolio construction and risk management rules.
```

Safety rule: company documents are only retrieved for the same ticker. For
example, an `AAPL` filing will not be used when analyzing `MSFT`.

## Reports

Generated reports are saved as Markdown files under `reports/`. Each report
includes:

- Market snapshot
- Investment logic
- Backtest risk metrics
- Sharpe Ratio
- Max drawdown
- Portfolio plan
- Recent news
- Data gaps
- Final AI analysis

## Notes

This project is an investment research assistant, not financial advice. Outputs
should be reviewed critically and validated against primary sources before
making investment decisions.
