"""Researcher agent and stock research context helpers."""

from __future__ import annotations

from typing import Any

from crewai import Agent, Task

from rag.retriever import retrieve_context
from tools.market_data import get_market_summary, get_price_history
from tools.news_tool import get_news_summary


DEFAULT_HISTORY_PERIOD = "1mo"
DEFAULT_HISTORY_INTERVAL = "1d"
DEFAULT_NEWS_LIMIT = 5
DEFAULT_RAG_TOP_K = 5


def create_researcher(llm: Any, verbose: bool = True) -> Agent:
    """Create the stock researcher agent."""

    return Agent(
        role="Stock Researcher",
        goal=(
            "Collect and summarize market data, recent news, and business context "
            "for a stock so downstream analysts can make an investment judgment."
        ),
        backstory=(
            "You are an experienced equity research associate. You focus on facts, "
            "recent catalysts, price action, business context, and risks. You avoid "
            "unsupported claims and clearly separate observed data from interpretation."
        ),
        llm=llm,
        verbose=verbose,
    )


def collect_research_context(
    symbol: str,
    history_period: str = DEFAULT_HISTORY_PERIOD,
    history_interval: str = DEFAULT_HISTORY_INTERVAL,
    news_limit: int = DEFAULT_NEWS_LIMIT,
    include_rag: bool = True,
    rag_top_k: int = DEFAULT_RAG_TOP_K,
) -> dict[str, Any]:
    """Collect market and news context for a ticker.

    The function keeps partial results when one source fails, so an agent can
    still work with the available information instead of losing the whole run.
    """

    clean_symbol = symbol.strip().upper()
    if not clean_symbol:
        raise ValueError("symbol cannot be empty")

    errors: list[str] = []

    try:
        market = get_market_summary(clean_symbol)
    except Exception as exc:
        market = None
        errors.append(f"market data unavailable: {exc}")

    try:
        history = get_price_history(
            clean_symbol,
            period=history_period,
            interval=history_interval,
        )
    except Exception as exc:
        history = []
        errors.append(f"price history unavailable: {exc}")

    try:
        news = get_news_summary(clean_symbol, limit=news_limit)
    except Exception as exc:
        news = {"symbol": clean_symbol, "articles": [], "text": "No recent news found."}
        errors.append(f"news unavailable: {exc}")

    if include_rag:
        try:
            rag = retrieve_context(clean_symbol, top_k=rag_top_k)
            if rag.get("data_gap"):
                errors.append(str(rag["data_gap"]))
        except Exception as exc:
            rag = {"symbol": clean_symbol, "results": [], "text": "Local RAG context unavailable."}
            errors.append(f"local RAG unavailable: {exc}")
    else:
        rag = {"symbol": clean_symbol, "results": [], "text": "Local RAG disabled."}

    return {
        "symbol": clean_symbol,
        "market": market,
        "history": history,
        "news": news,
        "rag": rag,
        "errors": errors,
        "text": format_research_context(
            symbol=clean_symbol,
            market=market,
            history=history,
            news=news,
            rag=rag,
            errors=errors,
        ),
    }


def format_research_context(
    symbol: str,
    market: dict[str, Any] | None,
    history: list[dict[str, Any]],
    news: dict[str, Any],
    rag: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> str:
    """Format collected context for a CrewAI task prompt."""

    sections = [f"Research context for {symbol}"]

    if market and market.get("text"):
        sections.append(f"Market snapshot:\n{market['text']}")

    performance_text = _format_price_performance(history)
    if performance_text:
        sections.append(f"Recent price performance:\n{performance_text}")

    news_text = news.get("text") if isinstance(news, dict) else None
    if news_text:
        sections.append(f"Recent news:\n{news_text}")

    rag_text = rag.get("text") if isinstance(rag, dict) else None
    if rag_text:
        sections.append(f"Local knowledge base:\n{rag_text}")

    if errors:
        sections.append("Data gaps:\n" + "\n".join(f"- {error}" for error in errors))

    return "\n\n".join(sections)


def _format_price_performance(history: list[dict[str, Any]]) -> str | None:
    closes = [row for row in history if row.get("close") is not None]
    if not closes:
        return None

    first = closes[0]
    last = closes[-1]
    first_close = float(first["close"])
    last_close = float(last["close"])
    change = last_close - first_close
    change_percent = (change / first_close * 100) if first_close else 0

    highs = [float(row["high"]) for row in history if row.get("high") is not None]
    lows = [float(row["low"]) for row in history if row.get("low") is not None]
    volumes = [int(row["volume"]) for row in history if row.get("volume") is not None]

    parts = [
        f"From {first.get('date')} to {last.get('date')}, close moved "
        f"from {first_close:.2f} to {last_close:.2f}, "
        f"{change:+.2f} ({change_percent:+.2f}%)."
    ]

    if lows and highs:
        parts.append(f"Range over the period: {min(lows):.2f}-{max(highs):.2f}.")
    if volumes:
        average_volume = sum(volumes) / len(volumes)
        parts.append(f"Average volume: {average_volume:,.0f}.")

    return " ".join(parts)


def create_research_task(
    symbol: str,
    researcher: Agent,
    context: dict[str, Any] | None = None,
) -> Task:
    """Create a research task for the provided researcher agent."""

    clean_symbol = symbol.strip().upper()
    research_context = context or collect_research_context(clean_symbol)

    return Task(
        description=(
            f"Prepare a concise equity research brief for {clean_symbol}.\n\n"
            f"{research_context['text']}\n\n"
            "Focus on current market snapshot, recent price action, notable news, "
            "business context, visible catalysts, and key risks. Do not issue a "
            "buy/hold/sell recommendation; leave valuation judgment to the analyst."
        ),
        expected_output=(
            "A concise stock research brief with sections for market snapshot, "
            "recent performance, news/catalysts, business context, and risks."
        ),
        agent=researcher,
    )


def build_researcher_bundle(
    symbol: str,
    llm: Any,
    verbose: bool = True,
) -> tuple[Agent, Task, dict[str, Any]]:
    """Create a researcher, task, and collected context together."""

    context = collect_research_context(symbol)
    researcher = create_researcher(llm=llm, verbose=verbose)
    task = create_research_task(symbol=symbol, researcher=researcher, context=context)
    return researcher, task, context
