"""Analyst agent and investment scoring logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from crewai import Agent, Task


POSITIVE_NEWS_TERMS = {
    "beat",
    "beats",
    "upgrade",
    "upgraded",
    "growth",
    "profit",
    "profits",
    "record",
    "strong",
    "raises",
    "raised",
    "surge",
    "surges",
    "bullish",
    "buy",
}

NEGATIVE_NEWS_TERMS = {
    "miss",
    "misses",
    "downgrade",
    "downgraded",
    "loss",
    "losses",
    "weak",
    "cuts",
    "cut",
    "falls",
    "decline",
    "declines",
    "lawsuit",
    "probe",
    "bearish",
    "sell",
}


@dataclass(frozen=True)
class InvestmentScore:
    """Deterministic scorecard used to guide the analyst agent."""

    symbol: str
    total_score: int
    recommendation_hint: str
    momentum_score: int
    news_score: int
    data_quality_score: int
    confidence: str
    rationale: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_analyst(llm: Any, verbose: bool = True) -> Agent:
    """Create the analyst agent that turns research into a recommendation."""

    return Agent(
        role="Stock Analyst",
        goal=(
            "Evaluate stock research and produce a clear buy, hold, or sell "
            "recommendation with concise reasoning."
        ),
        backstory=(
            "You are a disciplined equity analyst. You weigh business quality, "
            "recent catalysts, risk, price action, data quality, and uncertainty "
            "before making a recommendation. You are explicit when evidence is incomplete."
        ),
        llm=llm,
        verbose=verbose,
    )


def build_investment_score(context: dict[str, Any]) -> InvestmentScore:
    """Build a simple rules-based investment score from collected research data."""

    symbol = str(context.get("symbol", "")).upper()
    history = context.get("history") or []
    news = context.get("news") or {}
    errors = context.get("errors") or []

    momentum_score, momentum_reason = _score_momentum(history)
    news_score, news_reason = _score_news(news.get("articles", []))
    data_quality_score, data_reason = _score_data_quality(context, errors)

    total_score = momentum_score + news_score + data_quality_score
    recommendation_hint = _recommendation_from_score(total_score)
    confidence = _confidence_from_context(history, news.get("articles", []), errors)

    rationale = [momentum_reason, news_reason, data_reason]

    return InvestmentScore(
        symbol=symbol,
        total_score=total_score,
        recommendation_hint=recommendation_hint,
        momentum_score=momentum_score,
        news_score=news_score,
        data_quality_score=data_quality_score,
        confidence=confidence,
        rationale=rationale,
    )


def _score_momentum(history: list[dict[str, Any]]) -> tuple[int, str]:
    closes = [row for row in history if row.get("close") is not None]
    if len(closes) < 2:
        return 0, "Momentum: not enough price history to score."

    first_close = float(closes[0]["close"])
    last_close = float(closes[-1]["close"])
    if first_close == 0:
        return 0, "Momentum: first close is zero, so return cannot be scored."

    return_percent = (last_close - first_close) / first_close * 100

    if return_percent >= 8:
        score = 2
    elif return_percent >= 2:
        score = 1
    elif return_percent <= -8:
        score = -2
    elif return_percent <= -2:
        score = -1
    else:
        score = 0

    return score, f"Momentum: recent close-to-close return is {return_percent:+.2f}%."


def _score_news(articles: list[dict[str, Any]]) -> tuple[int, str]:
    if not articles:
        return 0, "News: no recent articles available to score."

    positive_hits = 0
    negative_hits = 0

    for article in articles:
        text = " ".join(
            str(article.get(field, ""))
            for field in ("title", "summary")
        ).lower()
        words = set(text.replace(",", " ").replace(".", " ").replace(":", " ").split())
        positive_hits += len(words & POSITIVE_NEWS_TERMS)
        negative_hits += len(words & NEGATIVE_NEWS_TERMS)

    net_hits = positive_hits - negative_hits
    if net_hits >= 3:
        score = 2
    elif net_hits >= 1:
        score = 1
    elif net_hits <= -3:
        score = -2
    elif net_hits <= -1:
        score = -1
    else:
        score = 0

    return score, f"News: {positive_hits} positive keyword hits and {negative_hits} negative keyword hits."


def _score_data_quality(context: dict[str, Any], errors: list[str]) -> tuple[int, str]:
    if errors:
        return -1, "Data quality: one or more data sources were unavailable."
    if context.get("market") and context.get("history") and context.get("news"):
        return 1, "Data quality: market, price history, and news data are available."
    return 0, "Data quality: enough data is available, but coverage is incomplete."


def _recommendation_from_score(score: int) -> str:
    if score >= 3:
        return "BUY"
    if score <= -3:
        return "SELL"
    return "HOLD"


def _confidence_from_context(
    history: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    errors: list[str],
) -> str:
    if errors or len(history) < 5:
        return "Low"
    if len(history) >= 15 and len(articles) >= 3:
        return "Medium"
    return "Low-Medium"


def format_investment_score(score: InvestmentScore) -> str:
    """Format the investment scorecard for prompt and report use."""

    lines = [
        f"Rules-based recommendation hint: {score.recommendation_hint}",
        f"Total score: {score.total_score}",
        f"Momentum score: {score.momentum_score}",
        f"News score: {score.news_score}",
        f"Data quality score: {score.data_quality_score}",
        f"Confidence: {score.confidence}",
        "Rationale:",
    ]
    lines.extend(f"- {item}" for item in score.rationale)
    return "\n".join(lines)


def create_analysis_task(
    symbol: str,
    analyst: Agent,
    research_task: Task,
    investment_score: InvestmentScore,
) -> Task:
    """Create the final investment analysis task."""

    clean_symbol = symbol.strip().upper()
    return Task(
        description=(
            f"Using the research brief for {clean_symbol}, evaluate whether the "
            "stock is attractive at the current market setup.\n\n"
            "Use this rules-based scorecard as a starting point, not as a final answer:\n"
            f"{format_investment_score(investment_score)}\n\n"
            "Provide a recommendation of buy, hold, or sell. Explain the key "
            "drivers, major risks, and what evidence would change the view. Keep "
            "the answer concise and practical for an individual investor."
        ),
        expected_output=(
            "A concise investment recommendation with rating, thesis, supporting "
            "evidence, key risks, confidence level, and watchlist items."
        ),
        agent=analyst,
        context=[research_task],
    )


def build_analyst_bundle(
    symbol: str,
    llm: Any,
    research_task: Task,
    research_context: dict[str, Any],
    verbose: bool = True,
) -> tuple[Agent, Task, InvestmentScore]:
    """Create an analyst, task, and investment score together."""

    investment_score = build_investment_score(research_context)
    analyst = create_analyst(llm=llm, verbose=verbose)
    task = create_analysis_task(
        symbol=symbol,
        analyst=analyst,
        research_task=research_task,
        investment_score=investment_score,
    )
    return analyst, task, investment_score
