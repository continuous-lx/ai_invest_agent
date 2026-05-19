"""Portfolio manager agent and allocation logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from crewai import Agent, Task

from agents.analyst import InvestmentScore
from backtests.runner import format_backtest_summary


@dataclass(frozen=True)
class PortfolioPlan:
    """Position sizing guidance for a single-stock research workflow."""

    symbol: str
    action: str
    target_weight: float
    max_position_weight: float
    risk_level: str
    rationale: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_portfolio_manager(llm: Any, verbose: bool = True) -> Agent:
    """Create the portfolio manager agent."""

    return Agent(
        role="Portfolio Manager",
        goal=(
            "Convert analyst recommendations and backtest risk metrics into "
            "position sizing and portfolio risk guidance."
        ),
        backstory=(
            "You are a risk-aware portfolio manager. You care about position sizing, "
            "drawdowns, volatility, diversification, and clear risk limits. You do "
            "not turn a single-stock view into an oversized portfolio bet."
        ),
        llm=llm,
        verbose=verbose,
    )


def build_portfolio_plan(
    symbol: str,
    investment_score: InvestmentScore | dict[str, Any],
    backtest: dict[str, Any] | None = None,
) -> PortfolioPlan:
    """Build deterministic position sizing guidance from score and risk metrics."""

    clean_symbol = symbol.strip().upper()
    score = _score_value(investment_score, "total_score", default=0)
    recommendation = str(
        _score_value(investment_score, "recommendation_hint", default="HOLD")
    ).upper()
    max_drawdown = _optional_float((backtest or {}).get("max_drawdown"))
    sharpe = _optional_float((backtest or {}).get("sharpe_ratio"))

    risk_level = _risk_level(max_drawdown=max_drawdown, sharpe=sharpe)
    target_weight = _target_weight(recommendation, score, risk_level)
    max_position_weight = _max_position_weight(risk_level)
    action = _portfolio_action(target_weight)

    rationale = [
        f"Analyst score suggests {recommendation} with total score {score}.",
        _risk_rationale(max_drawdown=max_drawdown, sharpe=sharpe, risk_level=risk_level),
        f"Suggested target weight is {target_weight:.1%}, capped at {max_position_weight:.1%}.",
    ]

    return PortfolioPlan(
        symbol=clean_symbol,
        action=action,
        target_weight=target_weight,
        max_position_weight=max_position_weight,
        risk_level=risk_level,
        rationale=rationale,
    )


def _score_value(score: InvestmentScore | dict[str, Any], key: str, default: Any) -> Any:
    if isinstance(score, InvestmentScore):
        return getattr(score, key)
    return score.get(key, default)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _risk_level(max_drawdown: float | None, sharpe: float | None) -> str:
    if max_drawdown is None:
        return "Unknown"
    if max_drawdown <= -0.35 or (sharpe is not None and sharpe < 0):
        return "High"
    if max_drawdown <= -0.20 or (sharpe is not None and sharpe < 0.5):
        return "Medium"
    return "Low"


def _target_weight(recommendation: str, score: int, risk_level: str) -> float:
    if recommendation == "SELL":
        return 0.0
    if recommendation == "HOLD":
        base_weight = 0.03
    elif score >= 4:
        base_weight = 0.08
    else:
        base_weight = 0.05

    if risk_level == "High":
        return min(base_weight, 0.02)
    if risk_level == "Medium":
        return min(base_weight, 0.05)
    if risk_level == "Unknown":
        return min(base_weight, 0.03)
    return base_weight


def _max_position_weight(risk_level: str) -> float:
    if risk_level == "High":
        return 0.03
    if risk_level == "Medium":
        return 0.07
    if risk_level == "Unknown":
        return 0.05
    return 0.10


def _portfolio_action(target_weight: float) -> str:
    if target_weight <= 0:
        return "Avoid or exit"
    if target_weight <= 0.03:
        return "Watchlist or small position"
    return "Build position gradually"


def _risk_rationale(max_drawdown: float | None, sharpe: float | None, risk_level: str) -> str:
    drawdown_text = "max drawdown unavailable" if max_drawdown is None else f"max drawdown {max_drawdown:.1%}"
    sharpe_text = "Sharpe unavailable" if sharpe is None else f"Sharpe Ratio {sharpe:.2f}"
    return f"Risk level is {risk_level}: {drawdown_text}, {sharpe_text}."


def format_portfolio_plan(plan: PortfolioPlan) -> str:
    """Format portfolio guidance for prompts and reports."""

    lines = [
        f"Portfolio action: {plan.action}",
        f"Target weight: {plan.target_weight:.1%}",
        f"Maximum position weight: {plan.max_position_weight:.1%}",
        f"Risk level: {plan.risk_level}",
        "Rationale:",
    ]
    lines.extend(f"- {item}" for item in plan.rationale)
    return "\n".join(lines)


def create_portfolio_task(
    symbol: str,
    portfolio_manager: Agent,
    analysis_task: Task,
    portfolio_plan: PortfolioPlan,
    backtest: dict[str, Any] | None = None,
) -> Task:
    """Create the portfolio management task."""

    clean_symbol = symbol.strip().upper()
    return Task(
        description=(
            f"Create portfolio guidance for {clean_symbol} based on the analyst's "
            "recommendation and the risk data below.\n\n"
            "Backtest summary:\n"
            f"{format_backtest_summary(backtest)}\n\n"
            "Rules-based portfolio plan:\n"
            f"{format_portfolio_plan(portfolio_plan)}\n\n"
            "Return practical guidance for position sizing, risk controls, and "
            "conditions that would justify increasing, reducing, or avoiding exposure."
        ),
        expected_output=(
            "Portfolio guidance with action, suggested target weight, maximum "
            "position size, risk controls, and monitoring triggers."
        ),
        agent=portfolio_manager,
        context=[analysis_task],
    )


def build_portfolio_manager_bundle(
    symbol: str,
    llm: Any,
    analysis_task: Task,
    investment_score: InvestmentScore | dict[str, Any],
    backtest: dict[str, Any] | None = None,
    verbose: bool = True,
) -> tuple[Agent, Task, PortfolioPlan]:
    """Create a portfolio manager, task, and portfolio plan together."""

    plan = build_portfolio_plan(
        symbol=symbol,
        investment_score=investment_score,
        backtest=backtest,
    )
    portfolio_manager = create_portfolio_manager(llm=llm, verbose=verbose)
    task = create_portfolio_task(
        symbol=symbol,
        portfolio_manager=portfolio_manager,
        analysis_task=analysis_task,
        portfolio_plan=plan,
        backtest=backtest,
    )
    return portfolio_manager, task, plan
