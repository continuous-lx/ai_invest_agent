"""Backtest runners for stock strategies."""

from __future__ import annotations

from typing import Any

from backtests.metrics import (
    annualized_return,
    annualized_volatility,
    calculate_returns,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
)
from tools.market_data import get_price_history


DEFAULT_BACKTEST_PERIOD = "1y"
DEFAULT_BACKTEST_INTERVAL = "1d"


def backtest_buy_and_hold(
    price_history: list[dict[str, Any]],
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Backtest a simple buy-and-hold strategy from historical close prices."""

    rows = [row for row in price_history if row.get("close") is not None]
    if len(rows) < 2:
        raise ValueError("at least two close prices are required for backtesting")

    prices = [float(row["close"]) for row in rows]
    returns = calculate_returns(prices)

    return {
        "strategy": "buy_and_hold",
        "start_date": rows[0].get("date"),
        "end_date": rows[-1].get("date"),
        "observations": len(rows),
        "start_price": prices[0],
        "end_price": prices[-1],
        "cumulative_return": cumulative_return(prices),
        "annualized_return": annualized_return(prices),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate=risk_free_rate),
        "max_drawdown": max_drawdown(prices),
    }


def run_symbol_backtest(
    symbol: str,
    period: str = DEFAULT_BACKTEST_PERIOD,
    interval: str = DEFAULT_BACKTEST_INTERVAL,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Fetch price history and run a buy-and-hold backtest for a symbol."""

    clean_symbol = symbol.strip().upper()
    if not clean_symbol:
        raise ValueError("symbol cannot be empty")

    history = get_price_history(clean_symbol, period=period, interval=interval)
    result = backtest_buy_and_hold(history, risk_free_rate=risk_free_rate)
    result["symbol"] = clean_symbol
    result["period"] = period
    result["interval"] = interval
    return result


def format_backtest_summary(result: dict[str, Any] | None) -> str:
    """Format a backtest result for prompts and reports."""

    if not result:
        return "Backtest unavailable."

    return (
        f"Strategy: {result.get('strategy', 'unknown')}\n"
        f"Period: {result.get('start_date')} to {result.get('end_date')}\n"
        f"Observations: {result.get('observations')}\n"
        f"Cumulative return: {_format_percent(result.get('cumulative_return'))}\n"
        f"Annualized return: {_format_percent(result.get('annualized_return'))}\n"
        f"Annualized volatility: {_format_percent(result.get('annualized_volatility'))}\n"
        f"Sharpe Ratio: {_format_number(result.get('sharpe_ratio'))}\n"
        f"Max drawdown: {_format_percent(result.get('max_drawdown'))}"
    )


def _format_percent(value: Any) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value) * 100:.2f}%"


def _format_number(value: Any) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):.2f}"
