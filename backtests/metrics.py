"""Portfolio and strategy risk metrics."""

from __future__ import annotations

from math import sqrt
from statistics import mean, stdev


TRADING_DAYS_PER_YEAR = 252


def calculate_returns(prices: list[float]) -> list[float]:
    """Convert a price series into simple period returns."""

    returns: list[float] = []
    for previous, current in zip(prices, prices[1:]):
        if previous == 0:
            continue
        returns.append((current - previous) / previous)
    return returns


def sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | None:
    """Calculate annualized Sharpe Ratio from period returns."""

    if len(returns) < 2:
        return None

    period_risk_free_rate = risk_free_rate / periods_per_year
    excess_returns = [value - period_risk_free_rate for value in returns]
    volatility = stdev(excess_returns)

    if volatility == 0:
        return None

    return mean(excess_returns) / volatility * sqrt(periods_per_year)


def max_drawdown(prices: list[float]) -> float | None:
    """Calculate maximum drawdown as a negative decimal value."""

    if not prices:
        return None

    peak = prices[0]
    worst_drawdown = 0.0

    for price in prices:
        if price > peak:
            peak = price
        if peak == 0:
            continue
        drawdown = (price - peak) / peak
        if drawdown < worst_drawdown:
            worst_drawdown = drawdown

    return worst_drawdown


def cumulative_return(prices: list[float]) -> float | None:
    """Calculate total return for a price series."""

    if len(prices) < 2 or prices[0] == 0:
        return None
    return (prices[-1] - prices[0]) / prices[0]


def annualized_return(
    prices: list[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | None:
    """Calculate annualized return from a price series."""

    if len(prices) < 2 or prices[0] == 0:
        return None

    total_return = prices[-1] / prices[0]
    years = (len(prices) - 1) / periods_per_year
    if years <= 0:
        return None

    return total_return ** (1 / years) - 1


def annualized_volatility(
    returns: list[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | None:
    """Calculate annualized volatility from period returns."""

    if len(returns) < 2:
        return None
    return stdev(returns) * sqrt(periods_per_year)
