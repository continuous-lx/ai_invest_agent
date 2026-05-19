"""Market data helpers backed by yfinance.

The functions in this module return plain Python objects so they can be used
directly from scripts, CrewAI tasks, or future tool wrappers without requiring
callers to understand pandas/yfinance internals.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import yfinance as yf


DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "1d"


@dataclass(frozen=True)
class PriceSnapshot:
    """Latest market snapshot for a ticker."""

    symbol: str
    price: float
    currency: str | None
    previous_close: float | None
    open: float | None
    day_high: float | None
    day_low: float | None
    volume: int | None
    market_cap: int | None
    timestamp: str

    @property
    def change(self) -> float | None:
        if self.previous_close is None:
            return None
        return self.price - self.previous_close

    @property
    def change_percent(self) -> float | None:
        if self.previous_close in (None, 0):
            return None
        return (self.price - self.previous_close) / self.previous_close * 100

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["change"] = self.change
        data["change_percent"] = self.change_percent
        return data


def _normalize_symbol(symbol: str) -> str:
    clean_symbol = symbol.strip().upper()
    if not clean_symbol:
        raise ValueError("symbol cannot be empty")
    return clean_symbol


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _last_close_from_history(ticker: yf.Ticker) -> float:
    history = ticker.history(period="5d", interval="1d", auto_adjust=False)
    if history.empty or "Close" not in history:
        raise ValueError("cannot fetch recent closing price")
    return float(history["Close"].dropna().iloc[-1])


def get_stock_price(symbol: str) -> float:
    """Return the latest available stock price for ``symbol``."""

    snapshot = get_price_snapshot(symbol)
    return snapshot.price


def get_price_snapshot(symbol: str) -> PriceSnapshot:
    """Fetch a current price snapshot for ``symbol``.

    yfinance's quote fields can be sparse for some tickers, so this function
    falls back to recent historical closes when the live quote price is missing.
    """

    clean_symbol = _normalize_symbol(symbol)
    ticker = yf.Ticker(clean_symbol)

    try:
        info = ticker.fast_info
    except Exception:
        info = {}

    price = _safe_float(_fast_info_get(info, "last_price", "lastPrice"))

    if price is None:
        price = _last_close_from_history(ticker)

    previous_close = _safe_float(_fast_info_get(info, "previous_close", "previousClose"))

    return PriceSnapshot(
        symbol=clean_symbol,
        price=price,
        currency=_fast_info_get(info, "currency"),
        previous_close=previous_close,
        open=_safe_float(_fast_info_get(info, "open")),
        day_high=_safe_float(_fast_info_get(info, "day_high", "dayHigh")),
        day_low=_safe_float(_fast_info_get(info, "day_low", "dayLow")),
        volume=_safe_int(_fast_info_get(info, "last_volume", "lastVolume")),
        market_cap=_safe_int(_fast_info_get(info, "market_cap", "marketCap")),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _fast_info_get(info: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(info, dict) and key in info:
            return info.get(key)

        try:
            return info[key]
        except Exception:
            pass

        try:
            return getattr(info, key)
        except Exception:
            pass

    return None


def get_price_history(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> list[dict[str, Any]]:
    """Return historical OHLCV rows for ``symbol``.

    Common yfinance periods include ``5d``, ``1mo``, ``3mo``, ``6mo``, ``1y``,
    and common intervals include ``1d``, ``1h``, ``15m``.
    """

    clean_symbol = _normalize_symbol(symbol)
    ticker = yf.Ticker(clean_symbol)
    history = ticker.history(period=period, interval=interval, auto_adjust=False)

    if history.empty:
        raise ValueError(f"cannot fetch history for {clean_symbol}")

    rows: list[dict[str, Any]] = []
    for index, row in history.reset_index().iterrows():
        date_value = row.get("Date") or row.get("Datetime")
        rows.append(
            {
                "symbol": clean_symbol,
                "date": date_value.isoformat() if hasattr(date_value, "isoformat") else str(date_value),
                "open": _safe_float(row.get("Open")),
                "high": _safe_float(row.get("High")),
                "low": _safe_float(row.get("Low")),
                "close": _safe_float(row.get("Close")),
                "adj_close": _safe_float(row.get("Adj Close")),
                "volume": _safe_int(row.get("Volume")),
            }
        )

    return rows


def get_company_profile(symbol: str) -> dict[str, Any]:
    """Return basic company metadata for ``symbol``."""

    clean_symbol = _normalize_symbol(symbol)
    ticker = yf.Ticker(clean_symbol)

    try:
        info = ticker.info
    except Exception as exc:
        raise ValueError(f"cannot fetch company profile for {clean_symbol}") from exc

    return {
        "symbol": clean_symbol,
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "website": info.get("website"),
        "business_summary": info.get("longBusinessSummary"),
    }


def get_market_summary(symbol: str) -> dict[str, Any]:
    """Return a compact market summary suitable for agent prompts."""

    snapshot = get_price_snapshot(symbol)
    profile = get_company_profile(symbol)

    return {
        "snapshot": snapshot.to_dict(),
        "company": profile,
        "text": format_market_summary(snapshot, profile),
    }


def format_market_summary(
    snapshot: PriceSnapshot,
    profile: dict[str, Any] | None = None,
) -> str:
    """Format a market snapshot as a concise human-readable summary."""

    profile = profile or {}
    currency = f" {snapshot.currency}" if snapshot.currency else ""
    change_text = "change unavailable"

    if snapshot.change is not None and snapshot.change_percent is not None:
        change_text = f"{snapshot.change:+.2f} ({snapshot.change_percent:+.2f}%)"

    parts = [
        f"{snapshot.symbol} latest price: {snapshot.price:.2f}{currency}",
        f"daily change: {change_text}",
    ]

    if snapshot.previous_close is not None:
        parts.append(f"previous close: {snapshot.previous_close:.2f}")
    if snapshot.day_low is not None and snapshot.day_high is not None:
        parts.append(f"day range: {snapshot.day_low:.2f}-{snapshot.day_high:.2f}")
    if snapshot.volume is not None:
        parts.append(f"volume: {snapshot.volume:,}")
    if profile.get("name"):
        parts.append(f"company: {profile['name']}")
    if profile.get("sector") or profile.get("industry"):
        parts.append(
            "classification: "
            + " / ".join(item for item in [profile.get("sector"), profile.get("industry")] if item)
        )

    return "; ".join(parts) + "."


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fetch stock market data.")
    parser.add_argument("symbol", help="Ticker symbol, for example AAPL or MSFT")
    parser.add_argument("--history", action="store_true", help="Print historical OHLCV data")
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    args = parser.parse_args()

    if args.history:
        result: Any = get_price_history(args.symbol, period=args.period, interval=args.interval)
    else:
        result = get_market_summary(args.symbol)

    print(json.dumps(result, ensure_ascii=False, indent=2))
