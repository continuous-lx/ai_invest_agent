"""News helpers for stock research workflows.

The module intentionally uses only the Python standard library plus yfinance,
matching the project's current dependency set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

import yfinance as yf


DEFAULT_NEWS_LIMIT = 10
YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"


@dataclass(frozen=True)
class NewsArticle:
    """Normalized news item used by agents and scripts."""

    title: str
    publisher: str | None
    link: str | None
    published_at: str | None
    summary: str | None
    source_symbol: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_symbol(symbol: str) -> str:
    clean_symbol = symbol.strip().upper()
    if not clean_symbol:
        raise ValueError("symbol cannot be empty")
    return clean_symbol


def _normalize_query(query: str) -> str:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query cannot be empty")
    return clean_query


def _utc_from_unix(timestamp: Any) -> str | None:
    if timestamp in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError):
        return None


def _fetch_json(url: str, timeout: int = 10) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": "ai-invest-agent/0.1",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("news endpoint returned an unexpected payload")
    return data


def _article_from_yahoo_news(item: dict[str, Any], source_symbol: str | None = None) -> NewsArticle | None:
    title = item.get("title") or item.get("headline")
    if not title:
        return None

    publisher = item.get("publisher") or item.get("provider") or item.get("source")
    link = item.get("link") or item.get("url")
    published_at = _utc_from_unix(item.get("providerPublishTime") or item.get("pubDate"))
    summary = item.get("summary") or item.get("description")

    return NewsArticle(
        title=str(title),
        publisher=str(publisher) if publisher else None,
        link=str(link) if link else None,
        published_at=published_at,
        summary=str(summary) if summary else None,
        source_symbol=source_symbol,
    )


def _article_from_yfinance_news(item: dict[str, Any], source_symbol: str) -> NewsArticle | None:
    """Normalize both old and new yfinance news response shapes."""

    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = content.get("title")
    if not title:
        return None

    provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
    canonical_url = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else {}
    click_url = content.get("clickThroughUrl") if isinstance(content.get("clickThroughUrl"), dict) else {}

    publisher = (
        content.get("publisher")
        or provider.get("displayName")
        or provider.get("name")
    )
    link = canonical_url.get("url") or click_url.get("url") or content.get("link")
    published_at = (
        content.get("pubDate")
        or content.get("displayTime")
        or _utc_from_unix(content.get("providerPublishTime"))
    )
    summary = content.get("summary") or content.get("description")

    return NewsArticle(
        title=str(title),
        publisher=str(publisher) if publisher else None,
        link=str(link) if link else None,
        published_at=str(published_at) if published_at else None,
        summary=str(summary) if summary else None,
        source_symbol=source_symbol,
    )


def search_news(query: str, limit: int = DEFAULT_NEWS_LIMIT) -> list[dict[str, Any]]:
    """Search finance news and return normalized article dictionaries."""

    clean_query = _normalize_query(query)
    if limit <= 0:
        return []

    params = urlencode(
        {
            "q": clean_query,
            "quotesCount": 0,
            "newsCount": limit,
            "enableFuzzyQuery": "false",
            "quotesQueryId": "tss_match_phrase_query",
            "newsQueryId": "news_cie_vespa",
        }
    )
    data = _fetch_json(f"{YAHOO_SEARCH_URL}?{params}")

    articles: list[NewsArticle] = []
    for item in data.get("news", []):
        if not isinstance(item, dict):
            continue
        article = _article_from_yahoo_news(item)
        if article:
            articles.append(article)

    return [article.to_dict() for article in articles[:limit]]


def get_stock_news(symbol: str, limit: int = DEFAULT_NEWS_LIMIT) -> list[dict[str, Any]]:
    """Return recent finance news for a ticker symbol."""

    clean_symbol = _normalize_symbol(symbol)
    if limit <= 0:
        return []

    try:
        articles = search_news(clean_symbol, limit=limit)
    except Exception:
        articles = []

    if not articles:
        articles = _get_stock_news_from_yfinance(clean_symbol, limit=limit)

    for article in articles:
        article["source_symbol"] = article.get("source_symbol") or clean_symbol

    return articles[:limit]


def _get_stock_news_from_yfinance(symbol: str, limit: int) -> list[dict[str, Any]]:
    ticker = yf.Ticker(symbol)

    try:
        raw_news = ticker.news
    except Exception as exc:
        raise ValueError(f"cannot fetch news for {symbol}") from exc

    articles: list[NewsArticle] = []
    for item in raw_news or []:
        if not isinstance(item, dict):
            continue
        article = _article_from_yfinance_news(item, source_symbol=symbol)
        if article:
            articles.append(article)

    return [article.to_dict() for article in articles[:limit]]


def format_news_summary(articles: list[dict[str, Any]], max_items: int = 5) -> str:
    """Format news items as compact text suitable for an agent prompt."""

    if not articles:
        return "No recent news found."

    lines: list[str] = []
    for index, article in enumerate(articles[:max_items], start=1):
        publisher = f" - {article['publisher']}" if article.get("publisher") else ""
        published_at = f" ({article['published_at']})" if article.get("published_at") else ""
        summary = f": {article['summary']}" if article.get("summary") else ""
        lines.append(f"{index}. {article['title']}{publisher}{published_at}{summary}")

    return "\n".join(lines)


def get_news_summary(symbol: str, limit: int = DEFAULT_NEWS_LIMIT) -> dict[str, Any]:
    """Return structured news plus a formatted summary for a ticker."""

    clean_symbol = _normalize_symbol(symbol)
    articles = get_stock_news(clean_symbol, limit=limit)
    return {
        "symbol": clean_symbol,
        "articles": articles,
        "text": format_news_summary(articles),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch recent finance news.")
    parser.add_argument("query", help="Ticker symbol or finance news query")
    parser.add_argument("--limit", type=int, default=DEFAULT_NEWS_LIMIT)
    parser.add_argument("--search", action="store_true", help="Use free-text search instead of ticker news")
    args = parser.parse_args()

    if args.search:
        result: Any = {
            "query": args.query,
            "articles": search_news(args.query, limit=args.limit),
        }
        result["text"] = format_news_summary(result["articles"])
    else:
        result = get_news_summary(args.query, limit=args.limit)

    print(json.dumps(result, ensure_ascii=False, indent=2))
