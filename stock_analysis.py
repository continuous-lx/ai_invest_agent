"""Command-line entry point for AI-assisted stock analysis."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from crewai import Crew, LLM, Process
from dotenv import load_dotenv

from agents.analyst import build_analyst_bundle, format_investment_score
from agents.portfolio_manager import build_portfolio_manager_bundle, format_portfolio_plan
from agents.researcher import build_researcher_bundle
from backtests.runner import format_backtest_summary, run_symbol_backtest


DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_SYMBOL = "AAPL"
DEFAULT_REPORT_DIR = "reports"


def create_llm() -> LLM:
    """Create the configured DeepSeek-backed CrewAI LLM."""

    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_url = os.getenv("DEEPSEEK_API_URL")
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)

    if not api_key or not api_url:
        raise ValueError(
            "Missing DEEPSEEK_API_KEY or DEEPSEEK_API_URL. "
            "Set them in your environment or .env file."
        )

    return LLM(
        model=model,
        api_key=api_key,
        base_url=api_url,
    )


def create_crew(symbol: str, llm: Any | None = None, verbose: bool = True) -> tuple[Crew, dict[str, Any]]:
    """Create the full research-and-analysis crew for a ticker."""

    clean_symbol = symbol.strip().upper()
    if not clean_symbol:
        raise ValueError("symbol cannot be empty")

    active_llm = llm or create_llm()
    researcher, research_task, research_context = build_researcher_bundle(
        symbol=clean_symbol,
        llm=active_llm,
        verbose=verbose,
    )
    analyst, analysis_task, investment_score = build_analyst_bundle(
        symbol=clean_symbol,
        llm=active_llm,
        research_task=research_task,
        research_context=research_context,
        verbose=verbose,
    )
    research_context["investment_score"] = investment_score.to_dict()
    research_context["investment_score_text"] = format_investment_score(investment_score)

    try:
        backtest = run_symbol_backtest(clean_symbol)
        research_context["backtest"] = backtest
        research_context["backtest_text"] = format_backtest_summary(backtest)
    except Exception as exc:
        backtest = None
        research_context["backtest"] = None
        research_context["backtest_text"] = "Backtest unavailable."
        research_context.setdefault("errors", []).append(f"backtest unavailable: {exc}")

    portfolio_manager, portfolio_task, portfolio_plan = build_portfolio_manager_bundle(
        symbol=clean_symbol,
        llm=active_llm,
        analysis_task=analysis_task,
        investment_score=investment_score,
        backtest=backtest,
        verbose=verbose,
    )
    research_context["portfolio_plan"] = portfolio_plan.to_dict()
    research_context["portfolio_plan_text"] = format_portfolio_plan(portfolio_plan)

    crew = Crew(
        agents=[researcher, analyst, portfolio_manager],
        tasks=[research_task, analysis_task, portfolio_task],
        process=Process.sequential,
        verbose=verbose,
    )
    return crew, research_context


def generate_report(
    symbol: str,
    research_context: dict[str, Any],
    analysis_result: Any,
    output_dir: str = DEFAULT_REPORT_DIR,
) -> Path:
    """Write a Markdown investment report and return its path."""

    clean_symbol = symbol.strip().upper()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{clean_symbol}_{timestamp}.md"

    market_text = "Market data unavailable."
    market = research_context.get("market")
    if market and market.get("text"):
        market_text = market["text"]

    news_text = "No recent news found."
    news = research_context.get("news")
    if isinstance(news, dict) and news.get("text"):
        news_text = news["text"]

    score_text = research_context.get("investment_score_text", "Investment score unavailable.")
    backtest_text = research_context.get("backtest_text", "Backtest unavailable.")
    portfolio_plan_text = research_context.get("portfolio_plan_text", "Portfolio plan unavailable.")
    data_gaps = research_context.get("errors") or []
    gap_text = "\n".join(f"- {error}" for error in data_gaps) if data_gaps else "None."

    report = (
        f"# {clean_symbol} Investment Report\n\n"
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "## Market Snapshot\n\n"
        f"{market_text}\n\n"
        "## Investment Logic\n\n"
        f"{score_text}\n\n"
        "## Backtest Risk Metrics\n\n"
        f"{backtest_text}\n\n"
        "## Portfolio Plan\n\n"
        f"{portfolio_plan_text}\n\n"
        "## Recent News\n\n"
        f"{news_text}\n\n"
        "## Data Gaps\n\n"
        f"{gap_text}\n\n"
        "## AI Analysis\n\n"
        f"{analysis_result}\n"
    )

    report_path.write_text(report, encoding="utf-8")
    return report_path


def run(
    symbol: str,
    verbose: bool = True,
    write_report: bool = True,
    report_dir: str = DEFAULT_REPORT_DIR,
) -> Any:
    """Run the stock analysis workflow and print the result."""

    clean_symbol = symbol.strip().upper()
    crew, research_context = create_crew(clean_symbol, verbose=verbose)

    market = research_context.get("market")
    if market and market.get("text"):
        print(market["text"])
        print()

    if research_context.get("errors"):
        print("Data gaps:")
        for error in research_context["errors"]:
            print(f"- {error}")
        print()

    result = crew.kickoff()

    print("\n=== Final Result ===\n")
    print(result)

    if write_report:
        report_path = generate_report(
            symbol=clean_symbol,
            research_context=research_context,
            analysis_result=result,
            output_dir=report_dir,
        )
        print(f"\nReport saved to: {report_path}")

    return result


def prompt_for_symbol() -> str:
    """Prompt the user for a ticker symbol until a non-empty value is provided."""

    while True:
        symbol = input("Enter a stock ticker symbol, for example AAPL, MSFT, or TSLA: ").strip().upper()
        if symbol:
            return symbol
        print("Ticker symbol cannot be empty. Please try again.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI-assisted stock analysis.")
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Ticker symbol, for example AAPL or MSFT. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce CrewAI verbose output.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write a Markdown report file.",
    )
    parser.add_argument(
        "--report-dir",
        default=DEFAULT_REPORT_DIR,
        help="Directory where Markdown reports are saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    symbol = args.symbol or os.getenv("STOCK_SYMBOL") or prompt_for_symbol()
    run(
        symbol,
        verbose=not args.quiet,
        write_report=not args.no_report,
        report_dir=args.report_dir,
    )
