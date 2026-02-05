#!/usr/bin/env python3
"""Demonstration of the stock screening module.

This script shows how to use the screening module to identify undervalued
stocks near support levels using a combination of fundamental and technical analysis.
"""

import os
import sys
from datetime import datetime, timedelta

from src.data import YahooFinanceFetcher, StockDatabase
from src.screening import screen_candidates

# Set environment for demo (use SQLite for simplicity)
os.environ['DATABASE_URL'] = 'sqlite:///./stock_screener_demo.db'


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def fetch_sample_data(tickers):
    """Fetch sample data for demonstration."""
    print("\n" + "=" * 80)
    print("STEP 1: Fetching Stock Data")
    print("=" * 80)

    fetcher = YahooFinanceFetcher(cache_dir="./data/cache")
    db = StockDatabase()

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Fetching data for {ticker}...")

        try:
            # Fetch fundamentals
            fundamentals = fetcher.fetch_fundamentals(ticker)
            if fundamentals:
                db.save_stock_fundamentals(ticker, fundamentals)
                print(f"  ✓ Fundamentals saved")
                print(f"    P/E: {fundamentals.get('pe_ratio')}")
                print(f"    P/B: {fundamentals.get('pb_ratio')}")
                print(f"    Price: ${fundamentals.get('current_price')}")
            else:
                print(f"  ✗ Failed to fetch fundamentals")
                continue

            # Fetch price history (1 year)
            prices = fetcher.fetch_price_history(ticker, period="1y")
            if not prices.empty:
                db.save_price_history(ticker, prices)
                print(f"  ✓ Price history saved ({len(prices)} days)")
            else:
                print(f"  ✗ Failed to fetch price history")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    return db


def run_screening(db, tickers):
    """Run the screening algorithm."""
    print("\n" + "=" * 80)
    print("STEP 2: Running Screening Algorithm")
    print("=" * 80)

    print("\nAnalyzing stocks based on:")
    print("  • Value Score (70%): P/E, P/B, FCF yield, debt levels")
    print("  • Support Score (30%): Price vs support, RSI, volume")
    print("\nProcessing...")

    results = screen_candidates(
        db,
        tickers,
        value_weight=0.7,
        support_weight=0.3
    )

    return results


def display_results(results):
    """Display screening results."""
    print("\n" + "=" * 80)
    print("STEP 3: Screening Results")
    print("=" * 80)

    if results.empty:
        print("\nNo screening results available.")
        return

    print(f"\n✓ Successfully screened {len(results)} stocks\n")

    # Summary statistics
    print("Summary Statistics:")
    print(f"  Average Buy Signal: {results['buy_signal'].mean():.1f}")
    print(f"  Average Value Score: {results['value_score'].mean():.1f}")
    print(f"  Average Support Score: {results['support_score'].mean():.1f}")

    # Top candidates
    print("\n" + "-" * 80)
    print("TOP CANDIDATES (sorted by Buy Signal)")
    print("-" * 80)

    for idx, row in results.iterrows():
        print(f"\n#{idx + 1}: {row['ticker']} - {row['name']}")
        print(f"  Sector: {row['sector']}")
        print(f"  Current Price: ${row['current_price']:.2f}")

        if row['nearest_support']:
            distance_pct = ((row['current_price'] - row['nearest_support']) / row['nearest_support']) * 100
            print(f"  Nearest Support: ${row['nearest_support']:.2f} ({distance_pct:+.1f}%)")

        print(f"\n  Scores:")
        print(f"    Buy Signal:     {row['buy_signal']:.1f}/100 {'★' * int(row['buy_signal'] / 20)}")
        print(f"    Value Score:    {row['value_score']:.1f}/100 {'■' * int(row['value_score'] / 20)}")
        print(f"    Support Score:  {row['support_score']:.1f}/100 {'▲' * int(row['support_score'] / 20)}")

        print(f"\n  Fundamentals:")
        if row['pe_ratio']:
            print(f"    P/E Ratio:      {row['pe_ratio']:.2f}")
        if row['pb_ratio']:
            print(f"    P/B Ratio:      {row['pb_ratio']:.2f}")

        print(f"\n  Technicals:")
        if row['rsi']:
            rsi_status = "Oversold" if row['rsi'] < 30 else "Neutral" if row['rsi'] < 70 else "Overbought"
            print(f"    RSI:            {row['rsi']:.1f} ({rsi_status})")

        # Buy signal interpretation
        if row['buy_signal'] >= 80:
            signal = "🔥 STRONG BUY"
        elif row['buy_signal'] >= 65:
            signal = "✅ BUY"
        elif row['buy_signal'] >= 50:
            signal = "⚡ CONSIDER"
        else:
            signal = "⏸️  WATCH"

        print(f"\n  Signal: {signal}")
        print("  " + "-" * 76)


def display_detailed_analysis(results, ticker):
    """Display detailed analysis for a specific stock."""
    stock = results[results['ticker'] == ticker]

    if stock.empty:
        print(f"\n{ticker} not found in results.")
        return

    row = stock.iloc[0]

    print("\n" + "=" * 80)
    print(f"DETAILED ANALYSIS: {row['ticker']} - {row['name']}")
    print("=" * 80)

    print(f"\nSector: {row['sector']}")
    print(f"Current Price: ${row['current_price']:.2f}")

    print("\n" + "-" * 40)
    print("VALUE ANALYSIS")
    print("-" * 40)
    print(f"Value Score: {row['value_score']:.1f}/100")
    if row['pe_ratio']:
        pe_assessment = "Excellent" if row['pe_ratio'] < 15 else "Good" if row['pe_ratio'] < 25 else "Fair"
        print(f"  P/E Ratio: {row['pe_ratio']:.2f} ({pe_assessment})")
    if row['pb_ratio']:
        pb_assessment = "Excellent" if row['pb_ratio'] < 1.5 else "Good" if row['pb_ratio'] < 3 else "Fair"
        print(f"  P/B Ratio: {row['pb_ratio']:.2f} ({pb_assessment})")

    print("\n" + "-" * 40)
    print("TECHNICAL ANALYSIS")
    print("-" * 40)
    print(f"Support Score: {row['support_score']:.1f}/100")

    if row['nearest_support']:
        distance_pct = ((row['current_price'] - row['nearest_support']) / row['nearest_support']) * 100
        proximity = "At support" if abs(distance_pct) < 2 else "Near support" if abs(distance_pct) < 5 else "Away from support"
        print(f"  Nearest Support: ${row['nearest_support']:.2f} ({distance_pct:+.1f}%) - {proximity}")

    if row['rsi']:
        if row['rsi'] < 30:
            rsi_signal = "Deeply oversold - potential bounce"
        elif row['rsi'] < 40:
            rsi_signal = "Oversold - buying opportunity"
        elif row['rsi'] < 60:
            rsi_signal = "Neutral"
        elif row['rsi'] < 70:
            rsi_signal = "Approaching overbought"
        else:
            rsi_signal = "Overbought - caution"
        print(f"  RSI: {row['rsi']:.1f} - {rsi_signal}")

    print("\n" + "-" * 40)
    print("COMBINED SIGNAL")
    print("-" * 40)
    print(f"Buy Signal: {row['buy_signal']:.1f}/100")

    if row['buy_signal'] >= 80:
        print("\n🔥 STRONG BUY SIGNAL")
        print("This stock shows excellent value characteristics and is at or near")
        print("a technical support level. Consider for immediate purchase.")
    elif row['buy_signal'] >= 65:
        print("\n✅ BUY SIGNAL")
        print("This stock shows good value and technical setup. Consider adding")
        print("to your portfolio.")
    elif row['buy_signal'] >= 50:
        print("\n⚡ CONSIDER")
        print("This stock shows decent characteristics. Monitor for better entry.")
    else:
        print("\n⏸️  WATCH")
        print("This stock needs improvement in value or technical setup before")
        print("considering purchase.")


def main():
    """Main demonstration function."""
    print("\n" + "=" * 80)
    print("STOCK SCREENER DEMONSTRATION")
    print("Identifying Undervalued Stocks at Support Levels")
    print("=" * 80)

    # Sample tickers (mix of sectors)
    tickers = [
        "AAPL",   # Technology
        "MSFT",   # Technology
        "JPM",    # Financial
        "JNJ",    # Healthcare
        "WMT",    # Consumer
        "XOM",    # Energy
        "DIS",    # Entertainment
        "BA"      # Industrial
    ]

    print(f"\nDemo Configuration:")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Time Period: 1 year")
    print(f"  Weighting: 70% Value, 30% Technical Support")

    # Step 1: Fetch data
    db = fetch_sample_data(tickers)

    # Step 2: Run screening
    results = run_screening(db, tickers)

    # Step 3: Display results
    display_results(results)

    # Step 4: Detailed analysis of top candidate
    if not results.empty:
        top_ticker = results.iloc[0]['ticker']
        display_detailed_analysis(results, top_ticker)

    # Summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)

    if not results.empty:
        print(f"\n✓ Screened {len(results)} stocks successfully")
        print(f"✓ Top candidate: {results.iloc[0]['ticker']} (Buy Signal: {results.iloc[0]['buy_signal']:.1f})")
    else:
        print("\n⚠ No results generated. Check if data was fetched successfully.")

    print("\nNext Steps:")
    print("  1. Run this script again to see caching in action (instant results!)")
    print("  2. Modify the tickers list to screen different stocks")
    print("  3. Adjust value_weight and support_weight for different strategies")
    print("  4. Integrate this into your trading workflow")

    print(f"\nDatabase: stock_screener_demo.db")
    print(f"Cache: ./data/cache/")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
