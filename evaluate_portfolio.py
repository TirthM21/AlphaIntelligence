import json
import logging
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.data.fetcher import YahooFinanceFetcher
from src.screening.phase_indicators import classify_phase, calculate_relative_strength
from src.screening.signal_engine import score_buy_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_portfolio(portfolio_file="portfolio.json", output_file="data/portfolio_evaluation.json"):
    """Evaluate current portfolio using Minervini criteria and scan methodology."""
    if not os.path.exists(portfolio_file):
        logger.error(f"{portfolio_file} not found.")
        return

    with open(portfolio_file, "r") as f:
        portfolio = json.load(f)

    fetcher = YahooFinanceFetcher()
    # Need SPY for Relative Strength
    spy_data = fetcher.fetch_price_history("SPY", period="2y")
    
    results = []
    
    for entry in portfolio:
        ticker = entry['ticker']
        logger.info(f"Evaluating {ticker}...")
        
        # 1. Fetch Data
        hist = fetcher.fetch_price_history(ticker, period="2y")
        if hist.empty:
            logger.warning(f"No history for {ticker}")
            continue
            
        current_price = hist['Close'].iloc[-1]
        
        # 2. Indicators
        phase_info = classify_phase(hist, current_price)
        rs_series = calculate_relative_strength(hist['Close'], spy_data['Close'])
        
        # 3. Minervini Score
        # We'll use the buy signal engine to see if it would be a "buy" today
        # This effectively evaluates its momentum/strength
        evaluation = score_buy_signal(
            ticker=ticker,
            price_data=hist,
            current_price=current_price,
            phase_info=phase_info,
            rs_series=rs_series
        )
        
        # 4. Performance
        cost_basis = entry['average_buy_price']
        gain_loss_pct = ((current_price - cost_basis) / cost_basis) * 100
        
        reasons = evaluation.get('reasons', [])
        if not reasons and 'reason' in evaluation:
            reasons = [evaluation['reason']]
            
        results.append({
            "ticker": ticker,
            "quantity": entry['quantity'],
            "cost_basis": cost_basis,
            "current_price": round(current_price, 2),
            "gain_loss_pct": round(gain_loss_pct, 2),
            "market_value": round(current_price * entry['quantity'], 2),
            "phase": phase_info['phase'],
            "phase_name": phase_info['phase_name'],
            "minervini_score": evaluation.get('score', 0),
            "is_minervini_compliant": evaluation.get('is_buy', False),
            "reasons": reasons
        })
        
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Evaluation complete. Saved to {output_file}")
    return results

if __name__ == "__main__":
    evaluate_portfolio()
