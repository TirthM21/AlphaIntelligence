from src.database.db_manager import DBManager
import os
from dotenv import load_dotenv

load_dotenv()

def seed_data():
    db = DBManager()
    if not db.db_url:
        return

    print("Seeding Hedge Fund dummy data...")
    
    # Simulate a few buy signals
    fake_buys = [
        {'ticker': 'AAPL', 'current_price': 190.50, 'sector': 'Technology', 'score': 92.5},
        {'ticker': 'MSFT', 'current_price': 405.20, 'sector': 'Technology', 'score': 88.0},
        {'ticker': 'NVDA', 'current_price': 720.10, 'sector': 'Technology', 'score': 95.5},
        {'ticker': 'TSLA', 'current_price': 185.30, 'sector': 'Automotive', 'score': 82.0}
    ]
    
    # Use existing method to populate portfolio
    db.update_portfolio_from_signals(fake_buys)
    
    # Record first performance entry (start at 100k)
    db.update_daily_performance(spy_price=480.0)
    
    # Record another day with growth
    from datetime import datetime, timedelta
    from sqlalchemy.orm import sessionmaker
    from src.database.db_manager import DailyPerformance, PortfolioHolding, TradeRecord
    
    # Manually update prices for simulation
    session = db.Session()
    session.query(PortfolioHolding).filter_by(ticker='NVDA').update({'current_price': 750.0})
    session.query(PortfolioHolding).filter_by(ticker='AAPL').update({'current_price': 195.0})
    session.commit()
    
    db.update_daily_performance(spy_price=485.0)
    
    print("✅ Seed performance and holdings recorded.")

if __name__ == "__main__":
    seed_data()
