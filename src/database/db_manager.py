"""Database management for users and newsletter subscribers using SQLAlchemy and Neon Postgres."""

import logging
import os
from datetime import datetime
from typing import List, Optional, Dict
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

Base = declarative_base()

class Subscriber(Base):
    """Newsletter subscriber model."""
    __tablename__ = 'subscribers'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    subscribed_at = Column(DateTime, default=datetime.utcnow)

class Recommendation(Base):
    """Stores generated buy/sell signals for historical tracking."""
    __tablename__ = 'recommendations'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)
    signal_type = Column(String(10), nullable=False) # 'BUY' or 'SELL'
    price_at_signal = Column(Float, nullable=False)
    score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    spy_price_at_signal = Column(Float) # For benchmarking

class PortfolioHolding(Base):
    """Current positions in the hedge fund portfolio."""
    __tablename__ = 'portfolio_holdings'
    
    ticker = Column(String(20), primary_key=True)
    quantity = Column(Integer, nullable=False)
    average_buy_price = Column(Float, nullable=False)
    current_price = Column(Float)
    sector = Column(String(100))
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TradeRecord(Base):
    """Historical ledger of all portfolio trades."""
    __tablename__ = 'trade_history'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)
    action = Column(String(10), nullable=False) # 'BUY' or 'SELL'
    price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DailyPerformance(Base):
    """Timeseries of total fund equity for performance analysis."""
    __tablename__ = 'daily_performance'
    
    date = Column(DateTime, primary_key=True, default=datetime.utcnow)
    total_equity = Column(Float, nullable=False) # Cash + Market Value
    cash_balance = Column(Float, default=100000.0) # Default $100k start
    benchmark_price = Column(Float) # SPY price
    nav = Column(Float) # Net Asset Value per share or scale

class AllocationPlan(Base):
    """Suggested allocations for new buy signals."""
    __tablename__ = 'allocation_plans'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)
    score = Column(Float)
    price = Column(Float)
    recommended_shares = Column(Integer)
    est_cost = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DBManager:
    """Handles connection and operations for the Neon Postgres database."""
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv('DATABASE_URL')
        if not self.db_url:
            logger.warning("DATABASE_URL not set. Database features will be disabled.")
            return

        try:
            # Fix for neon connection strings that might need 'postgresql://' instead of 'postgres://'
            if self.db_url.startswith("postgres://"):
                self.db_url = self.db_url.replace("postgres://", "postgresql://", 1)
            
            self.engine = create_engine(self.db_url)
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(self.engine)
            logger.info("Database connection established and tables verified.")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.db_url = None

    def add_subscriber(self, email: str, name: Optional[str] = None) -> bool:
        """Add a new newsletter subscriber."""
        if not self.db_url: return False
        
        session = self.Session()
        try:
            # Check if exists
            existing = session.query(Subscriber).filter_by(email=email).first()
            if existing:
                if not existing.is_active:
                    existing.is_active = True
                    session.commit()
                    return True
                return False # Already subscribed
            
            new_sub = Subscriber(email=email, name=name)
            session.add(new_sub)
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add subscriber: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def get_active_subscribers(self) -> List[str]:
        """Get all active subscriber emails."""
        if not self.db_url: return []
        
        session = self.Session()
        try:
            subs = session.query(Subscriber.email).filter_by(is_active=True).all()
            return [s.email for s in subs]
        except Exception as e:
            logger.error(f"Failed to fetch subscribers: {e}")
            return []
        finally:
            session.close()
            
    def unsubscribe(self, email: str) -> bool:
        """Deactivate a subscriber."""
        if not self.db_url: return False
        
        session = self.Session()
        try:
            sub = session.query(Subscriber).filter_by(email=email).first()
            if sub:
                sub.is_active = False
                session.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def record_recommendations(self, signals: List[Dict], spy_price: float):
        """Save a batch of signals to the database for tracking."""
        if not self.db_url or not signals: return
        
        session = self.Session()
        try:
            for s in signals:
                rec = Recommendation(
                    ticker=s['ticker'],
                    signal_type='BUY' if s.get('is_buy') else 'SELL',
                    price_at_signal=s['current_price'],
                    score=s['score'],
                    spy_price_at_signal=spy_price
                )
                session.add(rec)
            session.commit()
            logger.info(f"Recorded {len(signals)} recommendations for tracking.")
        except Exception as e:
            logger.error(f"Failed to record recommendations: {e}")
            session.rollback()
        finally:
            session.close()

    def get_recommendation_performance(self) -> List[Dict]:
        """Fetch historical recommendations for comparison."""
        if not self.db_url: return []
        
        session = self.Session()
        try:
            recs = session.query(Recommendation).all()
            return [
                {
                    'ticker': r.ticker,
                    'type': r.signal_type,
                    'entry_price': r.price_at_signal,
                    'spy_entry': r.spy_price_at_signal,
                    'date': r.timestamp
                } for r in recs
            ]
        finally:
            session.close()

    def get_latest_recommendations(self, limit: int = 200) -> List[Dict]:
        """Fetch the most recent buy signals recorded in the database."""
        if not self.db_url: return []
        
        session = self.Session()
        try:
            # We filter for BUY signals and sort by timestamp, then by score
            # In a real setup, we might filter by the exact 'latest' batch timestamp
            recs = session.query(Recommendation).filter_by(signal_type='BUY').order_by(Recommendation.timestamp.desc(), Recommendation.score.desc()).limit(limit).all()
            return [
                {
                    'ticker': r.ticker,
                    'score': r.score,
                    'price': r.price_at_signal,
                    'date': r.timestamp
                } for r in recs
            ]
        finally:
            session.close()

    def update_portfolio_from_signals(self, buy_signals: List[Dict], max_positions: int = 20):
        """Automatically update portfolio based on elite buy signals."""
        if not self.db_url or not buy_signals: return
        
        session = self.Session()
        try:
            # 1. Get current holdings
            current_holdings = session.query(PortfolioHolding).all()
            total_slots = max_positions - len(current_holdings)
            
            if total_slots <= 0:
                logger.info("Portfolio at max capacity. No new positions added.")
                return

            # 2. Add top buy signals until full
            for signal in buy_signals[:total_slots]:
                ticker = signal['ticker']
                # Check if already held
                existing = session.query(PortfolioHolding).filter_by(ticker=ticker).first()
                if not existing:
                    # New position - logical quantity (simulated)
                    qty = 100 # Default simulated qty
                    price = signal['current_price']
                    
                    new_pos = PortfolioHolding(
                        ticker=ticker,
                        quantity=qty,
                        average_buy_price=price,
                        current_price=price,
                        sector=signal.get('sector', 'Unknown')
                    )
                    session.add(new_pos)
                    
                    # Record in ledger
                    trade = TradeRecord(
                        ticker=ticker,
                        action='BUY',
                        price=price,
                        quantity=qty
                    )
                    session.add(trade)
                    logger.info(f"Hedge Fund: Added NEW position {ticker} at ${price}")
            
            session.commit()
        except Exception as e:
            logger.error(f"Failed to update portfolio: {e}")
            session.rollback()
        finally:
            session.close()

    def update_daily_performance(self, spy_price: float):
        """Record daily fund value and benchmark price."""
        if not self.db_url: return
        
        session = self.Session()
        try:
            holdings = session.query(PortfolioHolding).all()
            market_value = sum((h.current_price or h.average_buy_price) * h.quantity for h in holdings)
            
            # Simple cash balance logic - start 100k, subtract buys, add sells (simulation)
            # In a real app we'd track cash in DB, here we'll assume a balance
            cash = 100000.0 - sum(h.average_buy_price * h.quantity for h in holdings)
            total_equity = market_value + cash
            
            perf = DailyPerformance(
                date=datetime.utcnow(),
                total_equity=total_equity,
                cash_balance=cash,
                benchmark_price=spy_price,
                nav=total_equity / 1000.0 # Simulated NAV
            )
            session.add(perf)
            session.commit()
            logger.info(f"Hedge Fund: Recorded daily performance. Equity: ${total_equity:,.2f}")
        except Exception as e:
            logger.error(f"Failed to update performance: {e}")
            session.rollback()
        finally:
            session.close()

    def get_full_portfolio_data(self) -> Dict:
        """Fetch all data needed for Dashboards."""
        if not self.db_url: return {}
        
        session = self.Session()
        try:
            holdings = session.query(PortfolioHolding).all()
            history = session.query(DailyPerformance).order_by(DailyPerformance.date).all()
            trades = session.query(TradeRecord).order_by(TradeRecord.timestamp.desc()).limit(10).all()
            
            return {
                'holdings': [
                    {'ticker': h.ticker, 'quantity': h.quantity, 'average_buy_price': h.average_buy_price, 
                     'current_price': h.current_price, 'sector': h.sector} for h in holdings
                ],
                'equity_curve': [
                    {'date': p.date, 'equity': p.total_equity, 'cash': p.cash_balance, 
                     'spy': p.benchmark_price} for p in history
                ],
                'recent_trades': [
                    {'ticker': t.ticker, 'action': t.action, 'price': t.price, 
                     'qty': t.quantity, 'date': t.timestamp} for t in trades
                ]
            }
        finally:
            session.close()
    def save_allocation_plan(self, allocations: List[Dict]):
        """Save suggested allocation plan to database."""
        if not self.db_url or not allocations: return
        
        session = self.Session()
        try:
            # Clear old plan (optional, but usually we just want the latest)
            # Or we can just keep adding with timestamps
            for a in allocations:
                plan = AllocationPlan(
                    ticker=a['ticker'],
                    score=a['score'],
                    price=a['price'],
                    recommended_shares=a['recommended_shares'],
                    est_cost=a['est_cost']
                )
                session.add(plan)
            session.commit()
            logger.info(f"Saved {len(allocations)} allocation suggestions to SQL.")
        except Exception as e:
            logger.error(f"Failed to save allocation plan to SQL: {e}")
            session.rollback()
        finally:
            session.close()
