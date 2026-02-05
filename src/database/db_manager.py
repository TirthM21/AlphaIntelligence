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
