import logging
import sys
from src.data.marketaux_fetcher import MarketauxFetcher

logging.basicConfig(level=logging.INFO)

def test_fallback():
    fetcher = MarketauxFetcher()
    print("Testing Marketaux Trending (should trigger fallback if 403)...")
    entities = fetcher.fetch_trending_entities()
    print(f"Found {len(entities)} entities.")
    for e in entities[:3]:
        print(f"- {e.get('key')}: Sentiment {e.get('sentiment_avg')} ({e.get('total_documents')} docs)")

if __name__ == "__main__":
    test_fallback()
