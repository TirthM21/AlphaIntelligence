
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path.cwd()))

from src.data.fmp_fetcher import FMPFetcher

def test_fmp():
    print("Testing FMP Fetcher...")
    try:
        api_key = os.getenv('FMP_API_KEY')
        fetcher = FMPFetcher(api_key=api_key)
        data = fetcher.fetch_income_statement("AAPL", limit=1)
        if data:
            print(f"Success! Fetched income statement for AAPL. Revenue: {data[0].get('revenue')}")
        else:
            print("Failed: No data returned from FMP.")
    except Exception as e:
        print(f"FMP Fetcher failed: {e}")

if __name__ == "__main__":
    load_dotenv()
    test_fmp()
