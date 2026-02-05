
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path.cwd()))

from src.data.sec_fetcher import SECFetcher

def test_sec():
    print("Testing SEC Fetcher...")
    try:
        fetcher = SECFetcher(download_dir="./data/test_sec")
        result = fetcher.download_latest_10q("AAPL")
        print(f"Result for AAPL: {result}")
    except Exception as e:
        print(f"SEC Fetcher failed: {e}")

if __name__ == "__main__":
    load_dotenv()
    test_sec()
