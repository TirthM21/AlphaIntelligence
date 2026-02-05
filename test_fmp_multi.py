
import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

def test_fmp_v3():
    load_dotenv()
    api_key = os.getenv('FMP_API_KEY')
    endpoints = [
        "quote/AAPL",
        "income-statement/AAPL?period=quarter&limit=1",
        "balance-sheet-statement/AAPL?period=quarter&limit=1",
        "stock_news?limit=1"
    ]
    
    for ep in endpoints:
        url = f"https://financialmodelingprep.com/api/v3/{ep}"
        if '?' in ep:
            url += f"&apikey={api_key}"
        else:
            url += f"?apikey={api_key}"
            
        print(f"Testing: {url.replace(api_key, 'HIDDEN')}")
        try:
            response = requests.get(url, timeout=10)
            print(f"Status Code: {response.status_code}")
            print(f"Body snippet: {response.text[:200]}")
        except Exception as e:
            print(f"Fetch failed: {e}")
        print("-" * 20)

if __name__ == "__main__":
    test_fmp_v3()
