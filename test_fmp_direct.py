
import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

def test_fmp_direct():
    load_dotenv()
    api_key = os.getenv('FMP_API_KEY')
    url = f"https://financialmodelingprep.com/api/v3/income-statement/AAPL?limit=1&apikey={api_key}"
    
    print(f"Testing FMP direct: {url.replace(api_key, 'HIDDEN')}")
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")
    except Exception as e:
        print(f"Direct fetch failed: {e}")

if __name__ == "__main__":
    test_fmp_direct()
