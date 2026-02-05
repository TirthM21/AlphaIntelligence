
import os
import requests
from dotenv import load_dotenv

def test():
    load_dotenv()
    key = os.getenv('FMP_API_KEY')
    eps = [
        "quote/AAPL",
        "income-statement/AAPL?period=quarter&limit=1",
        "stock_news?limit=1"
    ]
    for ep in eps:
        sep = "&" if "?" in ep else "?"
        url = f"https://financialmodelingprep.com/api/v3/{ep}{sep}apikey={key}"
        try:
            r = requests.get(url)
            print(f"EP: {ep}")
            print(f"Status: {r.status_code}")
            print(f"JSON: {r.json()[:1] if isinstance(r.json(), list) else r.json()}")
        except Exception as e:
            print(f"Fail {ep}: {e}")
        print("-" * 20)

if __name__ == "__main__":
    test()
