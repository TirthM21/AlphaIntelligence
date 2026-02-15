import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_token = os.getenv('MARKETAUX_API_KEY') or "ElkIqFLhx7BrOtLorkT7uDD2jo0AgqHVZuxaFyQ6"

def test_marketaux():
    print(f"Testing Marketaux with token: {api_token[:5]}...")
    url = "https://api.marketaux.com/v1/entity/trending/aggregation"
    params = {
        'api_token': api_token,
        'countries': 'us,ca',
        'language': 'en',
        'limit': 5
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            entities = data.get('data', [])
            print(f"Found {len(entities)} trending entities.")
            for e in entities:
                print(f"- {e.get('key')}: Sentiment {e.get('sentiment_avg')}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_marketaux()
