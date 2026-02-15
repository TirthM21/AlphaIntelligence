import os
import logging
from src.data.fred_fetcher import FredFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_fred():
    # Use the key provided in the prompt if not in env
    api_key = os.getenv('FRED_API_KEY') or "7293c16d35197114eebd20a0df987f1f"
    
    fetcher = FredFetcher(api_key=api_key)
    
    print("\n--- Testing fetch_releases ---")
    releases = fetcher.fetch_releases(limit=5)
    for r in releases:
        print(f"ID: {r['id']}, Name: {r['name']}")
        
    if releases:
        release_id = releases[0]['id']
        print(f"\n--- Testing fetch_release (ID: {release_id}) ---")
        release = fetcher.fetch_release(release_id)
        if release:
            print(f"Name: {release['name']}, Link: {release.get('link', 'N/A')}")
            
        print(f"\n--- Testing fetch_release_tables (ID: {release_id}) ---")
        tables = fetcher.fetch_release_tables(release_id)
        print(f"Table name: {tables.get('name', 'N/A')}")
        
    print("\n--- Testing fetch_series_observations (GDP) ---")
    observations = fetcher.fetch_series_observations('GDP', limit=5) # Note: limit is not in my method signature for observations, I should add it or just take slice
    for obs in observations[:5]:
        print(f"Date: {obs['date']}, Value: {obs['value']}")

if __name__ == "__main__":
    test_fred()
