
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path.cwd()))

from src.data.fred_fetcher import FredFetcher
from src.data.marketaux_fetcher import MarketauxFetcher
from src.notifications.email_notifier import EmailNotifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Diagnostic")

def test_providers():
    load_dotenv()
    
    print("\n=== Data Provider Diagnostics ===")
    
    # 1. Test FRED
    print("\n[FRED]")
    fred = FredFetcher()
    if not fred.api_key:
        print("❌ FRED_API_KEY missing in .env")
    else:
        try:
            macro = fred.fetch_canonical_macro_bundle()
            indicators = list(macro.get('indicators', {}).keys())
            if indicators:
                print(f"✅ FRED connection successful. Indicators: {', '.join(indicators)}")
            else:
                print("⚠️ FRED connected but returned no indicators (check API key permissions or limits)")
        except Exception as e:
            print(f"❌ FRED connection failed: {e}")

    # 2. Test MarketAux
    print("\n[MarketAux]")
    marketaux = MarketauxFetcher()
    if not marketaux.api_key:
        print("❌ MARKETAUX_API_KEY missing in .env")
    else:
        try:
            news = marketaux.fetch_market_news(limit=2)
            if news:
                print(f"✅ MarketAux connection successful. News items: {len(news)}")
                for i, art in enumerate(news, 1):
                    print(f"   {i}. {art.get('title', 'No Title')[:60]}...")
            else:
                print("⚠️ MarketAux connected but returned no news (check API key limits)")
        except Exception as e:
            print(f"❌ MarketAux connection failed: {e}")

    # 3. Test Email
    print("\n=== Email Pipeline Diagnostics ===")
    notifier = EmailNotifier()
    if not notifier.enabled:
        print("❌ EmailNotifier is DISABLED (Check EMAIL_SENDER and EMAIL_PASSWORD in .env)")
    else:
        print(f"✅ EmailNotifier enabled (Sender: {notifier.sender_email}, Recipient: {notifier.recipient_email})")
        print("Attempting to send a test email...")
        success = notifier.test_connection()
        if success:
            print("✅ Test email sent successfully!")
        else:
            print("❌ Test email delivery failed. Check SMTP settings or Gmail App Password.")

if __name__ == "__main__":
    test_providers()
