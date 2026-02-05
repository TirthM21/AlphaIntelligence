
import os
import logging
from pathlib import Path
from sec_edgar_downloader import Downloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SECFetcher:
    """Fetcher for SEC filings using sec-edgar-downloader."""
    
    def __init__(self, download_dir: str = "./data/sec_filings"):
        """Initialize SEC fetcher.
        
        Args:
            download_dir: Directory to save downloaded filings
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Downloader (Requires Company Name, Email, and download path)
        # SEC requires a descriptive User-Agent: "Sample Company Name AdminContact@samplecompany.com"
        self.dl = Downloader("PersonalPortfolioScreener", "researcher@stockanalysis-bot.local", str(self.download_dir))
        
        logger.info(f"SECFetcher initialized (dir: {self.download_dir})")

    def download_latest_10q(self, ticker: str) -> str:
        """Download latest 10-Q filing.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Status string
        """
        try:
            # Quick check if already downloaded today (approximate)
            ticker_dir = self.download_dir / "sec-edgar-filings" / ticker / "10-Q"
            if ticker_dir.exists() and any(ticker_dir.iterdir()):
                logger.info(f"SEC CACHE HIT [{ticker}]: Filing already on disk.")
                return f"Success (Cached in {self.download_dir})"
                
            logger.info(f"SEC CACHE MISS [{ticker}]: Downloading latest 10-Q from EDGAR...")
            
            # Download latest 1 10-Q
            count = self.dl.get("10-Q", ticker, limit=1)
            
            if count > 0:
                return f"Success (Saved to {self.download_dir})"
            else:
                return "Failed (No filings found)"
                
        except Exception as e:
            logger.error(f"Error downloading 10-Q for {ticker}: {e}")
            return f"Error: {e}"

    def download_latest_10k(self, ticker: str) -> str:
        """Download latest 10-K filing."""
        try:
            logger.info(f"Downloading latest 10-K for {ticker}...")
            count = self.dl.get("10-K", ticker, limit=1)
            return f"Success (Saved to {self.download_dir})" if count > 0 else "Failed"
        except Exception as e:
            return f"Error: {e}"
