import logging
import sys
from pathlib import Path
from src.reporting.newsletter_generator import NewsletterGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_full_newsletter():
    """Test the daily newsletter generation process."""
    try:
        logger.info("Starting Daily Newsletter Verification...")
        gen = NewsletterGenerator()
        
        # Create mock data
        mock_market_status = {
            'spy': {'trend': 'UPTREND', 'current_price': 500.25},
            'breadth': {'advance_decline_ratio': 1.5, 'percent_above_200sma': 65.0}
        }
        
        mock_buys = [
            {
                'ticker': 'NVDA',
                'score': 95.5,
                'current_price': 725.10,
                'fundamental_snapshot': "Core AI leader with 200% EPS growth.",
                'details': {'volume_score': 9.0}
            },
            {
                'ticker': 'AMD',
                'score': 88.0,
                'current_price': 180.50,
                'fundamental_snapshot': "Expanding margins in data center segment.",
                'details': {'volume_score': 8.5}
            }
        ]
        
        output_file = "data/newsletters/test_daily_output.md"
        logger.info(f"Generating newsletter to {output_file}...")
        
        # Test the generator
        path = gen.generate_newsletter(
            market_status=mock_market_status,
            top_buys=mock_buys,
            top_sells=[],
            output_path=output_file
        )
        
        if Path(path).exists():
            logger.info("✓ Newsletter successfully generated!")
            logger.info(f"✓ Location: {path}")
            
            # Print a snippet
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info("\n--- PREVIEW ---\n")
                logger.info(Path(path).read_text(encoding='utf-8')[:800] + "...")
        else:
            logger.error("❌ Newsletter file was not created.")
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_quarterly_newsletter():
    """Test the quarterly newsletter generation process."""
    try:
        logger.info("\nStarting Quarterly Newsletter Verification...")
        gen = NewsletterGenerator()
        
        # Mock portfolio object
        class MockPortfolio:
            def __init__(self):
                self.total_score = 88.5
                self.sector_concentration = 0.12
                self.total_positions = 20
                self.core_allocations = ['NVDA', 'MSFT', 'META']
                self.satellite_allocations = ['XBI', 'SMH']
                self.allocations = {'NVDA': 0.15, 'MSFT': 0.10, 'META': 0.08, 'XBI': 0.05}
                self.sector_breakdown = {'Technology': 0.45, 'Healthcare': 0.15}
        
        mock_portfolio = MockPortfolio()
        mock_stocks = {'NVDA': {'sector': 'Technology', 'score': 95.0}}
        mock_etfs = {'XBI': {'theme': 'Biotech', 'score': 82.0}}
        
        output_file = "data/newsletters/quarterly/test_quarterly_output.md"
        path = gen.generate_quarterly_newsletter(
            portfolio=mock_portfolio,
            top_stocks=mock_stocks,
            top_etfs=mock_etfs,
            output_path=output_file
        )
        
        if Path(path).exists():
            logger.info("✓ Quarterly Newsletter successfully generated!")
            logger.info(f"✓ Location: {path}")
        else:
            logger.error("❌ Quarterly Newsletter was not created.")
            
    except Exception as e:
        logger.error(f"❌ Quarterly Test failed: {e}")

if __name__ == "__main__":
    test_full_newsletter()
    test_quarterly_newsletter()
