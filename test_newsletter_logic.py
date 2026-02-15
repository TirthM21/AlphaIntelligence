import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock
from src.reporting.newsletter_generator import NewsletterGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_fast_test():
    """Fast test of the newsletter generator with fully mocked API dependencies."""
    logger.info("Starting FAST Newsletter Verification...")
    
    # Initialize generator
    gen = NewsletterGenerator()
    
    # MOCK ALL API FETCHERS
    gen.finnhub = MagicMock()
    gen.marketaux = MagicMock()
    gen.fetcher = MagicMock()
    gen.ai_agent = MagicMock()
    gen.visualizer = MagicMock()
    
    # Mock return values for sections
    gen.marketaux.fetch_trending_entities.return_value = [
        {'key': 'NVDA', 'sentiment_avg': 0.85, 'total_documents': 42},
        {'key': 'TSLA', 'sentiment_avg': -0.12, 'total_documents': 30}
    ]
    gen.marketaux.fetch_market_news.return_value = [
        {'title': 'Market Surge', 'url': 'http://test.com', 'source': 'Reuters', 'snippet': 'Snippet...'}
    ]
    gen.fetcher.fmp_fetcher.fetch_economic_calendar.return_value = [
        {'date': '2026-02-16', 'event': 'Fed Meeting', 'impact': 'High', 'country': 'US'}
    ]
    gen.ai_agent.generate_qotd.return_value = {
        'question': 'How often does market crash?',
        'answer': 'Rarely.',
        'insight': 'Buy the dip.'
    }
    gen.ai_agent._call_ai.return_value = "Mocked institutional insight."
    gen.ai_agent.generate_commentary.return_value = "Mocked stock thesis."
    gen.ai_agent.enhance_newsletter = lambda x: x # Skip AI optimization for test speed
    
    gen.visualizer.generate_sector_chart.return_value = "data/charts/sector_performance.png"
    gen.visualizer.generate_cap_comparison.return_value = "data/charts/cap_comparison.png"
    
    # Create mock charts so file existence check passes
    Path("data/charts").mkdir(parents=True, exist_ok=True)
    Path("data/charts/sector_performance.png").touch()
    Path("data/charts/cap_comparison.png").touch()

    # Data for call
    mock_market_status = {
        'spy': {'trend': 'UPTREND', 'current_price': 500.25},
        'breadth': {'advance_decline_ratio': 1.5, 'percent_above_200sma': 65.0},
        'sectors': [{'sector': 'Tech', 'change': 1.2}],
        'caps': {'Large': 0.5, 'Mid': 0.2, 'Small': -0.1}
    }
    
    mock_buys = [
        {'ticker': 'AAPL', 'score': 90, 'current_price': 200, 'fundamental_snapshot': 'Strong'}
    ]

    output_file = "test_newsletter_fast.md"
    logger.info("Generating Fast Test Newsletter...")
    
    path = gen.generate_newsletter(
        market_status=mock_market_status,
        top_buys=mock_buys,
        top_sells=[],
        output_path=output_file
    )
    
    if Path(path).exists():
        logger.info(f"✓ Success! Generated {path}")
        logger.info("\n--- PREVIEW ---\n")
        print(Path(path).read_text(encoding='utf-8')[:1000])
    else:
        logger.error("❌ Failed to generate newsletter.")

if __name__ == "__main__":
    run_fast_test()
