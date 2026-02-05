"""
Long-Term Compounder Identification System

A production-grade framework for identifying elite 5-10 year compounders
through fundamental dominance, business quality, and multi-year trend persistence.

Modules:
- metrics: ROIC, WACC, CAGR calculations
- data_fetcher: LongTermFundamentalsFetcher for 5-year data
- compounder_engine: Multi-year quality scoring
- regime_classifier: Long-cycle regime classification
- moat_scoring: Quantifiable business moat proxies
"""

__version__ = "1.0.0"
__all__ = [
    "metrics",
    "data_fetcher",
    "compounder_engine",
    "regime_classifier",
    "moat_scoring",
]
