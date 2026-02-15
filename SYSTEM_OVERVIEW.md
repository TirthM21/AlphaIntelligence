# 🏦 AlphaIntelligence Capital — System Overview

This document provides a comprehensive breakdown of the AlphaIntelligence Capital systematic hedge fund engine architecture, module responsibilities, and operational metrics.

---

## 🚀 Core Entry Points
| File | Description |
| :--- | :--- |
| `run_optimized_scan.py` | **Main Engine**. Coordinates the full market scan, signal detection, report generation, AI enhancement, and email delivery. |
| `run_ai_report.py` | **Deep-Dive Engine**. Uses the latest scan data to generate a "Quantum Intelligence" hedge fund research report. |
| `comprehensive_system_test.py` | **Diagnostic Tool**. Tests all modules (DB, AI, FMP, SEC, Scanner) to ensure system health. |

---

## 📂 System Architecture (`src/`)

### 📊 Data Acquisition (`src/data/`)
*   **`universe_fetcher.py`**: Fetches the full list of US-listed stocks (~3,800+ symbols) from NASDAQ/NYSE.
*   **`fmp_fetcher.py`**: High-fidelity data from Financial Modeling Prep. Handles quarterly metrics, margins, and "Special/Legacy" endpoint fallbacks.
*   **`sec_fetcher.py`**: Automated retrieval of official SEC 10-Q and 10-K filings for institutional verification.
*   **`enhanced_fundamentals.py`**: The "Mixture of Data" engine. Consolidates FMP, yfinance, and SEC data into a unified fundamental profile.
*   **`fundamentals_fetcher.py`**: Core logic for calculating YoY/QoQ growth, inventory signals, and margin expansion.
*   **`git_storage_fetcher.py`**: Specialized storage layer for persistent tracking of fundamental changes.

### 🔍 Screening & Strategy (`src/screening/`)
*   **`optimized_batch_processor.py`**: The parallel execution core. Manages thread pools and rate limits to achieve 10-25 TPS safely.
*   **`signal_engine.py`**: **The Brain**. Implements Minervini Trend Templates and quantitative scoring (0-100) for Buy/Sell signals.
*   **`phase_indicators.py`**: Technical analysis layer that classifies stocks into Phase 1 (Base), Phase 2 (Uptrend), Phase 3 (Top), or Phase 4 (Downtrend).
*   **`benchmark.py`**: Analyzes market regime (Risk-On/Off) via SPY trend and market breadth (Advance/Decline).

### 🤖 Artificial Intelligence (`src/ai/`)
*   **`ai_agent.py`**: Integration with High-Performance LLMs via API. Generates investment theses, enhances newsletter prose, and sanitizes complex data.

### 📈 Reporting & Notifications (`src/reporting/` & `src/notifications/`)
*   **`newsletter_generator.py`**: Compiles technicals, fundamentals, news, and AI commentary into a professional Markdown newsletter.
*   **`portfolio_manager.py`**: Generates institutional reports: Allocation Plans, Ownership Tracking, and Rebalance Actions.
*   **`email_notifier.py`**: Institutional email delivery system for sending fund research and alerts to subscribers via encrypted SMTP.

---

## 📏 Operational Metrics
*   **Market Universe**: ~3,800+ US Stocks processed per full scan.
*   **Processing Speed**: 
    *   *Aggressive*: ~15-20 TPS (5-8 minutes full scan).
    *   *Optimized/PC-Safe*: ~2-5 TPS (15-25 minutes full scan).
*   **Signal Thresholds**:
    *   **Buy**: Requires Phase 2 + Minervini Template + Fundamental Score > 70.
    *   **Sell**: Triggered by Phase 3 transitions or >20% drawdown from peak.

---

## 🛠️ Usage Quick Reference
### Running a Safe Scan (Recommended for Home PCs)
```powershell
python run_optimized_scan.py --limit 50 --workers 1 --delay 2.0 --use-fmp
```

### Running System Diagnostics
```powershell
python comprehensive_system_test.py
```

### Generating Top-Tier AI Report
```powershell
python run_ai_report.py
```

---
*Last Updated: February 15, 2026*
