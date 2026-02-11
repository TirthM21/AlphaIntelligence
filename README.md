# 🏦 AlphaIntelligence

**A systematic trading and portfolio intelligence platform for momentum screening, long-term compounder selection, and automated reporting.**

AlphaIntelligence combines data engineering, quant scoring, and reporting automation into two production workflows:

- **Daily Momentum Engine** (short-term swing opportunities)
- **Quarterly Compounder Engine** (long-term ownership construction)

---

## ✨ What you can do

- Scan US equities for trend-template + phase-based momentum signals.
- Score long-term compounders and thematic ETFs for quarterly allocation.
- Generate AI-assisted investment commentary and newsletters.
- Run position/risk/portfolio operations in scheduled workflows.
- Backtest simple strategies and generate reproducible reports.

---

## 🧱 Architecture at a glance

### Core domains

- `src/data/` → data providers, fetching, fallback, cache, quality checks
- `src/screening/` → indicators, phase logic, signal scoring, batch processing
- `src/long_term/` → compounder models, ETF scoring, portfolio construction
- `src/reporting/` → newsletters and portfolio reports
- `src/notifications/` → email/slack and scheduler workflows
- `src/contracts/` → typed output contracts for stable payloads
- `src/observability/` → provider telemetry and reliability metrics
- `src/backtest/` → strategy backtest engine, metrics, and report generation

---

## 🚀 Quick start

### 1) Install

```bash
pip install -r requirements.txt
```

(Optional, for CLI install):

```bash
pip install -e .
```

### 2) Configure environment

Create `.env` with at least:

```env
FMP_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
FREE_LLM_API_KEY=your_key_here
DATABASE_URL=your_db_url
```

### 3) Run workflows (script or CLI)

| Goal | Script Command | CLI Command |
| :--- | :--- | :--- |
| Daily market scan | `python run_optimized_scan.py --limit 50 --use-fmp` | `alphaintel scan-daily -- --limit 50 --use-fmp` |
| Quarterly compounder scan | `python run_quarterly_compounder_scan.py --log-level INFO` | `alphaintel scan-quarterly -- --log-level INFO` |
| AI deep-dive report | `python run_ai_report.py` | `alphaintel report-ai` |
| Backtest | `python run_backtest.py --ticker AAPL --period 5y` | `alphaintel backtest -- --ticker AAPL --period 5y` |

---

## 📊 Backtesting

A built-in SMA crossover backtest pipeline is now available:

```bash
alphaintel backtest -- --ticker NVDA --period 5y --short-window 50 --long-window 200
```

Outputs:

- Console summary (return, drawdown, volatility, win rate)
- Markdown report at `data/reports/backtest_report.md` (default)

---

## 🧪 Testing

Run full test suite:

```bash
pytest -q tests
```

Compile sanity check:

```bash
python -m compileall src
```

---

## 🤖 GitHub Actions workflows

This repository includes automation for:

- daily scan + newsletter
- daily portfolio operations
- quarterly compounder scan
- data persistence jobs
- CI/unit testing (added)

See `.github/workflows/`.

---

## 📚 Documentation index

- `SYSTEM_OVERVIEW.md` → technical overview
- `CODEBASE_DOCUMENTATION.md` → file-by-file repo map
- `BUSINESS_READINESS_PLAN.md` → commercialization hardening roadmap
- `CODE_CHANGE_RECOMMENDATIONS.md` → immediate engineering cleanup plan
- `FEATURE_BACKLOG_CODEWISE.md` → prioritized code feature backlog
- `FMP_STABLE_API.md` → FMP integration reference

---

## ⚠️ Disclaimer

AlphaIntelligence is research and decision-support software, not financial advice. Validate all outputs and apply your own risk controls.
