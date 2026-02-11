# AlphaIntelligence Codewise Feature Backlog

A prioritized, implementation-focused feature backlog with concrete files to add/change.

## P0 (High impact, low ambiguity)

### 1) Unified CLI and command surface
**Why:** Reduce entrypoint sprawl and make product easier to run/sell.

**Add**
- `src/cli.py`
- `pyproject.toml` entrypoint: `alphaintel=src.cli:main`

**Change**
- `run_optimized_scan.py`, `run_quarterly_compounder_scan.py`, `run_ai_report.py` to expose reusable functions called by CLI.

**Acceptance**
- `alphaintel scan-daily`, `alphaintel scan-quarterly`, `alphaintel report-ai` all work.

---

### 2) Stable JSON output contracts
**Why:** Buyers/integrators need predictable payloads.

**Add**
- `src/contracts/schemas.py` (Pydantic/dataclasses)
  - `DailySignal`
  - `QuarterlyAllocation`
  - `ScanSummary`

**Change**
- `src/screening/signal_engine.py` emit schema-compliant dicts.
- `run_optimized_scan.py` write `data/daily_scans/latest.json`.
- `run_quarterly_compounder_scan.py` write `data/quarterly_reports/latest.json`.

**Acceptance**
- Validation step fails on schema drift.

---

### 3) Provider health/fallback telemetry
**Why:** Data provider reliability is a business-critical KPI.

**Add**
- `src/observability/provider_metrics.py`

**Change**
- `src/data/fmp_fetcher.py`, `src/data/finnhub_fetcher.py`, `src/data/fetcher.py`, `src/data/enhanced_fundamentals.py`
  - log provider used, latency, failure reason, fallback path.

**Acceptance**
- Each run emits provider success-rate and fallback counts.

---

## P1 (Product quality + maintainability)

### 4) Backtesting module for signal quality
**Why:** Improves sales narrative with evidence.

**Add**
- `src/backtest/engine.py`
- `src/backtest/metrics.py`
- `src/backtest/reports.py`

**Change**
- `src/screening/signal_engine.py` to export feature snapshots for replay.

**Acceptance**
- CLI command `alphaintel backtest --from YYYY-MM-DD --to YYYY-MM-DD` generates win rate, expectancy, max drawdown.

---

### 5) Portfolio risk engine (position sizing + correlation caps)
**Why:** Move from scanner to allocation product.

**Add**
- `src/risk/position_sizer.py`
- `src/risk/correlation_guard.py`
- `src/risk/exposure_limits.py`

**Change**
- `src/reporting/portfolio_manager.py` integrate risk outputs.
- `src/analysis/position_manager.py` consume risk signals for stop updates.

**Acceptance**
- Portfolio output includes per-position risk budget and concentration warnings.

---

### 6) Alert rules engine
**Why:** Buyers want actionable event-driven alerts.

**Add**
- `src/alerts/rules.py`
- `src/alerts/engine.py`
- `src/alerts/templates.py`

**Change**
- `src/notifications/email_notifier.py` and `src/notifications/slack_notifier.py`
  - support alert templates by event type.

**Acceptance**
- Alerts triggered on phase change, stop-loss breach, or earnings-risk window.

---

## P2 (Commercial readiness)

### 7) Multi-tenant project/account support
**Why:** Required for SaaS and enterprise installs.

**Add**
- `src/database/models_tenant.py`
- `src/auth/rbac.py`

**Change**
- `src/database/db_manager.py` include tenant scoping.
- all report paths namespaced by tenant.

**Acceptance**
- Two tenants can run scans without data collision.

---

### 8) HTTP API surface
**Why:** Enables integration sales.

**Add**
- `src/api/app.py` (FastAPI)
- `src/api/routes/signals.py`
- `src/api/routes/reports.py`
- `src/api/routes/health.py`

**Change**
- runners expose service layer functions used by API.

**Acceptance**
- `/health`, `/signals/latest`, `/reports/quarterly/latest` endpoints stable.

---

### 9) Deterministic demo mode
**Why:** Reliable sales demos without external outages.

**Add**
- `data/demo_fixtures/`
- `src/data/demo_provider.py`

**Change**
- `src/data/enhanced_fundamentals.py` support provider override `--demo-mode`.

**Acceptance**
- Demo run is reproducible and network-independent.

---

## Test and CI upgrades to support the above

**Add**
- `pytest.ini` markers: `unit`, `integration`, `smoke`
- `.github/workflows/tests_unit.yml`
- `.github/workflows/tests_integration.yml`

**Change**
- Move root `test_*.py` diagnostics to `tools/integration/`.

**Acceptance**
- CI clearly reports lane-specific quality gates.

---

## Suggested implementation order
1. Unified CLI
2. JSON contracts
3. Provider telemetry
4. Risk engine
5. Backtesting
6. Alerts
7. API
8. Multi-tenant support
9. Demo mode
