# Code Change Recommendations (Actionable)

This is a concrete engineering change list focused on reducing code sprawl, tightening contracts, and improving sellable reliability.

## Immediate fixes (implemented in this PR)

1. **`src/data/fetcher.py`**
   - Changed `YahooFinanceFetcher.__init__` default `retry_delay` from `5` to `2` to align with documented behavior and tests.
   - Standardized `fetch_price_history()` output by adding an explicit `Date` column while retaining `DatetimeIndex` for compatibility.

## High-value refactors to do next

### 1) Entrypoint consolidation

- **Add** `src/cli.py`:
  - `alphaintel scan-daily` -> wraps `run_optimized_scan.py`
  - `alphaintel scan-quarterly` -> wraps `run_quarterly_compounder_scan.py`
  - `alphaintel report-ai` -> wraps `run_ai_report.py`
- **Deprecate/move** top-level helper scripts into `tools/`:
  - `demo.py`, `screening_demo.py`, `quality_check_demo.py`, `check_positions.py`, `manage_positions.py`
- **Remove** duplicate production runners after parity confirmation:
  - Keep one daily runner (`run_optimized_scan.py`) as canonical.

### 2) Data contracts + typing

- **Add** `src/contracts/schemas.py` (Pydantic or dataclasses):
  - `PriceHistorySchema`
  - `SignalSchema`
  - `PortfolioActionSchema`
- **Update**:
  - `src/screening/signal_engine.py` to emit typed outputs.
  - `src/reporting/newsletter_generator.py` and `src/reporting/portfolio_manager.py` to consume typed payloads.

### 3) Test lane separation

- **Keep in `tests/`** pure unit tests only.
- **Move** network/manual test scripts from root into `tools/integration/`.
- **Add markers**:
  - `@pytest.mark.unit`
  - `@pytest.mark.integration`
  - `@pytest.mark.smoke`
- **Add CI jobs** by lane in `.github/workflows/`.

### 4) Runtime observability

- **Add** `src/observability/metrics.py`:
  - scan runtime
  - API error rates by provider
  - candidate counts / signal conversion rates
- **Update** `run_optimized_scan.py` and `run_quarterly_compounder_scan.py` to emit JSON metrics artifacts.

### 5) Packaging for sale

- **Add** `PRODUCT_OFFERING.md` with Starter/Pro/Enterprise scope mapped to modules.
- **Add** `API_CONTRACT.md` documenting stable payload versions.
- **Add** `CHANGELOG.md` + semantic version tags for compatibility trust.

## Suggested remove/archive candidates

- Archive root test scripts after migration to `tools/integration/`:
  - `test_fmp_*.py`, `test_sec_debug.py`, `test_email_full.py`, etc.
- Keep only one quickstart path in `README.md` for buyer clarity.

## Acceptance criteria for technical readiness

- One canonical CLI path for daily + quarterly flows.
- Green unit suite; integration suite allowed to be environment-key gated.
- Stable versioned output contracts with schema validation in CI.
- Workflows publish metrics + report artifacts each run.
