# Product Roadmap

This roadmap defines phased deliverables for AlphaIntelligence product maturity, with explicit ownership and measurable KPIs.

## Timeline at a Glance

- **Phase A (2–4 weeks):** Newsletter reliability hardening and operational visibility.
- **Phase B (4–8 weeks):** User-facing web dashboard for core product surfaces.
- **Phase C (8–12 weeks):** Strategy lab for quantitative diagnostics and research iteration.
- **Phase D (post-Phase C):** Alerting and personalization for sticky, daily user value.

---

## Phase A (2–4 weeks): Reliability & Observability Foundation

### Deliverables
1. **Newsletter reliability hardening**
   - Idempotent newsletter generation pipeline.
   - Retry + backoff for transient upstream/data failures.
   - Delivery fallback path (queue + replay tooling).
2. **Source health panel**
   - Per-source status: healthy/degraded/down.
   - Freshness timestamps and lag indicators.
   - Source-level error trend summary.
3. **Cache/error observability**
   - Cache hit/miss dashboards by component.
   - Error budget view (5xx, timeout, parse failures).
   - Traceability from failed newsletter sections to source/API root cause.

### Ownership
- **Platform/Backend:** reliability hardening, queue/retry systems.
- **Data Engineering:** source health definitions, freshness checks.
- **SRE/Infra:** observability stack, alert thresholds, incident playbooks.

### KPIs
- **Endpoint success rate:** ≥ 99.5% for critical newsletter/rendering APIs.
- **Newsletter uniqueness score:** ≥ 0.80 (semantic novelty versus prior 7-day sends).
- **Newsletter send failure rate:** ≤ 0.5% daily.
- **Mean time to detect source degradation:** < 10 minutes.

---

## Phase B (4–8 weeks): Web Dashboard

> Primary implementation target: **`dashboard/`** (preferred) or **`src/web/`** depending on existing architecture.

### Deliverables
Build and ship a web dashboard with the following pages:
1. **Market snapshot**
   - Global index/sector summary, volatility regime, top movers.
2. **Signal feed**
   - Time-ordered signals, confidence bands, rationale snippets.
3. **Portfolio PnL**
   - Daily/MTD/YTD PnL, contribution by strategy/factor.
4. **Macro regime**
   - Regime classifier output and key macro drivers.
5. **API health**
   - Public/internal endpoint status, latency percentiles, incident log.

### Ownership
- **Frontend/Product Engineering:** dashboard shell, page implementation, UX consistency.
- **Backend/API:** data contracts and endpoint performance.
- **Design/Product:** information hierarchy and user workflows.

### KPIs
- **Open rate (newsletter-to-dashboard traffic):** ≥ 35% of recipients open linked dashboard session weekly.
- **Click rate (dashboard CTAs):** ≥ 12% CTR on top-level actions (signal detail, portfolio drill-down, macro context).
- **Endpoint success rate (dashboard APIs):** ≥ 99.7%.
- **p95 dashboard page load:** < 2.5 seconds on core pages.

---

## Phase C (8–12 weeks): Strategy Lab

### Deliverables
1. **Factor exposure**
   - Rolling factor betas and exposure drift monitoring.
2. **Signal attribution**
   - Performance contribution by signal family and horizon.
3. **Walk-forward stats**
   - Forward-only evaluation windows with stability metrics.
4. **Drawdown diagnostics**
   - Peak-to-trough decomposition by market regime and factor shock.
5. **Benchmark-relative alpha decomposition**
   - Excess return explained by timing, selection, and residual alpha.

### Ownership
- **Quant Research:** model diagnostics methodology and interpretation.
- **Data Science/Analytics:** attribution pipelines and statistical validation.
- **Backend Platform:** compute orchestration and storage for research artifacts.

### KPIs
- **Signal hit rate:** target uplift of +5–10% versus baseline by end of phase.
- **Backtest-to-live drift:** ≤ 15% deviation in key strategy metrics.
- **Attribution coverage:** ≥ 95% of signals mapped to explainable components.
- **Drawdown root-cause turnaround:** < 1 business day for material drawdowns.

---

## Phase D: Alerting + Personalization

### Deliverables
1. **User watchlists**
   - Saved assets/sectors/themes with priority ranking.
2. **Trigger-based briefs**
   - Auto-generated briefs on user-defined threshold events.
3. **Intra-day digest**
   - Scheduled summaries during market hours.
4. **“What changed since open” updates**
   - Delta-focused update stream since market open baseline.

### Ownership
- **Product + Growth:** personalization strategy and engagement loops.
- **Backend/ML:** trigger evaluation engine and ranking/personalization logic.
- **Content/Research:** brief templates, narrative quality assurance.

### KPIs
- **Open rate (personalized alerts):** ≥ 45%.
- **Click rate (alert payload links):** ≥ 18%.
- **Signal hit rate (alerts tied to signals):** maintain or improve Phase C hit rate.
- **Newsletter uniqueness score (digest/brief content):** ≥ 0.85.
- **Endpoint success rate (alerting services):** ≥ 99.9%.

---

## Cross-Phase KPI Definitions

- **Open rate:** Percentage of delivered emails/alerts opened within 24h.
- **Click rate:** Percentage of opened messages that generate at least one click.
- **Signal hit rate:** Percentage of emitted actionable signals that realize expected directional move within configured horizon.
- **Newsletter uniqueness score:** Semantic distance score between current and recent issues to minimize repetitive content.
- **Endpoint success rate:** Ratio of successful (2xx/3xx expected) responses over total requests for scoped endpoints.

## Governance Cadence

- Weekly roadmap review (Eng, Product, Quant, SRE).
- Bi-weekly KPI checkpoint with corrective action owners.
- End-of-phase retrospective with carry-forward risk log.
