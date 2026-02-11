# AlphaIntelligence Business Readiness Plan

This document translates technical findings into a practical productization roadmap to make the project easier to sell, operate, and scale.

---

## Executive summary

The codebase has strong technical breadth, but business packaging is diluted by too many overlapping entrypoints, mixed testing styles, and limited product-boundary definition.

### Core finding

You currently have **many runnable scripts** at the repository root, while only a small subset is treated as production paths in automation. This creates buyer confusion and raises maintenance cost.

### Goal

Move from a "power-user research repo" to a **clean product platform** with:

1. one primary daily workflow,
2. one primary quarterly workflow,
3. a stable API/service interface,
4. measurable reliability (SLOs), and
5. clear paid packaging.

---

## What is currently hurting commercial readiness

## 1) Too many disconnected entrypoints

The repository includes numerous top-level executable scripts (`run_*`, `demo*`, `test_*`, utility scripts), but production workflows focus mostly on `run_optimized_scan.py` and `run_quarterly_compounder_scan.py`.

### Why this matters for sales

- Prospects want a single, clear product path.
- Integrators want stable interfaces, not script discovery.
- Investors/enterprise buyers see script sprawl as operational risk.

### Recommendation

- Keep only **2 first-class commands** documented for buyers:
  - `alphaintel scan-daily`
  - `alphaintel scan-quarterly`
- Move all support scripts into `tools/` and mark as internal.
- Mark demos/experimental scripts as non-production.

---

## 2) Quality contract inconsistencies (example errors)

Two concrete inconsistencies surfaced during test execution:

1. `YahooFinanceFetcher` docstring states retry delay default is 2 seconds, while constructor currently sets 5 seconds.
2. Price history interface expectation is ambiguous (`Date` as index vs explicit column), creating test/API mismatch risk.

### Why this matters for sales

Inconsistent behavior reduces trust in your product claims and increases onboarding/support overhead.

### Recommendation

- Publish an explicit **data contract** for all fetcher outputs (schema + index rules).
- Enforce that contract with tests and typed models.
- Add semantic versioning for output schema changes.

---

## 3) Mixed test strategy creates confidence gap

The project has both structured package tests (`tests/`) and many top-level ad-hoc test scripts (`test_*.py`) that resemble diagnostics/smoke tests.

### Why this matters for sales

Commercial buyers expect a CI status that maps cleanly to release quality.

### Recommendation

- Separate into three lanes:
  - **unit** (fast, deterministic, no network),
  - **integration** (network/API keys),
  - **smoke/e2e** (workflow-level acceptance).
- Gate releases on unit + selected integration tests.
- Keep manual diagnostics under `tools/diagnostics/`.

---

## 4) Product boundary is not explicit enough

Current structure is excellent for development, but less clear as a product SKU:

- Is this sold as signals feed, report generator, portfolio operating system, or API?
- Which components are GA vs experimental?

### Recommendation

Define SKUs now:

- **Starter**: daily scan + weekly report PDF/email.
- **Pro**: daily + quarterly + portfolio rebalancing alerts.
- **Enterprise/API**: webhook/API outputs + SLA + audit logs.

Then map code modules to SKU guarantees.

---

## 5) Operational hardening needed for enterprise buyers

### Gaps

- No explicit SLO dashboard/metrics doc.
- Limited runbook standardization for incidents.
- Artifact/version policy for generated reports not formalized.

### Recommendation

- Add metrics: scan success rate, median runtime, API failure rate, report generation latency.
- Create `RUNBOOK.md` with incident triage playbooks.
- Add release notes + changelog with compatibility guarantees.

---

## Priority implementation roadmap (30/60/90)

## 0-30 days: clarity and cleanup

1. Consolidate commands into a single CLI package (`alphaintel`).
2. Freeze and publish core output schemas for:
   - scan results,
   - signal payloads,
   - portfolio actions.
3. Move ad-hoc scripts into `tools/` and label stability (`experimental`, `internal`, `deprecated`).
4. Add `PRODUCT_OFFERING.md` with tier definitions and included features.

## 31-60 days: reliability and observability

1. Restructure test matrix into unit/integration/smoke.
2. Add CI gates and badges by lane.
3. Implement structured logging and KPI summary output per run.
4. Add deterministic mock-data mode for demos/sales calls.

## 61-90 days: sellable platform packaging

1. Add customer-facing API layer (REST/webhook) over scan/report outputs.
2. Add tenant/account separation and RBAC baseline.
3. Introduce versioned contracts and migration notes.
4. Create deployment options:
   - managed cloud,
   - self-hosted container,
   - enterprise private deployment.

---

## Specific technical fixes to do immediately

1. Resolve retry-delay default mismatch between documentation and constructor behavior in `YahooFinanceFetcher`.
2. Standardize price-history output format (`Date` handling) and align all callers/tests.
3. Define a single canonical daily engine (`run_optimized_scan.py`) and deprecate overlapping runners.
4. Define a single canonical quarterly engine and remove duplicated orchestration paths.
5. Add a compatibility test that validates generated report schemas across daily + quarterly flows.

---

## Commercial positioning improvements (non-code)

1. Build a public "How AlphaIntelligence makes money" page:
   - target user profile,
   - measurable pain solved,
   - pricing tiers,
   - sample ROI case studies.
2. Build a polished sample output bundle:
   - one daily signal report,
   - one quarterly ownership report,
   - one rebalancing recommendation sheet.
3. Add trust assets:
   - methodology whitepaper,
   - data-source policy,
   - model risk disclaimer,
   - uptime/SLA policy.

---

## Definition of done for “ready to sell"

You are ready for active sales when all are true:

- Buyer can run one command and get one high-quality, repeatable output.
- Output schema is versioned and documented.
- CI is green across unit + integration gates.
- You can demonstrate 30+ days of stable scan operations.
- Pricing and feature matrix are documented and unambiguous.

