# Known limitations

The following tickers are intentionally excluded from the configured ingestion universe. They do not
currently satisfy the requirement of having both canonical structured facts and substantive narrative
chunks extracted into real filing sections.

| Ticker | Resolved CIK | Limitation | Repairability |
|---|---:|---|---|
| XOM | 2115436 | The current SEC ticker dataset resolves XOM to ExxonMobil Holdings Corp. This holding entity has no exploitable 10-K in its submissions and produced neither canonical facts nor narrative chunks. | Potentially repairable through a reviewed issuer-lineage resolver that maps the market ticker to the active SEC reporting entity. This must not be guessed from a historical CIK. |
| WFC | 72971 | The primary 10-K document contains only incorporation references. The substantive Risk Factors, MD&A, financial statements, and Notes live in the separate `wfc-20251231.htm` annual-report companion, which the current fetch stage does not retrieve. | Repairable by extending filing discovery to identify, validate, fetch, and retain provenance for incorporated companion documents. This is a separate acquisition feature, not a cleaner heuristic. |
| TSLA | 1318605 | The persisted narrative fell back to `Full Filing` because the relevant Part/Item headings are not safely recognized when their visible labels are split across spans; the latest normalized filing family also includes a partial 10-K/A. | Repairable by a separately validated cross-span heading reconstruction and by selecting the base 10-K narrative independently from later amendments. Both changes require a full-universe regression gate. |

Until those repairs are implemented and validated, these tickers must not be reintroduced into
`TICKER_UNIVERSE`.

## Retained tickers with partial narrative coverage

| Ticker | CIK | Limitation | Available coverage | Repair path |
|---|---:|---|---|---|
| CVX | 93410 | MD&A is unavailable: the filing only refers to the Financial Table of Contents. This may be an incorporation-by-reference or companion-document case and has not yet been diagnosed. | Notes and Risk Factors are exploitable; 328 canonical structured facts are retained. | Repairable later through a dedicated MD&A diagnostic if needed. |
