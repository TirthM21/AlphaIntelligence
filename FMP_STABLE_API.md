# FMP Stable API Reference (2026)

## Authorization
* **Header**: `apikey: YOUR_KEY`
* **Query**: `?apikey=YOUR_KEY`

## Stable Endpoints (New)
| Feature | Stable Endpoint |
| :--- | :--- |
| **Search** | `https://financialmodelingprep.com/stable/search-symbol?query=AAPL` |
| **Profile** | `https://financialmodelingprep.com/stable/profile?symbol=AAPL` |
| **Quote** | `https://financialmodelingprep.com/stable/quote?symbol=AAPL` |
| **Income Statement** | `https://financialmodelingprep.com/stable/income-statement?symbol=AAPL` |
| **Balance Sheet** | `https://financialmodelingprep.com/stable/balance-sheet-statement?symbol=AAPL` |
| **Cash Flow** | `https://financialmodelingprep.com/stable/cash-flow-statement?symbol=AAPL` |
| **Key Metrics** | `https://financialmodelingprep.com/stable/key-metrics?symbol=AAPL` |
| **Ratios** | `https://financialmodelingprep.com/stable/ratios?symbol=AAPL` |
| **Market Cap** | `https://financialmodelingprep.com/stable/market-capitalization?symbol=AAPL` |
| **Earnings** | `https://financialmodelingprep.com/stable/earnings?symbol=AAPL` |
| **Dividends** | `https://financialmodelingprep.com/stable/dividends?symbol=AAPL` |

---
*Note: The new stable API prefers `?symbol=TICKER` instead of path-based `/TICKER`.*
