# LedgerLens Feature Reference

This document describes the feature groups computed by the LedgerLens feature
pipeline. For the full feature set including Benford, trade pattern, volume,
and wallet graph features, see the [README](../README.md#machine-learning-layer).

---

## Wallet Lifecycle Features (`features/wallet_lifecycle_features.py`)

Temporal behavioural signals derived from wallet age and trading activity
history. Wash trading accounts are often newly created (low age) or show
sudden bursts of activity after long dormancy (high recency gap).

All features are computed relative to the **UTC timestamp of the scoring
request** (`now` parameter) for reproducibility in backtesting. Do not use
wall-clock time.

| Feature | Type | Description |
|---|---|---|
| `wallet_age_days` | float | Days elapsed since the account was created on Stellar. |
| `days_since_first_trade` | float | Days between the wallet's first observed trade and `now`. |
| `days_since_last_trade` | float | Days between the wallet's most recent trade and `now`. High values indicate dormancy followed by a burst. |
| `active_days_ratio` | float | Ratio of days with at least one trade to `wallet_age_days`. 1.0 = active every day; values near 0 indicate sparse or burst activity. |
| `burst_score` | float | Ratio of trades in the last 24h to the daily average over the preceding 29 days. High values indicate sudden activity bursts. |

### Data source

`account_created_at` is fetched from the Stellar Horizon
[`/accounts/{id}`](https://developers.stellar.org/api/horizon/resources/accounts/object/)
endpoint and **cached per wallet with a 24-hour TTL** to avoid repeated API
calls during a scoring run.

### NaN handling policy

- If `account_created_at` is unavailable (API error, account not found), the
  features `wallet_age_days` and `active_days_ratio` are set to `float('nan')`.
  They are **not** substituted with a default value; downstream models must
  handle `NaN` via imputation or `NaN`-safe decision trees.
- `burst_score` is `NaN` when the 30-day average trade count is zero (no
  historical baseline). This prevents division by zero while preserving the
  semantic distinction between "no burst" and "no baseline".
- `days_since_first_trade` and `days_since_last_trade` are `NaN` when the
  wallet has no trades in the input DataFrame.

---

## Token Velocity Features (`features/velocity_features.py`)

Supply-relative velocity metrics that capture how rapidly the same tokens are
cycled. Wash traders often produce a token velocity ratio orders of magnitude
higher than organic trading.

| Feature | Type | Description |
|---|---|---|
| `token_velocity_1h` | float | Volume traded in the last 1h divided by circulating supply. |
| `token_velocity_24h` | float | Volume traded in the last 24h divided by circulating supply. |
| `token_velocity_7d` | float | Volume traded in the last 7d (168h) divided by circulating supply. |

### Data source

Circulating supply is fetched from the Stellar Horizon
[`/assets`](https://developers.stellar.org/api/horizon/resources/assets/)
endpoint (`amount` field) and **cached in Redis with a 1-hour TTL**.
When Redis is unavailable, an in-process dict cache is used as a fallback.
Cache refresh is synchronous on miss; callers that need non-blocking behaviour
should run `ingestion.asset_metadata_fetcher.get_asset_supply` in a thread pool.

### NaN handling policy

- If asset supply is unavailable, zero, or negative, all three velocity
  features are set to `float('nan')` and a warning is logged. No exception
  is raised.
- Supply values are validated to be a **positive finite float** before use in
  division.

### SHAP labels

Human-readable SHAP labels for the velocity features:
- `token_velocity_1h` → "Token turnover in the last 1 hour relative to supply"
- `token_velocity_24h` → "Token turnover in the last 24 hours relative to supply"
- `token_velocity_7d` → "Token turnover in the last 7 days relative to supply"
