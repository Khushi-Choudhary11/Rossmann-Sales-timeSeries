# Retail Sales Forecasting (Rossmann, Store 1)

Daily and weekly sales forecasting for a single Rossmann drugstore. Statistical baselines (ARIMA, SARIMA, seasonal naive) are compared with Prophet using calendar regressors, followed by a test-period forecast and a simple anomaly detector.

## Problem

Given historical daily sales for a retail store, forecast future sales so that inventory and staffing can be planned. The data has structural zero-sales days (store closed on Sundays), promotion-driven spikes and yearly seasonality.

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle). The raw files are **not included** in this repo (competition rules and size); download `train.csv`, `test.csv` and `store.csv` from Kaggle into `data/`. Files used in `data/`:

| File | Rows | Description |
|---|---|---|
| `train.csv` | 1,017,209 | Daily sales for 1,115 stores, Jan 2013 to Jul 2015 |
| `test.csv` | 41,088 | Future dates to predict (Aug to Sep 2015) |
| `store.csv` | 1,115 | Store metadata (type, assortment, competition, Promo2) |
| `processed_store1.csv` | 942 | Store 1 as `ds`, `y` plus `Open`, `Promo`, `StateHoliday`, `SchoolHoliday` (written by notebook 1; the only data file kept in the repo) |

The analysis focuses on **Store 1** (942 daily rows, 161 of them closed days).

## Project structure

```
TimeSeriesProject/
  data/                         raw Kaggle CSVs (download yourself) and processed_store1.csv
  notebooks/
    01_eda.ipynb                loading, merging, EDA, decomposition, ADF test
    02_forecasting.ipynb        baselines, Prophet + regressors, backtest, weekly model,
                                test forecast, anomaly detection
  outputs/
    forecast_store1_test.csv    48-day forecast for the test period (+ plot)
    anomalies_store1.csv        flagged days in the held-out window (+ plot)
    prophet_store1.json         final fitted Prophet model
  requirements.txt              direct dependencies
  requirements.freeze.txt       full pip freeze from the original (WSL) environment
  LICENSE                       MIT
```

## Approach

1. **EDA** (`01_eda.ipynb`): merge train with store metadata, parse and sort dates, select Store 1, plot daily sales, `seasonal_decompose` (additive, period 7), promo vs non-promo comparison, weekday averages, ADF test.
2. **Split**: time-based 80/20 split (no shuffling). Train is the first 753 days, validation the last 189.
3. **Baselines**: ARIMA(1,1,1), SARIMA(1,1,1)(1,1,1,7), seasonal naive (same weekday last week).
4. **Prophet**: first with weekly and yearly seasonality only, then with `Open`, `Promo`, `StateHoliday`, `SchoolHoliday` as regressors and a closed-day mask (forecast forced to 0 when `Open == 0`). The regressors are known in advance and provided in `test.csv`, so this is not leakage.
5. **Rolling-origin backtest**: 5 consecutive 48-day windows (the length of the real test horizon).
6. **Weekly view**: daily sales resampled to weekly totals, Prophet with yearly seasonality only.
7. **Final model**: Prophet + regressors + mask refit on all history, forecast for the `test.csv` dates of Store 1.
8. **Anomaly detection**: residual-based flags on the held-out window.

## Key findings

- Strong weekly seasonality. Sunday sales are exactly 0 (store closed); Monday and Saturday are the strongest open days.
- Promotion days have higher sales than non-promo days.
- ADF test: statistic -4.37, p = 0.0003, so the series is stationary at the 5% level (the strong weekly cycle and the zeros are still present, so seasonal terms are needed regardless).
- Most of the gain over plain ARIMA comes from modelling seasonality: SARIMA matches Prophet without regressors. The real improvement comes from telling the model about `Open` and `Promo`.

## Results

### Validation set (last 189 days, daily)

| Model | MAE | RMSE | SMAPE % | RMSPE (open days) |
|---|---|---|---|---|
| ARIMA(1,1,1) | 1321.1 | 1892.0 | 48.8 | 0.184 |
| Seasonal naive (t-7) | 1198.1 | 1685.4 | 32.6 | 0.356 |
| SARIMA(1,1,1)(1,1,1,7) | 696.8 | 1102.0 | 47.0 | 0.189 |
| Prophet (no regressors) | 701.4 | 1083.4 | 46.7 | 0.184 |
| Prophet + regressors | 433.7 | 561.2 | 43.6 | 0.139 |
| **Prophet + regressors + closed-day mask** | **400.7** | **550.0** | **8.7** | **0.139** |

SMAPE is high for the unmasked models because the true value on closed days is 0, so any non-zero prediction scores the maximum error on those days.

### Rolling backtest (5 windows of 48 days, mean and std across folds)

| Model | MAE | RMSPE (open days) |
|---|---|---|
| Prophet + regressors + mask | 443 (std 147) | 0.150 |
| SARIMA | 864 (std 368) | 0.209 |
| Seasonal naive | 1283 (std 404) | 0.358 |

### Weekly totals

Prophet on weekly totals: MAE 2984, RMSE 3416, which is 11.5% of the mean weekly sales (26,026) in the validation window. A "same as last week" baseline has MAE 5684. Weekly errors are on a different scale from the daily table and are not comparable to it.

### Anomaly detection

Noise level is estimated from training residuals (MAD-based sigma = 512). A held-out open day is flagged when its residual exceeds 2.5 sigma (about 1279). 6 of 156 open days (3.8%) were flagged: three sales drops in Feb and Mar 2015, and three spikes on 1, 2 and 4 April 2015 just before Easter. Prophet's own 95% interval flagged about 17% of days, so it was not used.

## Status

| Item | State |
|---|---|
| EDA | Done |
| Baselines (ARIMA, SARIMA, seasonal naive) | Done |
| Prophet with regressors and closed-day mask | Done |
| Rolling-origin backtest | Done |
| Weekly model | Done |
| Test-period forecast for Store 1 | Done (`outputs/forecast_store1_test.csv`) |
| Saved model | Done (`outputs/prophet_store1.json`) |
| Anomaly detection | Done (single store, held-out window) |
| All-store (global) model | Not done |
| Deployment, API, tests, CI | Not started |

## Limitations

- Only one store is modelled. A global model over all 1,115 stores (for example gradient boosting with lag features and store metadata) would match the real Kaggle task.
- Holiday handling is a single `StateHoliday` flag; Easter and Christmas are not distinguished, and the pre-Easter spike is flagged as an anomaly instead of being modelled.
- Forecast has no ground truth: `test.csv` has no `Sales` column, so test-period accuracy cannot be measured here.
- Anomaly threshold (2.5 sigma) is a judgement call and was tuned on one window.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter lab
```

Put the Kaggle CSVs in `data/`, then run `notebooks/01_eda.ipynb` first (it writes `data/processed_store1.csv`), then `notebooks/02_forecasting.ipynb` (it writes the files in `outputs/`). Both notebooks use relative paths and are meant to be run from the `notebooks/` directory.
