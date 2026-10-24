# Simple UPI forecasting

## Run
    pip install -r requirements_simple.txt     (prophet is optional; the script skips it if missing)
    python upi_simple.py                       # your real NPCI files in data/upi_monthly/
    python upi_simple.py --sample              # fake data, just to test your setup

Charts and tables appear in `outputs/`.

## How to read ACF and PACF (plain English)
First difference the series (we use the monthly growth of log volume). Then:

- **ACF** shows how strongly each month is related to the month 1, 2, 3... steps earlier.
  Each bar is one lag. Bars outside the blue band are "significant".
- **PACF** shows the same relationship, but only the *direct* link at that lag
  (the effect of the in-between months is removed).

Rules of thumb:
| What you see | Suggests |
|---|---|
| PACF has spikes for the first p lags, then cuts off; ACF fades slowly | AR(p) -> set p |
| ACF has spikes for the first q lags, then cuts off; PACF fades slowly | MA(q) -> set q |
| Both fade slowly | mix, try ARMA(1,1) |
| Spike at lag 12 (and 24) | yearly seasonality -> use SARIMA, seasonal period 12 |
| Almost no bars outside the band | series is close to random noise after differencing |

Then in the script change `order=(p, d, q)` (d = number of differences, usually 1).
Compare models with AIC (lower is better) and with test MAPE.

## What each model is
- **ARIMA(p,d,q)**: predicts from its own past values (AR), past errors (MA) after differencing (I). No seasonality.
- **SARIMA**: ARIMA plus a repeating yearly pattern.
- **Holt-Winters**: smooths level, trend and seasonality. No orders to choose.
- **Prophet**: trend + yearly seasonality, built by Facebook. Good with little tuning.

## Limits to state in your report
- Test set is only 9 months, so MAPE comparisons are rough.
- Post-COVID recovery (2021-22) makes early growth unusual.
- Forecasts assume past patterns continue; policy changes or shocks are not modelled.
