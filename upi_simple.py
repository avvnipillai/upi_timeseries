"""
Simple UPI forecasting:
ACF/PACF -> ARIMA -> SARIMA -> Holt-Winters -> Prophet -> compare -> future forecast.

Run real data:   python upi_simple.py
Run fake data:   python upi_simple.py --sample
All charts and tables are saved in the outputs/ folder.
"""
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ------------------------------------------------------------ SETTINGS (edit here)
ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "data" / "upi_monthly"      # a folder of files OR one file, e.g. data/sample_monthly.csv
LAST_MONTH = "2026-09"                    # last complete month (later rows are dropped)
TRAIN_END = "2025-12"                     # train up to here, test after
FUTURE = 6                                # months to forecast ahead

ap = argparse.ArgumentParser()
ap.add_argument("--sample", action="store_true", help="use the fake data/sample_monthly.csv")
if ap.parse_args().sample:
    DATA = ROOT / "data" / "sample_monthly.csv"


# ------------------------------------------------------------ 1. LOAD DATA
def load():
    if DATA.is_dir():
        files = [f for f in sorted(DATA.iterdir())
                 if f.suffix.lower() in (".csv", ".xlsx", ".xls") and not f.name.startswith("~")]
    else:
        files = [DATA]
    if not files:
        raise FileNotFoundError(f"No data files found in {DATA}")
    frames = []
    for f in files:
        df = pd.read_excel(f) if f.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(f)
        df.columns = df.columns.astype(str).str.strip()
        df = df.rename(columns={"Month": "month"})
        if "Volume (In Mn.)" in df.columns:               # NPCI files: million -> billion
            vol = pd.to_numeric(df["Volume (In Mn.)"].astype(str).str.replace(",", ""),
                                errors="coerce") / 1000
            df["volume_bn"] = df["volume_bn"].combine_first(vol) if "volume_bn" in df.columns else vol
        frames.append(df[["month", "volume_bn"]])
        print(f"  read {f.name}: {len(df)} rows")
    df = pd.concat(frames)
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
    df = df.drop_duplicates("month", keep="last").sort_values("month").set_index("month")
    s = df["volume_bn"].astype(float)
    s = s[s.index <= pd.Timestamp(LAST_MONTH + "-01")]
    s = s.asfreq("MS").interpolate()                      # fills any missing month
    return s


y = load()
print(f"\nLoaded {len(y)} months: {y.index[0]:%Y-%m} to {y.index[-1]:%Y-%m}")

# We model the LOG of volume, because UPI grows by percentages, not fixed amounts.
ly = np.log(y)

# ------------------------------------------------------------ 2. LOOK AT THE DATA
plt.figure(figsize=(10, 4))
plt.plot(y)
plt.title("UPI monthly volume (billion)")
plt.ylabel("billion")
plt.savefig(OUT / "1_series.png", dpi=120)
plt.close()

# ------------------------------------------------------------ 3. STATIONARITY (ADF test)
# p-value < 0.05 means stationary. If not, we difference the series (the "I" in ARIMA).
print("\nADF p-value, log series:        ", round(adfuller(ly)[1], 4))
d1 = ly.diff().dropna()
print("ADF p-value, 1st difference:    ", round(adfuller(d1)[1], 4))

# ------------------------------------------------------------ 4. ACF and PACF
# ACF  -> helps choose q (MA order).    PACF -> helps choose p (AR order).
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(d1, lags=24, ax=ax[0])
plot_pacf(d1, lags=24, ax=ax[1], method="ywm")
ax[0].set_title("ACF (differenced log volume)")
ax[1].set_title("PACF (differenced log volume)")
fig.savefig(OUT / "2_acf_pacf.png", dpi=120)
plt.close(fig)
print("ACF/PACF saved -> outputs/2_acf_pacf.png")

# ------------------------------------------------------------ 5. TRAIN / TEST SPLIT
train = ly[:TRAIN_END]
test = ly[ly.index > pd.Timestamp(TRAIN_END + "-01")]
h = len(test)
print(f"\nTrain: {len(train)} months, Test: {h} months")

# ------------------------------------------------------------ 6. ARIMA
arima = ARIMA(train, order=(1, 1, 1)).fit()
print("\nARIMA(1,1,1) AIC:", round(arima.aic, 1))
forecasts = {"ARIMA": np.exp(arima.forecast(h))}

# ------------------------------------------------------------ 7. SARIMA (adds yearly seasonality, period 12)
sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
print("SARIMA(1,1,1)(0,1,1,12) AIC:", round(sarima.aic, 1))
forecasts["SARIMA"] = np.exp(sarima.forecast(h))

# Residual check: good residuals look like random noise
sarima.plot_diagnostics(figsize=(10, 7))
plt.savefig(OUT / "3_sarima_diagnostics.png", dpi=120)
plt.close()

# ------------------------------------------------------------ 8. HOLT-WINTERS (exponential smoothing)
# Level + damped trend + yearly seasonal pattern. No p,d,q to choose.
hw = ExponentialSmoothing(train, trend="add", damped_trend=True,
                          seasonal="add", seasonal_periods=12).fit()
forecasts["HoltWinters"] = np.exp(hw.forecast(h))

# ------------------------------------------------------------ 9. PROPHET (optional)
try:
    from prophet import Prophet
    p_df = pd.DataFrame({"ds": train.index, "y": train.values})
    pm = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    pm.fit(p_df)
    pf = pm.predict(pd.DataFrame({"ds": test.index}))
    forecasts["Prophet"] = pd.Series(np.exp(pf["yhat"].values), index=test.index)
except Exception as e:
    print("\nProphet skipped (install with: pip install prophet):", e)

# ------------------------------------------------------------ 10. COMPARE MODELS
# MAPE = average % error on the test months. Lower is better.
actual = np.exp(test)
print("\nMODEL COMPARISON on test period (lower MAPE is better)")
scores = {}
for name, f in forecasts.items():
    scores[name] = float(np.mean(np.abs(np.asarray(f) / actual.values - 1)) * 100)
    print(f"  {name:12s} MAPE = {scores[name]:.2f}%")
pd.Series(scores, name="MAPE_%").round(2).to_csv(OUT / "model_scores.csv")
print("Best model:", min(scores, key=scores.get))

plt.figure(figsize=(10, 4.5))
plt.plot(y.index[-24:], y.values[-24:], "k", label="Actual")
for name, f in forecasts.items():
    plt.plot(test.index, np.asarray(f), "--", label=name)
plt.axvline(pd.Timestamp(TRAIN_END + "-01"), color="grey", ls=":")
plt.title("Test-period forecasts vs actual")
plt.ylabel("UPI volume (billion)")
plt.legend()
plt.savefig(OUT / "4_model_comparison.png", dpi=120)
plt.close()

# ------------------------------------------------------------ 11. FUTURE FORECAST
# Uses whichever of ARIMA / SARIMA scored better (they give proper 95% intervals).
best = "ARIMA" if scores["ARIMA"] < scores["SARIMA"] else "SARIMA"
print(f"\nFuture forecast uses {best}, refit on all data")
final = (ARIMA(ly, order=(1, 1, 1)).fit() if best == "ARIMA"
         else SARIMAX(ly, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False))
fc = final.get_forecast(FUTURE)
ci = np.exp(fc.conf_int(alpha=0.05))
out = pd.DataFrame({"forecast_bn": np.exp(fc.predicted_mean),
                    "lower95": ci.iloc[:, 0], "upper95": ci.iloc[:, 1]}).round(2)
out.to_csv(OUT / "future_forecast.csv")
print(out)

plt.figure(figsize=(10, 4.5))
plt.plot(y.index[-24:], y.values[-24:], "k", label="Actual")
plt.plot(out.index, out.forecast_bn, label=f"{best} forecast")
plt.fill_between(out.index, out.lower95, out.upper95, alpha=0.2, label="95% interval")
plt.title("UPI volume forecast")
plt.ylabel("billion")
plt.legend()
plt.savefig(OUT / "5_future_forecast.png", dpi=120)
plt.close()
print("\nDone. All charts and CSVs are in the outputs/ folder.")
