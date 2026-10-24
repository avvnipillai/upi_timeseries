"""
UPI forecasting WITH festivals (builds on upi_simple.py).

New compared with upi_simple.py:
  1. Volume PER DAY  -> removes the "short February" effect.
  2. Festival flags (Diwali, Eid) added to SARIMA  -> "SARIMAX".
  3. Festival effect table (% change in a festival month) with confidence range.
  4. SARIMA vs SARIMAX vs Holt-Winters vs Prophet, scored on 2026.
  5. Future forecast that knows the festival dates (Diwali = 8 Nov 2026).
  6. Lists which months were missing and filled in.

Run real data:   python upi_festivals.py
Run fake data:   python upi_festivals.py --sample
"""
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ------------------------------------------------------------ SETTINGS (edit here)
ROOT = Path(__file__).parent
OUT = ROOT / "outputs_festivals"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "data" / "upi_monthly"
LAST_MONTH = "2026-09"
TRAIN_END = "2025-12"
FUTURE = 6                       # 6 months = Oct 2026 .. Mar 2027 (festival dates below cover this)

# Only festivals whose MONTH CHANGES from year to year are useful as flags.
# Holi (always March), Dussehra (always Oct), Christmas (always Dec) repeat in the same month every
# year, so the seasonal part of SARIMA already captures them. Diwali moves between Oct and Nov,
# and Eid moves between Mar and May, so a flag can measure them.
# (Check these dates yourself before submitting.)
FESTIVALS = {
    "diwali": ["2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08"],
    "eid":    ["2021-05-14", "2022-05-03", "2023-04-22", "2024-04-11", "2025-03-31", "2026-03-21",
               "2027-03-10"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--sample", action="store_true")
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
        if "Volume (In Mn.)" in df.columns:
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
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    missing = full.difference(s.index)
    s = s.reindex(full).interpolate()                # fill missing months with a straight line
    return s, missing


y, missing = load()
print(f"\nLoaded {len(y)} months: {y.index[0]:%Y-%m} to {y.index[-1]:%Y-%m}")
if len(missing):
    print(f"WARNING: {len(missing)} missing month(s) were filled by a straight line:",
          list(missing.strftime("%Y-%m")))
    pd.Series(missing.strftime("%Y-%m"), name="filled_month").to_csv(OUT / "filled_months.csv", index=False)
else:
    print("No missing months.")

# ------------------------------------------------------------ 2. VOLUME PER DAY (calendar fix)
# Months have 28-31 days. Dividing by days removes that effect, so what remains is real behaviour.
y_day = y / y.index.days_in_month
ly = np.log(y_day)                                   # log, because UPI grows by percentages


# ------------------------------------------------------------ 3. FESTIVAL FLAGS
def festival_flags(index):
    """1 if the festival falls inside that month, else 0. One column per festival."""
    flags = pd.DataFrame(0.0, index=index, columns=list(FESTIVALS))
    for name, dates in FESTIVALS.items():
        months = {pd.Timestamp(d).to_period("M").to_timestamp() for d in dates}
        flags[name] = index.isin(months).astype(float)
    return flags


X = festival_flags(y.index)
print("\nFestival months found in the data:")
for c in X.columns:
    print(f"  {c}: {list(X.index[X[c] == 1].strftime('%Y-%m'))}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_day * 1000, color="tab:blue")
ax.scatter(X.index[X.diwali == 1], (y_day * 1000)[X.diwali == 1], color="orange", s=60, zorder=3, label="Diwali month")
ax.scatter(X.index[X.eid == 1], (y_day * 1000)[X.eid == 1], color="green", s=60, zorder=3, label="Eid month")
ax.set_title("UPI volume per day (million) with festival months marked")
ax.set_ylabel("million per day")
ax.legend()
fig.savefig(OUT / "1_per_day_with_festivals.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------ 4. ADF + ACF/PACF
d1 = ly.diff().dropna()
print("\nADF p-value, log per-day series:", round(adfuller(ly)[1], 4))
print("ADF p-value, 1st difference:    ", round(adfuller(d1)[1], 4))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(d1, lags=24, ax=ax[0])
plot_pacf(d1, lags=24, ax=ax[1], method="ywm")
ax[0].set_title("ACF (differenced log per-day volume)")
ax[1].set_title("PACF (differenced log per-day volume)")
fig.savefig(OUT / "2_acf_pacf.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------ 5. TRAIN / TEST SPLIT
cut = pd.Timestamp(TRAIN_END + "-01")
train, test = ly[ly.index <= cut], ly[ly.index > cut]
Xtr, Xte = X.loc[train.index], X.loc[test.index]
h = len(test)
days_te = test.index.days_in_month
actual = y.loc[test.index]
print(f"\nTrain: {len(train)} months, Test: {h} months")

# Keep only flags that actually vary inside the training period
use = [c for c in X.columns if Xtr[c].nunique() > 1]
print("Festival flags used in the model:", use)


def to_volume(log_per_day_forecast):
    """log per-day forecast -> monthly volume in billion."""
    return pd.Series(np.exp(np.asarray(log_per_day_forecast)) * days_te, index=test.index)


# ------------------------------------------------------------ 6. MODELS
forecasts = {}

# (a) SARIMA - no festivals
sar = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
forecasts["SARIMA"] = to_volume(sar.forecast(h))

# (b) SARIMAX - SARIMA + festival flags
sarx = SARIMAX(train, exog=Xtr[use], order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
forecasts["SARIMAX+festivals"] = to_volume(sarx.forecast(h, exog=Xte[use]))
print(f"\nAIC  SARIMA: {sar.aic:.1f}   SARIMAX+festivals: {sarx.aic:.1f}   (lower is better)")

# (c) Holt-Winters
hw = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit()
forecasts["HoltWinters"] = to_volume(hw.forecast(h))

# (d) Prophet (optional)
try:
    from prophet import Prophet
    pm = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    pm.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    forecasts["Prophet"] = to_volume(pm.predict(pd.DataFrame({"ds": test.index}))["yhat"].values)
except Exception as e:
    print("Prophet skipped (pip install prophet):", e)

# ------------------------------------------------------------ 7. COMPARE (MAPE on monthly volume)
print("\nMODEL COMPARISON on test period (lower MAPE is better)")
scores = {n: float(np.mean(np.abs(f.values / actual.values - 1)) * 100) for n, f in forecasts.items()}
for n, v in scores.items():
    print(f"  {n:18s} MAPE = {v:.2f}%")
pd.Series(scores, name="MAPE_%").round(2).to_csv(OUT / "model_scores.csv")
print("Best model:", min(scores, key=scores.get))

plt.figure(figsize=(10, 4.5))
plt.plot(y.index[-24:], y.values[-24:], "k", label="Actual")
for n, f in forecasts.items():
    plt.plot(f.index, f.values, "--", label=n)
plt.axvline(cut, color="grey", ls=":")
plt.title("Test-period forecasts vs actual")
plt.ylabel("UPI volume (billion)")
plt.legend()
plt.savefig(OUT / "3_model_comparison.png", dpi=120)
plt.close()

# ------------------------------------------------------------ 8. FESTIVAL EFFECTS (the key result)
# Refit on ALL data. Coefficient b on log scale -> % change = (e^b - 1) x 100.
full_fit = SARIMAX(ly, exog=X[use], order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
ci = full_fit.conf_int()
rows = []
for c in use:
    b = full_fit.params[c]
    rows.append({"festival": c,
                 "effect_%": round(np.expm1(b) * 100, 2),
                 "low95_%": round(np.expm1(ci.loc[c, 0]) * 100, 2),
                 "high95_%": round(np.expm1(ci.loc[c, 1]) * 100, 2),
                 "p_value": round(full_fit.pvalues[c], 3)})
eff = pd.DataFrame(rows)
eff.to_csv(OUT / "festival_effects.csv", index=False)
print("\nFESTIVAL EFFECTS (change in per-day volume in a festival month, other things equal)")
print(eff.to_string(index=False))
print("If the 95% range includes 0, or p_value > 0.05, the effect is NOT statistically clear.")

# ------------------------------------------------------------ 9. FUTURE FORECAST
fut_idx = pd.date_range(ly.index[-1] + pd.offsets.MonthBegin(1), periods=FUTURE, freq="MS")
Xf = festival_flags(fut_idx)[use]
fut_days = fut_idx.days_in_month
fc = full_fit.get_forecast(FUTURE, exog=Xf)
cif = np.exp(fc.conf_int(alpha=0.05))
res = pd.DataFrame({"forecast_bn": np.exp(fc.predicted_mean) * fut_days,
                    "lower95": cif.iloc[:, 0].values * fut_days,
                    "upper95": cif.iloc[:, 1].values * fut_days}, index=fut_idx)
hw_full = ExponentialSmoothing(ly, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit()
res["holtwinters_bn"] = np.exp(hw_full.forecast(FUTURE)).values * fut_days
res = res.round(2)
res.to_csv(OUT / "future_forecast.csv")
print("\nFUTURE FORECAST (billion). Festival months in the future window:")
print(Xf[Xf.sum(axis=1) > 0].astype(int).to_string())
print(res)

plt.figure(figsize=(10, 4.5))
plt.plot(y.index[-24:], y.values[-24:], "k", label="Actual")
plt.plot(res.index, res.forecast_bn, label="SARIMAX + festivals")
plt.fill_between(res.index, res.lower95, res.upper95, alpha=0.2, label="95% interval")
plt.plot(res.index, res.holtwinters_bn, "--", label="Holt-Winters")
plt.title("UPI volume forecast (festival-aware)")
plt.ylabel("billion")
plt.legend()
plt.savefig(OUT / "4_future_forecast.png", dpi=120)
plt.close()
print(f"\nDone. Everything is in the {OUT.name}/ folder.")
