"""
Fixed app.py — FGI Stock Predictor
- Robust FGI fallback
- MultiIndex-safe joins between stock features and FGI
- Graceful handling when data is missing
- Ready-to-run (replace existing app.py)
"""

from __future__ import annotations
import warnings, math, time, json
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from dateutil import tz

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from flask import Flask, request, jsonify, render_template

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Put your API Ninjas key here if you have one; otherwise leave as None
API_NINJAS_KEY = None  # e.g. "abc123xyz"  -> replace with your key if available

# -------------------------------------------------------------------
# Fear & Greed Index utilities (Fixed with fallback + API Ninjas)
# -------------------------------------------------------------------

CNN_FGI_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

def fetch_fgi_json(max_retries: int = 2, timeout: int = 6) -> dict:
    """
    Try CNN first (may be unreachable). If that fails, try API Ninjas sentiment.
    If all fail, return small static fallback.
    """
    headers_base = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
    }

    # Try CNN endpoint
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(CNN_FGI_URL, headers=headers_base, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if data:
                app.logger.info("✅ CNN FGI fetched")
                return data
        except Exception as e:
            app.logger.info(f"⚠️ CNN FGI attempt {attempt} failed: {e}")
            time.sleep(0.8 * attempt)

    # Try API Ninjas (approximate sentiment -> map to FGI)
    if API_NINJAS_KEY:
        try:
            api_url = "https://api.api-ninjas.com/v1/sentiment?text=stock market"
            headers = {"X-Api-Key": API_NINJAS_KEY}
            r = requests.get(api_url, headers=headers, timeout=5)
            if r.status_code == 200:
                sentiment = r.json()
                sentiment_value = sentiment.get("sentiment", "neutral")
                if sentiment_value == "positive":
                    score = 70
                elif sentiment_value == "negative":
                    score = 30
                else:
                    score = 50
                return {"historical": [{"x": int(time.time() * 1000), "y": score}]}
        except Exception as e:
            app.logger.info(f"⚠️ API Ninjas fallback failed: {e}")

    # Final static fallback (neutral-ish)
    app.logger.info("⚠️ Using static FGI fallback (neutral)")
    return {"historical": [{"x": int(time.time() * 1000), "y": 50}]}


def fgi_to_dataframe(fgi_json: dict) -> pd.DataFrame:
    hist = None
    if isinstance(fgi_json, dict):
        if "fear_and_greed_historical" in fgi_json and "data" in fgi_json["fear_and_greed_historical"]:
            hist = fgi_json["fear_and_greed_historical"]["data"]
        elif "fear_and_greed" in fgi_json and "historical" in fgi_json["fear_and_greed"]:
            hist = fgi_json["fear_and_greed"]["historical"]
        elif "historical" in fgi_json:
            hist = fgi_json["historical"]
        else:
            for k, v in fgi_json.items():
                if isinstance(v, dict) and "historical" in v:
                    hist = v["historical"]
                    break

    rows = []
    if isinstance(hist, list):
        for d in hist:
            ts = d.get("x") or d.get("t")
            val = d.get("y") or d.get("v") or d.get("value")
            if ts is None or val is None:
                continue
            # common: ms epoch vs s epoch
            if ts > 10**12:
                dt = datetime.utcfromtimestamp(ts / 1000).date()
            else:
                dt = datetime.utcfromtimestamp(ts).date()
            rows.append({"date": pd.to_datetime(dt), "FGI": float(val)})
    df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date")
    if df.empty:
        # create a small fallback series
        today = datetime.today()
        dates = pd.date_range(today - timedelta(days=30), today, freq="D")
        df = pd.DataFrame({"date": dates, "FGI": 50.0})
    return df.set_index("date")


def get_fgi_series() -> pd.Series:
    """
    Returns a clean single-level DatetimeIndex Series containing FGI values.
    Always safe to join with daily stock features.
    """
    try:
        data = fetch_fgi_json()
        df = fgi_to_dataframe(data)
    except Exception as e:
        app.logger.info("⚠️ FGI fetch failed, creating fallback series: %s", e)
        today = datetime.today()
        dates = pd.date_range(today - timedelta(days=180), today, freq="D")
        df = pd.DataFrame({"date": dates, "FGI": np.linspace(45, 55, len(dates))}).set_index("date")

    # Normalize index to single-level datetime index
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.asfreq("D").ffill()
    if "FGI_dev50" not in df.columns:
        df["FGI_dev50"] = (df["FGI"] - 50.0) / 100.0
    return df["FGI"]

# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # ensure expected columns exist
    if "Adj Close" not in out.columns and "Close" in out.columns:
        out["Adj Close"] = out["Close"]
    out["ret_1d"] = out["Adj Close"].pct_change()
    for w in (5, 10, 20, 50, 100, 200):
        out[f"sma_{w}"] = out["Adj Close"].rolling(w).mean()
        out[f"ema_{w}"] = out["Adj Close"].ewm(span=w, adjust=False).mean()
    out["rsi_14"] = rsi(out["Adj Close"], 14)
    out["vol_10"] = out["ret_1d"].rolling(10).std()
    out["hl_spread"] = (out.get("High", out["Adj Close"]) - out.get("Low", out["Adj Close"])) / out.get("Close", out["Adj Close"])
    out["slope_10"] = out["Adj Close"].pct_change(10) / 10.0
    return out


def make_supervised(df: pd.DataFrame, horizon: int = 1, target_col: str = "Adj Close"):
    data = df.copy()
    data["y"] = data[target_col].shift(-horizon)
    data = data.dropna()
    X = data.drop(columns=["y"])
    y = data["y"]
    return X, y

# -------------------------------------------------------------------
# Currency conversion
# -------------------------------------------------------------------

def get_to_inr_factor(ticker_info: dict) -> float:
    try:
        ccy = (ticker_info.get("currency") or "USD").upper()
    except Exception:
        ccy = "USD"
    if ccy == "INR":
        return 1.0
    pair = f"{ccy}INR=X"
    fx = yf.download(pair, period="10d", interval="1d", progress=False)
    if not fx.empty:
        return float(fx["Close"].iloc[-1])
    usd_inr = yf.download("USDINR=X", period="10d", interval="1d", progress=False)
    if ccy == "USD" and not usd_inr.empty:
        return float(usd_inr["Close"].iloc[-1])
    ccy_usd = yf.download(f"{ccy}=X", period="10d", interval="1d", progress=False)
    if not usd_inr.empty and not ccy_usd.empty:
        return float(usd_inr["Close"].iloc[-1] / ccy_usd["Close"].iloc[-1])
    return float("nan")

# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------

def arima_forecast(adj_close: pd.Series, order=(1,1,1)) -> float:
    s = adj_close.dropna()
    if len(s) < 50:
        return float("nan")
    try:
        model = ARIMA(s, order=order)
        res = model.fit(method_kwargs={"warn_convergence": False})
        fc = res.forecast(steps=1)
        return float(fc.iloc[-1])
    except Exception:
        return float("nan")


def rf_forecast(X_train: pd.DataFrame, y_train: pd.Series, X_next: pd.DataFrame):
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_next = float(rf.predict(X_next)[0])
    return pred_next, rf

# -------------------------------------------------------------------
# Core pipeline
# -------------------------------------------------------------------


def run_pipeline(tickers: list[str], start: str | None, end: str | None, horizon: int = 1):
    """
    Robust pipeline with all safety checks for missing data, NoneType ops, and FGI mismatches.
    """
    fgi_series = get_fgi_series()
    fgi_df = fgi_series.to_frame(name="FGI").copy()
    fgi_df.index.name = None
    fgi_df["FGI_dev50"] = (fgi_df["FGI"] - 50.0) / 100.0

    start_dt = pd.to_datetime(start) if start else None
    end_dt = pd.to_datetime(end) if end else None
    results = []

    for t in tickers:
        # ---- Download stock ----
        stock = yf.download(t, period="max", interval="1d", auto_adjust=False, progress=False)
        if stock.empty or "Adj Close" not in stock.columns:
            results.append({"Ticker": t, "error": "No data"})
            continue

        df = stock.copy()
        df.index = pd.to_datetime(df.index.date)

        if start_dt:
            df = df[df.index >= start_dt]
        if end_dt:
            df = df[df.index <= end_dt]
        if df.empty:
            results.append({"Ticker": t, "error": "No data in range"})
            continue

        # ---- Feature Engineering ----
        feat = add_features(df)

        # Make sure indices match properly (avoid multiindex)
        feat.index = pd.to_datetime(feat.index)
        fgi_df.index = pd.to_datetime(fgi_df.index)

        if isinstance(fgi_df.index, pd.MultiIndex):
            fgi_df = fgi_df.reset_index(level=list(range(fgi_df.index.nlevels - 1)), drop=True)

        tmp = feat.join(fgi_df, how="left").ffill()

        # --- SAFE adj_tech_price calculation ---
        base_series = None
        if "ema_20" in tmp.columns:
            base_series = tmp["ema_20"]
        elif "Adj Close" in tmp.columns:
            base_series = tmp["Adj Close"]
        elif "Close" in tmp.columns:
            base_series = tmp["Close"]
        else:
            base_series = pd.Series(0.0, index=tmp.index)

        base_series = pd.to_numeric(base_series, errors="coerce").fillna(method="ffill").fillna(method="bfill").fillna(0.0)
        if "FGI_dev50" not in tmp.columns:
            tmp["FGI_dev50"] = 0.0
        tmp["FGI_dev50"] = pd.to_numeric(tmp["FGI_dev50"], errors="coerce").fillna(0.0)
        tmp["adj_tech_price"] = base_series * (1.0 + 0.15 * tmp["FGI_dev50"])
        # ---------------------------------------

        feature_cols = [
            "Adj Close", "ret_1d",
            "sma_5","sma_10","sma_20","sma_50","sma_100","sma_200",
            "ema_5","ema_10","ema_20","ema_50","ema_100","ema_200",
            "rsi_14","vol_10","hl_spread","slope_10",
            "FGI","FGI_dev50","adj_tech_price",
        ]
        tmp_ml = tmp[feature_cols].dropna()
        short_note = None
        if len(tmp_ml) < 250:
            short_note = "Limited history for robust training"

        X, y = make_supervised(tmp_ml, horizon=horizon, target_col="Adj Close")
        if len(X) < 50:
            results.append({"Ticker": t, "error": "Insufficient samples after features"})
            # ........................................................# 

    # ..........................................................# 
            continue

        # ---- Split + train ----
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test,  y_test  = X.iloc[split:], y.iloc[split:]
        X_next = tmp_ml.iloc[[-1]]

        rf_next, rf_model = (float("nan"), None)
        try:
            rf_next, rf_model = rf_forecast(X_train, y_train, X_next)
        except Exception as e:
            print("⚠️ RF model failed:", e)

        arima_next = arima_forecast(tmp["Adj Close"])
        preds = [p for p in [rf_next, arima_next] if isinstance(p, float) and not math.isnan(p)]

        if not preds:
            results.append({"Ticker": t, "error": "Model prediction failed"})
            continue

        ens_next = 0.7 * rf_next + 0.3 * (arima_next if not math.isnan(arima_next) else rf_next)

        metrics = {}
        if rf_model is not None and len(X_test) > 0:
            yhat = rf_model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, yhat) * 100.0
            rmse = math.sqrt(mean_squared_error(y_test, yhat))
            metrics = {"MAPE_%": mape, "RMSE": rmse, "Accuracy_%": 100.0 - mape}

        # ---- Currency conversion safe ----
        try:
            info = yf.Ticker(t).info or {}
            to_inr = get_to_inr_factor(info)
        except Exception:
            info, to_inr = {}, float("nan")

        if to_inr is None or math.isnan(to_inr):
            to_inr = 83.0 if not t.endswith(".NS") else 1.0

        last_close = float(df["Adj Close"].iloc[-1])
        last_close_inr = last_close * to_inr
        pred_local = ens_next
        pred_inr = ens_next * to_inr

        results.append({
            "Ticker": t,
            "Currency": (info.get("currency") or "USD"),
            "From": str(df.index.min().date()),
            "To": str(df.index.max().date()),
            "Horizon_Days": horizon,
            "Last_Close_Local": last_close,
            "Last_Close_INR": last_close_inr,
            "Predicted_Next_Close_Local": pred_local,
            "Predicted_Next_Close_INR": pred_inr,
            "Note": short_note or "OK",
            **metrics,
        })

    return pd.DataFrame(results)



# -------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------

@app.route("/")
def index():
    # If you don't have a template, this will try to render index.html from templates folder.
    # You can also return a simple message for quick tests.
    try:
        return render_template("index.html")
    except Exception:
        return "<h3>FGI Stock Predictor running. Use /api/predict (POST JSON) or add a template 'index.html'.</h3>"

@app.route("/favicon.ico")
def favicon():
    return '', 204

@app.route("/api/fgi")
def api_fgi():
    try:
        start = request.args.get("start")
        end = request.args.get("end")
        s = get_fgi_series()
        df = s.to_frame("FGI")
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        df = df.reset_index().rename(columns={"index": "date"})
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        app.logger.info(f"⚠️ /api/fgi failed: {e}")
        return jsonify({"error": "Fear & Greed data unavailable. Using fallback."}), 500

@app.route("/api/predict", methods=["POST","GET"])
def api_predict():
    try:
        if request.method == "GET":
            return jsonify({"info": "POST JSON: {\"tickers\": [\"AAPL\",\"TCS.NS\"], \"horizon\": 1}"})
        payload = request.get_json(force=True)
        tickers = payload.get("tickers") or []
        start = payload.get("start")
        end = payload.get("end")
        horizon = int(payload.get("horizon") or 1)
        if not tickers:
            return jsonify({"error": "tickers required (list)"}), 400
        df = run_pipeline(tickers, start, end, horizon)
        # ensure numeric columns are serializable; round where possible
        for c in ["Last_Close_Local","Last_Close_INR","Predicted_Next_Close_Local","Predicted_Next_Close_INR","MAPE_%","RMSE","Accuracy_%"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').round(4)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        app.logger.info(f"⚠️ /api/predict failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
