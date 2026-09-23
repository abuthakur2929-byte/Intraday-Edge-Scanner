import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

st.set_page_config(
    page_title="Intraday Edge Scanner",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Intraday Edge Scanner — V1")
st.caption("Live NSE research scanner • No automatic order placement")

# =========================================================
# SETTINGS
# =========================================================

UPSTOX_TOKEN = st.secrets.get("UPSTOX_ANALYTICS_TOKEN", "")

# Major liquid NSE stocks for first live test
STOCKS = {
    "RELIANCE": "NSE_EQ|INE002A01018",
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "ICICIBANK": "NSE_EQ|INE090A01021",
    "SBIN": "NSE_EQ|INE062A01020",
    "INFY": "NSE_EQ|INE009A01021",
    "TCS": "NSE_EQ|INE467B01029",
    "ITC": "NSE_EQ|INE154A01025",
    "LT": "NSE_EQ|INE018A01030",
    "AXISBANK": "NSE_EQ|INE238A01034",
    "BHARTIARTL": "NSE_EQ|INE397D01024",
    "KOTAKBANK": "NSE_EQ|INE237A01028",
    "MARUTI": "NSE585A01020",
    "TATAMOTORS": "NSE_EQ|INE155A01022",
    "SUNPHARMA": "NSE_EQ|INE044A01036",
    "ADANIENT": "NSE_EQ|INE423A01024",
}

# =========================================================
# INDICATORS
# =========================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] - df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"] - df["Close"].shift()
    ).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def add_indicators(df):
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = df["Close"].ewm(
        span=200,
        adjust=False
    ).mean()

    df["RSI14"] = calculate_rsi(
        df["Close"],
        14
    )

    df["ATR14"] = calculate_atr(
        df,
        14
    )

    df["AvgVol20"] = df["Volume"].rolling(
        20
    ).mean()

    df["RVOL"] = (
        df["Volume"] /
        df["AvgVol20"].replace(0, np.nan)
    )

    # Intraday VWAP
    typical_price = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    date_key = df["Datetime"].dt.date

    cumulative_pv = (
        typical_price * df["Volume"]
    ).groupby(date_key).cumsum()

    cumulative_volume = (
        df["Volume"]
        .groupby(date_key)
        .cumsum()
    )

    df["VWAP"] = (
        cumulative_pv /
        cumulative_volume.replace(0, np.nan)
    )

    # Previous 20-bar high/low
    df["Prev20High"] = (
        df["High"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df["Prev20Low"] = (
        df["Low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    df["Breakout"] = (
        df["Close"] > df["Prev20High"]
    )

    df["Breakdown"] = (
        df["Close"] < df["Prev20Low"]
    )

    return df


# =========================================================
# UPSTOX DATA
# =========================================================

def get_intraday_data(instrument_key, interval=5):
    if not UPSTOX_TOKEN:
        raise ValueError(
            "UPSTOX_ANALYTICS_TOKEN is missing in Streamlit Secrets."
        )

    url = (
        "https://api.upstox.com/v3/"
        f"historical-candle/intraday/"
        f"{instrument_key}/minutes/{interval}"
    )

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_TOKEN}"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Upstox API error {response.status_code}: "
            f"{response.text}"
        )

    payload = response.json()

    candles = (
        payload
        .get("data", {})
        .get("candles", [])
    )

    if not candles:
        return pd.DataFrame()

    rows = []

    for candle in candles:
        rows.append({
            "Datetime": pd.to_datetime(candle[0]),
            "Open": float(candle[1]),
            "High": float(candle[2]),
            "Low": float(candle[3]),
            "Close": float(candle[4]),
            "Volume": float(candle[5])
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "Datetime"
    ).reset_index(drop=True)

    return df


# =========================================================
# SCORING
# =========================================================

def scan_stock(symbol, instrument_key):
    df = get_intraday_data(
        instrument_key,
        interval=5
    )

    if df.empty:
        return None

    if len(df) < 220:
        return None

    df = add_indicators(df)

    latest = df.iloc[-1]

    required = [
        "EMA20",
        "EMA50",
        "EMA200",
        "VWAP",
        "RSI14",
        "ATR14",
        "RVOL"
    ]

    if any(
        pd.isna(latest[x])
        for x in required
    ):
        return None

    long_score = 0
    short_score = 0

    # Trend
    if (
        latest["EMA20"] >
        latest["EMA50"] >
        latest["EMA200"]
    ):
        long_score += 20

    if (
        latest["EMA20"] <
        latest["EMA50"] <
        latest["EMA200"]
    ):
        short_score += 20

    # VWAP
    if latest["Close"] > latest["VWAP"]:
        long_score += 15

    if latest["Close"] < latest["VWAP"]:
        short_score += 15

    # Breakout / Breakdown
    if latest["Breakout"]:
        long_score += 25

    if latest["Breakdown"]:
        short_score += 25

    # Relative Volume
    if latest["RVOL"] >= 1.5:
        long_score += 15
        short_score += 15

    # RSI
    if 55 <= latest["RSI14"] <= 75:
        long_score += 10

    if 25 <= latest["RSI14"] <= 45:
        short_score += 10

    # Small confirmation baseline
    long_score += 5
    short_score += 5

    if long_score >= short_score:
        direction = "LONG"
        score = long_score
    else:
        direction = "SHORT"
        score = short_score

    if score < 55:
        return None

    entry = float(latest["Close"])
    atr = float(latest["ATR14"])

    if direction == "LONG":
        sl = entry - (1.2 * atr)
        target = entry + (2.0 * atr)
    else:
        sl = entry + (1.2 * atr)
        target = entry - (2.0 * atr)

    return {
        "Symbol": symbol,
        "Direction": direction,
        "Score": round(score, 1),
        "Entry": round(entry, 2),
        "SL": round(sl, 2),
        "Target": round(target, 2),
        "ATR14": round(atr, 2),
        "RSI14": round(float(latest["RSI14"]), 1),
        "RVOL": round(float(latest["RVOL"]), 2),
        "VWAP": round(float(latest["VWAP"]), 2),
        "Last Candle": latest["Datetime"]
    }


# =========================================================
# UI
# =========================================================

st.subheader("⚡ Live NSE Scanner")

interval = st.selectbox(
    "Candle interval",
    [1, 3, 5, 10, 15],
    index=2
)

selected_stocks = st.multiselect(
    "Stocks to scan",
    list(STOCKS.keys()),
    default=list(STOCKS.keys())
)

scan_button = st.button(
    "🔎 Scan Market",
    type="primary"
)

if not UPSTOX_TOKEN:
    st.error(
        "Upstox Analytics Token नहीं मिला। "
        "Streamlit Secrets में UPSTOX_ANALYTICS_TOKEN check करें."
    )
    st.stop()

if scan_button:

    if not selected_stocks:
        st.warning(
            "कम से कम एक stock select करो."
        )
        st.stop()

    results = []
    progress = st.progress(0)

    for i, symbol in enumerate(selected_stocks):

        try:
            data = get_intraday_data(
                STOCKS[symbol],
                interval=interval
            )

            if data.empty:
                continue

            if len(data) < 220:
                continue

            data = add_indicators(data)

            latest = data.iloc[-1]

            long_score = 5
            short_score = 5

            if (
                latest["EMA20"] >
                latest["EMA50"] >
                latest["EMA200"]
            ):
                long_score += 20

            if (
                latest["EMA20"] <
                latest["EMA50"] <
                latest["EMA200"]
            ):
                short_score += 20

            if latest["Close"] > latest["VWAP"]:
                long_score += 15

            if latest["Close"] < latest["VWAP"]:
                short_score += 15

            if latest["Breakout"]:
                long_score += 25

            if latest["Breakdown"]:
                short_score += 25

            if latest["RVOL"] >= 1.5:
                long_score += 15
                short_score += 15

            if 55 <= latest["RSI14"] <= 75:
                long_score += 10

            if 25 <= latest["RSI14"] <= 45:
                short_score += 10

            if max(
                long_score,
                short_score
            ) < 55:
                continue

            if long_score >= short_score:
                direction = "LONG"
                score = long_score
                entry = float(latest["Close"])
                sl = entry - 1.2 * float(latest["ATR14"])
                target = entry + 2.0 * float(latest["ATR14"])
            else:
                direction = "SHORT"
                score = short_score
                entry = float(latest["Close"])
                sl = entry + 1.2 * float(latest["ATR14"])
                target = entry - 2.0 * float(latest["ATR14"])

            results.append({
                "Symbol": symbol,
                "Direction": direction,
                "Score": round(score, 1),
                "Entry": round(entry, 2),
                "SL": round(sl, 2),
                "Target": round(target, 2),
                "ATR14": round(float(latest["ATR14"]), 2),
                "RSI14": round(float(latest["RSI14"]), 1),
                "RVOL": round(float(latest["RVOL"]), 2),
                "VWAP": round(float(latest["VWAP"]), 2),
                "Last Candle": latest["Datetime"]
            })

        except Exception as e:
            st.warning(
                f"{symbol}: {str(e)}"
            )

        progress.progress(
            (i + 1) / len(selected_stocks)
        )

    if results:

        result_df = pd.DataFrame(
            results
        ).sort_values(
            "Score",
            ascending=False
        )

        st.success(
            f"{len(result_df)} setup(s) मिले."
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Scanner Results",
            csv,
            "intraday_edge_live_scan.csv",
            "text/csv"
        )

    else:
        st.info(
            "अभी कोई 55+ setup नहीं मिला। "
            "यह जरूरी नहीं कि market में कोई trade नहीं है; "
            "हमारा current filter strict है."
        )

else:
    st.info(
        "Stocks select करके **Scan Market** दबाओ."
    )

st.divider()

st.caption(
    "Research / scanning only • No automatic order placement"
)

st.caption(
    "Signals are rule-based research outputs, "
    "not a guarantee of profitable trades."
)
