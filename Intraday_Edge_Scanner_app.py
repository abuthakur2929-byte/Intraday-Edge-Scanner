import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Intraday Edge Scanner V1", page_icon="🎯", layout="wide")
st.title("🎯 Intraday Edge Scanner — V1")
st.caption("Research / scanning only • No automatic order placement")

uploaded = st.file_uploader("Upload intraday OHLCV CSV", type=["csv"])

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def scan_symbol(g):
    g = g.sort_values("Datetime").copy()
    if len(g) < 220:
        return None

    g["EMA20"] = ema(g["Close"], 20)
    g["EMA50"] = ema(g["Close"], 50)
    g["EMA200"] = ema(g["Close"], 200)
    g["RSI"] = rsi(g["Close"], 14)
    g["ATR"] = atr(g, 14)
    g["AvgVol20"] = g["Volume"].rolling(20).mean()
    g["RVOL"] = g["Volume"] / g["AvgVol20"].replace(0, np.nan)

    typical = (g["High"] + g["Low"] + g["Close"]) / 3
    day = g["Datetime"].dt.date
    g["TPV"] = typical * g["Volume"]
    g["CumTPV"] = g.groupby(day)["TPV"].cumsum()
    g["CumVol"] = g.groupby(day)["Volume"].cumsum()
    g["VWAP"] = g["CumTPV"] / g["CumVol"].replace(0, np.nan)

    last = g.iloc[-1]
    prev20_high = g["High"].shift(1).rolling(20).max().iloc[-1]
    prev20_low = g["Low"].shift(1).rolling(20).min().iloc[-1]

    long_score = 0
    short_score = 0
    long_reasons, short_reasons = [], []

    if last["EMA20"] > last["EMA50"] > last["EMA200"]:
        long_score += 20
        long_reasons.append("EMA trend")
    if last["EMA20"] < last["EMA50"] < last["EMA200"]:
        short_score += 20
        short_reasons.append("EMA trend")

    if last["Close"] > last["VWAP"]:
        long_score += 15
        long_reasons.append("Above VWAP")
    if last["Close"] < last["VWAP"]:
        short_score += 15
        short_reasons.append("Below VWAP")

    if last["Close"] > prev20_high:
        long_score += 25
        long_reasons.append("20-bar breakout")
    if last["Close"] < prev20_low:
        short_score += 25
        short_reasons.append("20-bar breakdown")

    if last["RVOL"] >= 1.5:
        long_score += 15
        short_score += 15
        long_reasons.append(f"RVOL {last['RVOL']:.1f}x")
        short_reasons.append(f"RVOL {last['RVOL']:.1f}x")

    if 55 <= last["RSI"] <= 75:
        long_score += 10
        long_reasons.append("RSI momentum")
    if 25 <= last["RSI"] <= 45:
        short_score += 10
        short_reasons.append("RSI momentum")

    long_score += 5
    short_score += 5

    side = "LONG" if long_score >= short_score else "SHORT"
    score = max(long_score, short_score)

    if score < 55:
        return None

    entry = float(last["Close"])
    if side == "LONG":
        sl = entry - 1.2 * float(last["ATR"])
        target = entry + 2.0 * float(last["ATR"])
        reasons = long_reasons
    else:
        sl = entry + 1.2 * float(last["ATR"])
        target = entry - 2.0 * float(last["ATR"])
        reasons = short_reasons

    return {
        "Symbol": str(last["Symbol"]),
        "Signal": side,
        "Score": int(score),
        "Close": round(entry, 2),
        "ATR": round(float(last["ATR"]), 2),
        "Entry": round(entry, 2),
        "Stop Loss": round(sl, 2),
        "Target": round(target, 2),
        "RVOL": round(float(last["RVOL"]), 2),
        "RSI": round(float(last["RSI"]), 1),
        "VWAP": round(float(last["VWAP"]), 2),
        "Reasons": " • ".join(reasons),
    }

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        df.columns = [c.strip() for c in df.columns]
        required = {"Datetime","Symbol","Open","High","Low","Close","Volume"}
        missing = required - set(df.columns)
        if missing:
            st.error("Missing columns: " + ", ".join(sorted(missing)))
            st.stop()

        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        for c in ["Open","High","Low","Close","Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Datetime","Symbol","Open","High","Low","Close","Volume"])

        results = []
        for symbol, group in df.groupby("Symbol"):
            r = scan_symbol(group)
            if r:
                results.append(r)

        if not results:
            st.warning("No setups passed the V1 filters.")
        else:
            out = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.subheader("🔥 Top Intraday Setups")
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button(
                "Download scan results",
                out.to_csv(index=False).encode("utf-8"),
                "intraday_scan_results.csv",
                "text/csv"
            )
    except Exception as e:
        st.error(f"Could not process file: {e}")
else:
    st.markdown("""
### V1 scanner
- EMA 20 / 50 / 200
- Intraday VWAP
- 20-bar breakout / breakdown
- Relative Volume
- RSI 14
- ATR 14
- Setup score
- LONG / SHORT ranking
- Indicative ATR-based SL and target

This is a research scanner, not a guarantee of profitable trades.
""")
