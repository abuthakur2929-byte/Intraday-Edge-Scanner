import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Intraday Edge Scanner", page_icon="🎯", layout="wide")

st.title("🎯 Intraday Edge Scanner — V3")
st.caption("Live NSE research scanner + rule-based backtest • No automatic order placement")

UPSTOX_TOKEN = st.secrets.get("UPSTOX_ANALYTICS_TOKEN", "")
IST = ZoneInfo("Asia/Kolkata")

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
    "MARUTI": "NSE_EQ|INE585B01010",
    "TATAMOTORS": "NSE_EQ|INE155A01022",
    "SUNPHARMA": "NSE_EQ|INE044A01036",
    "ADANIENT": "NSE_EQ|INE423A01024",
}

def now_ist():
    return datetime.now(IST)

def candles_to_df(candles):
    if not candles:
        return pd.DataFrame()
    rows = []
    for candle in candles:
        if len(candle) < 6:
            continue
        rows.append({
            "Datetime": pd.to_datetime(candle[0]),
            "Open": float(candle[1]),
            "High": float(candle[2]),
            "Low": float(candle[3]),
            "Close": float(candle[4]),
            "Volume": float(candle[5]),
        })
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["Datetime"])
        .sort_values("Datetime")
        .reset_index(drop=True)
    )

def upstox_get(url):
    if not UPSTOX_TOKEN:
        raise ValueError("UPSTOX_ANALYTICS_TOKEN नहीं मिला।")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_TOKEN}",
    }
    response = requests.get(url, headers=headers, timeout=25)
    if response.status_code != 200:
        raise RuntimeError(
            f"Upstox API error {response.status_code}: {response.text[:300]}"
        )
    payload = response.json()
    return payload.get("data", {}).get("candles", [])

def get_historical_data(instrument_key, interval=5, days_back=30):
    today = now_ist().date()
    to_date = today - timedelta(days=1)
    from_date = today - timedelta(days=days_back)
    url = (
        "https://api.upstox.com/v3/historical-candle/"
        f"{instrument_key}/minutes/{interval}/"
        f"{to_date.isoformat()}/{from_date.isoformat()}"
    )
    return candles_to_df(upstox_get(url))

def get_intraday_data(instrument_key, interval=5):
    url = (
        "https://api.upstox.com/v3/historical-candle/intraday/"
        f"{instrument_key}/minutes/{interval}"
    )
    return candles_to_df(upstox_get(url))

def get_combined_data(instrument_key, interval=5):
    frames = []
    historical = get_historical_data(instrument_key, interval, 30)
    intraday = get_intraday_data(instrument_key, interval)
    if not historical.empty:
        frames.append(historical)
    if not intraday.empty:
        frames.append(intraday)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["Datetime"])
        .sort_values("Datetime")
        .reset_index(drop=True)
    )

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

def add_indicators(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["ATR14"] = calculate_atr(df, 14)

    day_key = df["Datetime"].dt.date
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical * df["Volume"]
    df["VWAP"] = (
        pv.groupby(day_key).cumsum()
        / df["Volume"].groupby(day_key).cumsum().replace(0, np.nan)
    )

    df["AvgVol20"] = df["Volume"].rolling(20).mean()
    df["RVOL"] = df["Volume"] / df["AvgVol20"].replace(0, np.nan)

    df["Prev20High"] = df.groupby(day_key)["High"].transform(
        lambda x: x.rolling(20).max().shift(1)
    )
    df["Prev20Low"] = df.groupby(day_key)["Low"].transform(
        lambda x: x.rolling(20).min().shift(1)
    )
    df["Breakout"] = df["Close"] > df["Prev20High"]
    df["Breakdown"] = df["Close"] < df["Prev20Low"]
    return df

def score_row(row):
    required = ["EMA20", "EMA50", "EMA200", "VWAP", "RSI14", "ATR14", "RVOL"]
    if any(pd.isna(row[c]) for c in required):
        return None

    long_score = 5
    short_score = 5

    if row["EMA20"] > row["EMA50"] > row["EMA200"]:
        long_score += 20
    if row["EMA20"] < row["EMA50"] < row["EMA200"]:
        short_score += 20

    if row["Close"] > row["VWAP"]:
        long_score += 15
    if row["Close"] < row["VWAP"]:
        short_score += 15

    if bool(row["Breakout"]):
        long_score += 25
    if bool(row["Breakdown"]):
        short_score += 25

    if row["RVOL"] >= 1.5:
        long_score += 15
        short_score += 15

    if 55 <= row["RSI14"] <= 75:
        long_score += 10
    if 25 <= row["RSI14"] <= 45:
        short_score += 10

    direction = "LONG" if long_score >= short_score else "SHORT"
    score = max(long_score, short_score)

    if score < 55:
        return None

    entry = float(row["Close"])
    atr = float(row["ATR14"])
    sl = entry - 1.2 * atr if direction == "LONG" else entry + 1.2 * atr
    target = entry + 2.0 * atr if direction == "LONG" else entry - 2.0 * atr

    return {
        "Direction": direction,
        "Score": score,
        "Entry": entry,
        "SL": sl,
        "Target": target,
        "ATR14": atr,
        "RSI14": float(row["RSI14"]),
        "RVOL": float(row["RVOL"]),
        "VWAP": float(row["VWAP"]),
        "EMA20": float(row["EMA20"]),
        "EMA50": float(row["EMA50"]),
        "EMA200": float(row["EMA200"]),
        "Datetime": row["Datetime"],
    }

def live_signal(df, symbol):
    if df.empty:
        return None
    work = add_indicators(df)
    result = score_row(work.iloc[-1])
    if result is None:
        return None
    result["Symbol"] = symbol
    return result

def backtest_symbol(df, symbol, min_score=55, max_hold_bars=24):
    if df.empty:
        return []
    work = add_indicators(df)
    trades = []

    # Signal is generated at candle close; execution begins from the NEXT candle.
    for i in range(220, len(work) - 1):
        row = work.iloc[i]
        signal = score_row(row)
        if signal is None:
            continue

        entry_idx = i + 1
        entry_row = work.iloc[entry_idx]
        entry = float(entry_row["Open"])

        # Keep the original ATR-based distance from the signal candle.
        atr = float(row["ATR14"])
        if signal["Direction"] == "LONG":
            sl = entry - 1.2 * atr
            target = entry + 2.0 * atr
        else:
            sl = entry + 1.2 * atr
            target = entry - 2.0 * atr

        exit_price = None
        exit_time = None
        outcome = "TIME_EXIT"

        last_idx = min(len(work) - 1, entry_idx + max_hold_bars)

        for j in range(entry_idx, last_idx + 1):
            bar = work.iloc[j]

            if signal["Direction"] == "LONG":
                hit_sl = bar["Low"] <= sl
                hit_target = bar["High"] >= target
                if hit_sl and hit_target:
                    # Conservative same-candle assumption.
                    exit_price = sl
                    outcome = "SL"
                elif hit_sl:
                    exit_price = sl
                    outcome = "SL"
                elif hit_target:
                    exit_price = target
                    outcome = "TARGET"
            else:
                hit_sl = bar["High"] >= sl
                hit_target = bar["Low"] <= target
                if hit_sl and hit_target:
                    exit_price = sl
                    outcome = "SL"
                elif hit_sl:
                    exit_price = sl
                    outcome = "SL"
                elif hit_target:
                    exit_price = target
                    outcome = "TARGET"

            if exit_price is not None:
                exit_time = bar["Datetime"]
                break

        if exit_price is None:
            exit_price = float(work.iloc[last_idx]["Close"])
            exit_time = work.iloc[last_idx]["Datetime"]

        if signal["Direction"] == "LONG":
            pnl_r = (exit_price - entry) / (1.2 * atr)
        else:
            pnl_r = (entry - exit_price) / (1.2 * atr)

        trades.append({
            "Symbol": symbol,
            "Direction": signal["Direction"],
            "Score": round(signal["Score"], 1),
            "Signal Time": signal["Datetime"],
            "Entry Time": entry_row["Datetime"],
            "Entry": round(entry, 2),
            "SL": round(sl, 2),
            "Target": round(target, 2),
            "Exit": round(exit_price, 2),
            "Exit Time": exit_time,
            "Outcome": outcome,
            "R": round(pnl_r, 3),
        })

    return trades

def backtest_summary(trades):
    if not trades:
        return {
            "Trades": 0, "Wins": 0, "Losses": 0, "Win Rate %": 0.0,
            "Net R": 0.0, "Profit Factor": 0.0, "Max Drawdown R": 0.0
        }

    t = pd.DataFrame(trades)
    wins = int((t["R"] > 0).sum())
    losses = int((t["R"] < 0).sum())
    gross_profit = t.loc[t["R"] > 0, "R"].sum()
    gross_loss = abs(t.loc[t["R"] < 0, "R"].sum())
    equity = t["R"].cumsum()
    drawdown = equity - equity.cummax()

    return {
        "Trades": len(t),
        "Wins": wins,
        "Losses": losses,
        "Win Rate %": round(100 * wins / len(t), 2),
        "Net R": round(t["R"].sum(), 3),
        "Profit Factor": round(gross_profit / gross_loss, 3) if gross_loss else float("inf"),
        "Max Drawdown R": round(drawdown.min(), 3),
    }

if not UPSTOX_TOKEN:
    st.error("Upstox Analytics Token नहीं मिला। Streamlit Secrets में UPSTOX_ANALYTICS_TOKEN check करें.")
    st.stop()

st.info(f"Scanner clock: {now_ist().strftime('%d-%m-%Y %H:%M:%S IST')}")

tab_live, tab_backtest = st.tabs(["⚡ Live Scan", "📊 Backtest"])

with tab_live:
    st.subheader("Live NSE Scanner")
    interval = st.selectbox("Candle interval", [1, 3, 5, 10, 15], index=2, key="live_interval")
    selected = st.multiselect(
        "Stocks to scan",
        list(STOCKS.keys()),
        default=list(STOCKS.keys()),
        key="live_stocks",
    )

    if st.button("🔎 Scan Market", type="primary", key="scan"):
        if not selected:
            st.warning("कम से कम एक stock select करो.")
        else:
            results = []
            errors = []
            progress = st.progress(0)
            status = st.empty()

            for i, symbol in enumerate(selected):
                status.write(f"Scanning {symbol}...")
                try:
                    data = get_combined_data(STOCKS[symbol], interval)
                    if data.empty:
                        continue
                    result = live_signal(data, symbol)
                    if result:
                        results.append(result)
                except Exception as e:
                    errors.append(f"{symbol}: {e}")
                progress.progress((i + 1) / len(selected))

            status.empty()

            if results:
                result_df = pd.DataFrame(results).sort_values(
                    "Score", ascending=False
                ).reset_index(drop=True)
                st.success(f"{len(result_df)} setup(s) मिले.")
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download Live Results",
                    result_df.to_csv(index=False).encode("utf-8"),
                    "intraday_edge_live_scan.csv",
                    "text/csv",
                )
            else:
                st.info("अभी कोई 55+ setup नहीं मिला.")

            if errors:
                with st.expander("API / data errors"):
                    for err in errors:
                        st.write(err)

with tab_backtest:
    st.subheader("Rule-Based Backtest")
    st.caption(
        "Signal candle के close पर setup बनता है और entry अगले candle के open से शुरू होती है. "
        "Same-candle SL + target hit होने पर conservative assumption में SL माना जाता है."
    )

    bt_interval = st.selectbox(
        "Backtest candle interval",
        [5, 10, 15],
        index=0,
        key="bt_interval",
    )
    bt_stocks = st.multiselect(
        "Backtest stocks",
        list(STOCKS.keys()),
        default=list(STOCKS.keys()),
        key="bt_stocks",
    )
    min_score = st.slider("Minimum setup score", 55, 90, 55, 5)
    max_hold = st.slider("Maximum holding candles", 6, 48, 24, 3)

    if st.button("📊 Run Backtest", type="primary", key="backtest"):
        if not bt_stocks:
            st.warning("कम से कम एक stock select करो.")
        else:
            all_trades = []
            errors = []
            progress = st.progress(0)
            status = st.empty()

            for i, symbol in enumerate(bt_stocks):
                status.write(f"Backtesting {symbol}...")
                try:
                    data = get_historical_data(
                        STOCKS[symbol],
                        interval=bt_interval,
                        days_back=30,
                    )
                    if not data.empty:
                        trades = backtest_symbol(
                            data, symbol, min_score=min_score, max_hold_bars=max_hold
                        )
                        # Enforce selected score threshold here as well.
                        trades = [x for x in trades if x["Score"] >= min_score]
                        all_trades.extend(trades)
                except Exception as e:
                    errors.append(f"{symbol}: {e}")
                progress.progress((i + 1) / len(bt_stocks))

            status.empty()

            summary = backtest_summary(all_trades)
            st.subheader("Backtest Summary")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Trades", summary["Trades"])
            c2.metric("Win Rate", f'{summary["Win Rate %"]}%')
            c3.metric("Net R", summary["Net R"])
            c4.metric("Profit Factor", summary["Profit Factor"])
            c5.metric("Max DD (R)", summary["Max Drawdown R"])
            c6.metric("Wins / Losses", f'{summary["Wins"]} / {summary["Losses"]}')

            if all_trades:
                trades_df = pd.DataFrame(all_trades).sort_values(
                    "Signal Time", ascending=False
                )
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download Backtest Trades",
                    trades_df.to_csv(index=False).encode("utf-8"),
                    "intraday_edge_backtest.csv",
                    "text/csv",
                )
            else:
                st.warning(
                    "Is period / score filter में कोई qualifying trade नहीं मिला."
                )

            if errors:
                with st.expander("Backtest API / data errors"):
                    for err in errors:
                        st.write(err)

st.divider()
st.caption("Research / scanning only • No automatic order placement")
st.caption("Backtest results are historical simulations, not a guarantee of future performance.")
