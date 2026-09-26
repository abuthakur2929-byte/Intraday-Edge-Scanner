import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Intraday Edge Scanner", page_icon="🎯", layout="wide")

st.title("🎯 Intraday Edge Scanner — V9")
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


def score_direction(row, direction):
    required = ["EMA20", "EMA50", "EMA200", "VWAP", "RSI14", "ATR14", "RVOL"]
    if any(pd.isna(row[c]) for c in required):
        return None

    score = 5

    if direction == "LONG":
        if row["EMA20"] > row["EMA50"] > row["EMA200"]:
            score += 20
        if row["Close"] > row["VWAP"]:
            score += 15
        if bool(row["Breakout"]):
            score += 25
        if row["RVOL"] >= 1.5:
            score += 15
        if 55 <= row["RSI14"] <= 75:
            score += 10
    else:
        if row["EMA20"] < row["EMA50"] < row["EMA200"]:
            score += 20
        if row["Close"] < row["VWAP"]:
            score += 15
        if bool(row["Breakdown"]):
            score += 25
        if row["RVOL"] >= 1.5:
            score += 15
        if 25 <= row["RSI14"] <= 45:
            score += 10

    return score


def v6_backtest_symbol(df, symbol, direction, min_score=55, max_hold_bars=24,
                       use_regime=True):
    """V6: independent LONG/SHORT testing, one active trade per stock,
    and a simple higher-timeframe trend regime using EMA50/EMA200."""
    if df.empty:
        return []

    work = add_indicators(df)
    trades = []
    i = 220

    while i < len(work) - 1:
        row = work.iloc[i]
        score = score_direction(row, direction)

        if score is None or score < min_score:
            i += 1
            continue

        # V6 regime filter: only trade in the stock's prevailing trend.
        # This is deliberately based only on information available at signal close.
        if use_regime:
            if direction == "LONG" and not (row["EMA50"] > row["EMA200"]):
                i += 1
                continue
            if direction == "SHORT" and not (row["EMA50"] < row["EMA200"]):
                i += 1
                continue

        entry_idx = i + 1
        entry_row = work.iloc[entry_idx]
        # V9: intraday-only. Never carry a signal into the next trading day.
        if row["Datetime"].date() != entry_row["Datetime"].date():
            i += 1
            continue
        entry = float(entry_row["Open"])
        atr = float(row["ATR14"])

        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        if direction == "LONG":
            sl = entry - 1.2 * atr
            target = entry + 2.0 * atr
        else:
            sl = entry + 1.2 * atr
            target = entry - 2.0 * atr

        exit_price = None
        exit_time = None
        outcome = "TIME_EXIT"
        exit_idx = min(len(work) - 1, entry_idx + max_hold_bars)

        for j in range(entry_idx, exit_idx + 1):
            bar = work.iloc[j]
            # V9: force an intraday exit at the final bar of the signal day.
            if bar["Datetime"].date() != row["Datetime"].date():
                prev = work.iloc[j - 1]
                exit_price = float(prev["Close"])
                exit_time = prev["Datetime"]
                outcome = "SESSION_EXIT"
                break

            if direction == "LONG":
                hit_sl = bar["Low"] <= sl
                hit_target = bar["High"] >= target
            else:
                hit_sl = bar["High"] >= sl
                hit_target = bar["Low"] <= target

            # Conservative same-candle rule: SL wins if both are touched.
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
            exit_price = float(work.iloc[exit_idx]["Close"])
            exit_time = work.iloc[exit_idx]["Datetime"]

        if direction == "LONG":
            pnl_r = (exit_price - entry) / (1.2 * atr)
        else:
            pnl_r = (entry - exit_price) / (1.2 * atr)

        trades.append({
            "Symbol": symbol,
            "Direction": direction,
            "Score": round(score, 1),
            "Signal Time": row["Datetime"],
            "Entry Time": entry_row["Datetime"],
            "Entry": round(entry, 2),
            "SL": round(sl, 2),
            "Target": round(target, 2),
            "Exit": round(exit_price, 2),
            "Exit Time": exit_time,
            "Outcome": outcome,
            "R": round(pnl_r, 3),
        })

        # No overlapping positions.
        i = exit_idx + 1

    return trades


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

    # V6: one active position per stock at a time.
    # A new signal is not evaluated until the previous trade has exited.
    i = 220
    while i < len(work) - 1:
        row = work.iloc[i]
        signal = score_row(row)
        if signal is None or signal["Score"] < min_score:
            i += 1
            continue

        entry_idx = i + 1
        entry_row = work.iloc[entry_idx]
        # V9: intraday-only. Never carry a signal into the next trading day.
        if row["Datetime"].date() != entry_row["Datetime"].date():
            i += 1
            continue
        entry = float(entry_row["Open"])
        atr = float(row["ATR14"])
        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        if signal["Direction"] == "LONG":
            sl = entry - 1.2 * atr
            target = entry + 2.0 * atr
        else:
            sl = entry + 1.2 * atr
            target = entry - 2.0 * atr

        exit_price = None
        exit_time = None
        outcome = "TIME_EXIT"
        exit_idx = min(len(work) - 1, entry_idx + max_hold_bars)

        for j in range(entry_idx, exit_idx + 1):
            bar = work.iloc[j]
            if bar["Datetime"].date() != signal["Datetime"].date():
                prev = work.iloc[j - 1]
                exit_price = float(prev["Close"])
                exit_time = prev["Datetime"]
                outcome = "SESSION_EXIT"
                break
            if signal["Direction"] == "LONG":
                hit_sl = bar["Low"] <= sl
                hit_target = bar["High"] >= target
                if hit_sl and hit_target:
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
            exit_price = float(work.iloc[exit_idx]["Close"])
            exit_time = work.iloc[exit_idx]["Datetime"]

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

        # Critical V6 fix: skip every candle while this trade is active.
        i = exit_idx + 1

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

tab_live, tab_backtest, tab_diagnosis = st.tabs(["⚡ Live Scan", "📊 Backtest", "🧪 Strategy Diagnosis"])

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
    st.subheader("Rule-Based Backtest — V9")
    st.caption(
        "Signal candle के close पर setup बनता है और entry अगले candle के open से शुरू होती है. "
        "Same-candle SL + target hit होने पर conservative assumption में SL माना जाता है. V9 में एक stock पर एक समय में केवल एक active trade होगा और overnight entry को block किया जाता है."
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


def v7_backtest_symbol(df, symbol, direction, score_min, score_max,
                       max_hold_bars=24, use_regime=False):
    """Legacy validation: test one direction and one score band only.
    No overlapping trades. Optional regime filter is disabled by default
    so the score-band effect is isolated from V6.
    """
    if df.empty:
        return []

    work = add_indicators(df)
    trades = []
    i = 220

    while i < len(work) - 1:
        row = work.iloc[i]
        score = score_direction(row, direction)

        if score is None or score < score_min or score > score_max:
            i += 1
            continue

        if use_regime:
            if direction == "LONG" and not (row["EMA50"] > row["EMA200"]):
                i += 1
                continue
            if direction == "SHORT" and not (row["EMA50"] < row["EMA200"]):
                i += 1
                continue

        entry_idx = i + 1
        entry_row = work.iloc[entry_idx]
        # V9: intraday-only. Never carry a signal into the next trading day.
        if row["Datetime"].date() != entry_row["Datetime"].date():
            i += 1
            continue
        entry = float(entry_row["Open"])
        atr = float(row["ATR14"])

        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        sl = entry - 1.2 * atr if direction == "LONG" else entry + 1.2 * atr
        target = entry + 2.0 * atr if direction == "LONG" else entry - 2.0 * atr

        exit_idx = min(len(work) - 1, entry_idx + max_hold_bars)
        exit_price = None
        exit_time = None
        outcome = "TIME_EXIT"

        for j in range(entry_idx, exit_idx + 1):
            bar = work.iloc[j]
            # V9: force an intraday exit at the final bar of the signal day.
            if bar["Datetime"].date() != row["Datetime"].date():
                prev = work.iloc[j - 1]
                exit_price = float(prev["Close"])
                exit_time = prev["Datetime"]
                outcome = "SESSION_EXIT"
                break

            if direction == "LONG":
                hit_sl = bar["Low"] <= sl
                hit_target = bar["High"] >= target
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
            exit_price = float(work.iloc[exit_idx]["Close"])
            exit_time = work.iloc[exit_idx]["Datetime"]

        pnl_r = (
            (exit_price - entry) / (1.2 * atr)
            if direction == "LONG"
            else (entry - exit_price) / (1.2 * atr)
        )

        trades.append({
            "Symbol": symbol,
            "Direction": direction,
            "Score": round(score, 1),
            "Signal Time": row["Datetime"],
            "Entry Time": entry_row["Datetime"],
            "Entry": round(entry, 2),
            "SL": round(sl, 2),
            "Target": round(target, 2),
            "Exit": round(exit_price, 2),
            "Exit Time": exit_time,
            "Outcome": outcome,
            "R": round(pnl_r, 3),
        })

        i = exit_idx + 1

    return trades


def v7_stats(df):
    if df.empty:
        return {
            "Trades": 0, "Win Rate %": 0.0,
            "Net R": 0.0, "Profit Factor": 0.0, "Avg R": 0.0
        }
    wins = int((df["R"] > 0).sum())
    gp = df.loc[df["R"] > 0, "R"].sum()
    gl = abs(df.loc[df["R"] < 0, "R"].sum())
    return {
        "Trades": len(df),
        "Win Rate %": round(100 * wins / len(df), 2),
        "Net R": round(df["R"].sum(), 3),
        "Profit Factor": round(gp / gl, 3) if gl else float("inf"),
        "Avg R": round(df["R"].mean(), 4),
    }


def v7_walkforward_split(df, train_days=15):
    """Chronological split: first 15 calendar days train, remaining test."""
    if df.empty:
        return df, df
    d = pd.to_datetime(df["Signal Time"])
    cutoff = d.min() + pd.Timedelta(days=train_days)
    return df[d < cutoff].copy(), df[d >= cutoff].copy()


with tab_diagnosis:
    st.subheader("🧪 Strategy Validation — V9")
    st.caption(
        "V7 ka purpose promising score bands ko isolate karke validate karna hai. "
        "Default mein regime filter OFF hai, taaki score-band effect alag measure ho."
    )

    v7_interval = st.selectbox(
        "Validation candle interval", [5, 10, 15], index=0, key="v7_interval"
    )
    v7_stocks = st.multiselect(
        "Validation stocks",
        list(STOCKS.keys()),
        default=list(STOCKS.keys()),
        key="v7_stocks",
    )
    v7_hold = st.slider(
        "Maximum holding candles", 6, 48, 24, 3, key="v7_hold"
    )
    v7_regime = st.checkbox(
        "Apply EMA50/EMA200 regime filter",
        value=False,
        key="v7_regime",
    )

    st.markdown("**Focused tests:** SHORT 65–69, LONG 90+, plus broader comparison bands.")
    if st.button("🧪 Run V7 Validation", type="primary", key="v7_run"):
        if not v7_stocks:
            st.warning("कम से कम एक stock select करो.")
        else:
            tests = [
                ("SHORT", 65, 69),
                ("LONG", 90, 100),
                ("SHORT", 70, 100),
                ("LONG", 75, 100),
                ("SHORT", 55, 64),
                ("LONG", 55, 74),
            ]

            all_rows = []
            errors = []
            progress = st.progress(0)
            status = st.empty()

            total = len(v7_stocks) * len(tests)
            done = 0

            for symbol in v7_stocks:
                status.write(f"Validating {symbol}...")
                try:
                    data = get_historical_data(
                        STOCKS[symbol],
                        interval=v7_interval,
                        days_back=30,
                    )
                    if not data.empty:
                        for direction, lo, hi in tests:
                            trades = v7_backtest_symbol(
                                data, symbol, direction, lo, hi,
                                max_hold_bars=v7_hold,
                                use_regime=v7_regime,
                            )
                            all_rows.extend(trades)
                except Exception as e:
                    errors.append(f"{symbol}: {e}")

                done += len(tests)
                progress.progress(done / total)

            status.empty()

            if all_rows:
                dt = pd.DataFrame(all_rows)

                # Reconstruct test label from direction + score.
                def label(row):
                    if row["Direction"] == "SHORT" and 65 <= row["Score"] <= 69:
                        return "SHORT 65–69"
                    if row["Direction"] == "LONG" and 90 <= row["Score"] <= 100:
                        return "LONG 90+"
                    if row["Direction"] == "SHORT" and 70 <= row["Score"] <= 100:
                        return "SHORT 70+"
                    if row["Direction"] == "LONG" and 75 <= row["Score"] <= 100:
                        return "LONG 75+"
                    if row["Direction"] == "SHORT" and 55 <= row["Score"] <= 64:
                        return "SHORT 55–64"
                    if row["Direction"] == "LONG" and 55 <= row["Score"] <= 74:
                        return "LONG 55–74"
                    return "Other"

                dt["Test"] = dt.apply(label, axis=1)
                dt = dt[dt["Test"] != "Other"].copy()

                summary_rows = []
                for test, s in dt.groupby("Test"):
                    stt = v7_stats(s)
                    summary_rows.append({"Test": test, **stt})

                summary_df = pd.DataFrame(summary_rows)
                st.markdown("### Legacy V7 comparison (kept for reference)")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                stock_rows = []
                for (test, symbol), s in dt.groupby(["Test", "Symbol"]):
                    stt = v7_stats(s)
                    stock_rows.append({
                        "Test": test, "Symbol": symbol, **stt
                    })

                stock_df = pd.DataFrame(stock_rows)
                st.markdown("### Stock-level validation")
                st.dataframe(stock_df, use_container_width=True, hide_index=True)

                # Chronological walk-forward-style check on the focused setups.
                wf_rows = []
                for test, s in dt.groupby("Test"):
                    train, test_df = v7_walkforward_split(s, train_days=15)
                    tr = v7_stats(train)
                    te = v7_stats(test_df)
                    wf_rows.append({
                        "Test": test,
                        "Train Trades": tr["Trades"],
                        "Train Net R": tr["Net R"],
                        "Train PF": tr["Profit Factor"],
                        "Test Trades": te["Trades"],
                        "Test Net R": te["Net R"],
                        "Test PF": te["Profit Factor"],
                    })

                st.markdown("### Chronological validation (first 15 days vs remaining days)")
                st.dataframe(
                    pd.DataFrame(wf_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "⬇️ Download V7 Validation Trades",
                    dt.to_csv(index=False).encode(),
                    "intraday_edge_strategy_validation_v7.csv",
                    "text/csv",
                )
            else:
                st.warning("इस period में qualifying validation trade नहीं मिला.")

            if errors:
                with st.expander("Validation API / data errors"):
                    for err in errors:
                        st.write(err)


# ---------------- V9 Balanced Precision Lab ----------------
def v9_backtest_symbol(df, symbol, direction, min_score, max_score=100,
                       max_hold_bars=18, min_rvol=1.2, use_regime=False,
                       avoid_first_bars=3, avoid_last_bars=1):
    """V9: broader than V8, but still quality-controlled.
    - Direction-specific score
    - Same-session only
    - No overlapping trades
    - Optional EMA regime
    - Avoids opening/closing noise windows
    - Target 1.5R / risk 1R for a balanced hit-rate/expectancy test
    """
    if df.empty:
        return []
    work = add_indicators(df)
    work["Date"] = work["Datetime"].dt.date
    trades = []
    i = 220
    while i < len(work) - 1:
        row = work.iloc[i]
        score = score_direction(row, direction)
        if score is None or score < min_score or score > max_score:
            i += 1
            continue
        if row["RVOL"] < min_rvol:
            i += 1
            continue
        if use_regime:
            if direction == "LONG" and not (row["EMA50"] > row["EMA200"]):
                i += 1; continue
            if direction == "SHORT" and not (row["EMA50"] < row["EMA200"]):
                i += 1; continue
        day_rows = work.index[work["Date"] == row["Date"]].tolist()
        if not day_rows:
            i += 1; continue
        first_i, last_i = min(day_rows), max(day_rows)
        if i < first_i + avoid_first_bars or i > last_i - avoid_last_bars:
            i += 1; continue
        entry_idx = i + 1
        if entry_idx >= len(work): break
        entry_row = work.iloc[entry_idx]
        if entry_row["Date"] != row["Date"]:
            i += 1; continue
        entry = float(entry_row["Open"])
        atr = float(row["ATR14"])
        if not np.isfinite(atr) or atr <= 0:
            i += 1; continue
        risk = 1.0 * atr
        target_r = 1.5
        sl = entry - risk if direction == "LONG" else entry + risk
        target = entry + target_r*risk if direction == "LONG" else entry - target_r*risk
        exit_price = None; exit_time = None; outcome = "TIME_EXIT"
        exit_idx = min(last_i, entry_idx + max_hold_bars)
        for j in range(entry_idx, exit_idx + 1):
            bar = work.iloc[j]
            if direction == "LONG":
                hit_sl = bar["Low"] <= sl; hit_target = bar["High"] >= target
            else:
                hit_sl = bar["High"] >= sl; hit_target = bar["Low"] <= target
            if hit_sl and hit_target:
                exit_price = sl; outcome = "SL"
            elif hit_sl:
                exit_price = sl; outcome = "SL"
            elif hit_target:
                exit_price = target; outcome = "TARGET"
            if exit_price is not None:
                exit_time = bar["Datetime"]; break
        if exit_price is None:
            exit_price = float(work.iloc[exit_idx]["Close"])
            exit_time = work.iloc[exit_idx]["Datetime"]
        pnl_r = ((exit_price-entry)/risk) if direction == "LONG" else ((entry-exit_price)/risk)
        trades.append({"Symbol":symbol,"Direction":direction,"Score":round(score,1),
                       "Signal Time":row["Datetime"],"Entry Time":entry_row["Datetime"],
                       "Entry":round(entry,2),"SL":round(sl,2),"Target":round(target,2),
                       "Exit":round(exit_price,2),"Exit Time":exit_time,"Outcome":outcome,
                       "R":round(pnl_r,3),"RVOL":round(float(row["RVOL"]),2),
                       "RSI14":round(float(row["RSI14"]),2)})
        i = exit_idx + 1
    return trades

def v9_stats(df):
    if df.empty:
        return {"Trades":0,"Wins":0,"Losses":0,"Win Rate %":0.0,"Net R":0.0,"Profit Factor":0.0,"Max DD R":0.0}
    wins=int((df["R"]>0).sum()); losses=int((df["R"]<0).sum())
    gp=df.loc[df["R"]>0,"R"].sum(); gl=abs(df.loc[df["R"]<0,"R"].sum())
    eq=df["R"].cumsum(); dd=(eq-eq.cummax()).min()
    return {"Trades":len(df),"Wins":wins,"Losses":losses,"Win Rate %":round(100*wins/len(df),2),
            "Net R":round(df["R"].sum(),3),"Profit Factor":round(gp/gl,3) if gl else float("inf"),"Max DD R":round(dd,3)}

def v9_walkforward(df, train_ratio=0.5):
    if df.empty: return df,df
    d=pd.to_datetime(df["Signal Time"]).sort_values()
    cut=d.iloc[max(0,int(len(d)*train_ratio)-1)]
    return df[pd.to_datetime(df["Signal Time"])<=cut].copy(), df[pd.to_datetime(df["Signal Time"])>cut].copy()

with st.expander("🧠 V9 Balanced Precision Lab", expanded=True):
    st.write("V9 V8 se less restrictive hai: same-session only, controlled score/RVOL filters, aur walk-forward validation. 70–80% win rate ko claim nahi kiya jayega jab tak sufficient out-of-sample trades support na karein.")
    c1,c2,c3,c4=st.columns(4)
    v9_interval=c1.selectbox("Candle",[5,10,15],index=0,key="v9_interval")
    v9_score=c2.slider("Min score",55,85,60,5,key="v9_score")
    v9_rvol=c3.slider("Min RVOL",1.0,2.0,1.2,0.1,key="v9_rvol")
    v9_hold=c4.slider("Max hold",6,36,18,3,key="v9_hold")
    v9_regime=st.checkbox("EMA50/200 regime filter",False,key="v9_regime")
    v9_tests=[("SHORT",v9_score,100),("LONG",v9_score,100)]
    if st.button("🚀 Run V9 Validation",type="primary",key="v9_run"):
        rows=[]; errors=[]; progress=st.progress(0); status=st.empty()
        symbols=list(STOCKS.keys())
        for n,symbol in enumerate(symbols):
            status.write(f"V9 validating {symbol}...")
            try:
                data=get_historical_data(STOCKS[symbol],interval=v9_interval,days_back=30)
                for direction,lo,hi in v9_tests:
                    rows.extend(v9_backtest_symbol(data,symbol,direction,lo,hi,v9_hold,v9_rvol,v9_regime))
            except Exception as e: errors.append(f"{symbol}: {e}")
            progress.progress((n+1)/len(symbols))
        status.empty()
        if rows:
            d=pd.DataFrame(rows)
            overall=v9_stats(d)
            st.markdown("### V9 overall")
            a,b,c,d1,e,f=st.columns(6)
            a.metric("Trades",overall["Trades"]); b.metric("Win Rate",f'{overall["Win Rate %"]}%'); c.metric("Net R",overall["Net R"]); d1.metric("PF",overall["Profit Factor"]); e.metric("Max DD",overall["Max DD R"]); f.metric("Wins/Losses",f'{overall["Wins"]}/{overall["Losses"]}')
            summary=[]
            for direction,s in d.groupby("Direction"):
                summary.append({"Direction":direction,**v9_stats(s)})
            st.dataframe(pd.DataFrame(summary),use_container_width=True,hide_index=True)
            wf=[]
            for direction,s in d.groupby("Direction"):
                tr,te=v9_walkforward(s)
                a1=v9_stats(tr); a2=v9_stats(te)
                wf.append({"Direction":direction,"Train Trades":a1["Trades"],"Train WR":a1["Win Rate %"],"Train Net R":a1["Net R"],"Test Trades":a2["Trades"],"Test WR":a2["Win Rate %"],"Test Net R":a2["Net R"],"Test PF":a2["Profit Factor"]})
            st.markdown("### Chronological walk-forward")
            st.dataframe(pd.DataFrame(wf),use_container_width=True,hide_index=True)
            st.markdown("### Stock-level")
            stock=[]
            for (direction,symbol),s in d.groupby(["Direction","Symbol"]): stock.append({"Direction":direction,"Symbol":symbol,**v9_stats(s)})
            st.dataframe(pd.DataFrame(stock),use_container_width=True,hide_index=True)
            st.download_button("⬇️ Download V9 Validation CSV",d.to_csv(index=False).encode(),"intraday_edge_strategy_validation_v9.csv","text/csv")
        else: st.warning("V9 filters ke saath qualifying trades nahi mile.")
        if errors:
            with st.expander("V9 API / data errors"):
                for err in errors: st.write(err)

st.divider()
st.caption("Research / scanning only • No automatic order placement")
st.caption("Backtest results are historical simulations, not a guarantee of future performance.")
