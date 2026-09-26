import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Intraday Edge Scanner", page_icon="🎯", layout="wide")

st.title("🎯 Intraday Edge Scanner — V8")
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
    st.subheader("Rule-Based Backtest")
    st.caption(
        "Signal candle के close पर setup बनता है और entry अगले candle के open से शुरू होती है. "
        "Same-candle SL + target hit होने पर conservative assumption में SL माना जाता है. V7 में एक stock पर एक समय में केवल एक active trade होगा."
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
    """V7 validation: test one direction and one score band only.
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



def v8_precision_backtest(df, symbol, direction="SHORT", min_score=65, max_score=69,
                         max_hold_bars=24, sl_atr=1.2, target_atr=1.0,
                         require_regime=True, require_breakdown=True,
                         require_vwap=True, min_rvol=1.5,
                         rsi_min=None, rsi_max=None):
    """V8 precision test. Conditions are explicit, entry is next candle open,
    and only one active position per stock is allowed. Designed to test whether
    higher selectivity can improve hit rate without look-ahead."""
    if df.empty:
        return []
    work = add_indicators(df)
    trades = []
    i = 220
    while i < len(work) - 1:
        row = work.iloc[i]
        score = score_direction(row, direction)
        if score is None or score < min_score or score > max_score:
            i += 1
            continue

        if direction == "SHORT":
            if require_regime and not (row["EMA50"] < row["EMA200"]):
                i += 1; continue
            if require_breakdown and not bool(row["Breakdown"]):
                i += 1; continue
            if require_vwap and not (row["Close"] < row["VWAP"]):
                i += 1; continue
            if row["RVOL"] < min_rvol:
                i += 1; continue
        else:
            if require_regime and not (row["EMA50"] > row["EMA200"]):
                i += 1; continue
            if require_breakdown and not bool(row["Breakout"]):
                i += 1; continue
            if require_vwap and not (row["Close"] > row["VWAP"]):
                i += 1; continue
            if row["RVOL"] < min_rvol:
                i += 1; continue

        if rsi_min is not None and float(row["RSI14"]) < rsi_min:
            i += 1; continue
        if rsi_max is not None and float(row["RSI14"]) > rsi_max:
            i += 1; continue

        entry_idx = i + 1
        entry_row = work.iloc[entry_idx]
        entry = float(entry_row["Open"])
        atr = float(row["ATR14"])
        if not np.isfinite(atr) or atr <= 0:
            i += 1; continue

        if direction == "SHORT":
            sl = entry + sl_atr * atr
            target = entry - target_atr * atr
        else:
            sl = entry - sl_atr * atr
            target = entry + target_atr * atr

        exit_idx = min(len(work) - 1, entry_idx + max_hold_bars)
        exit_price = None; exit_time = None; outcome = "TIME_EXIT"
        for j in range(entry_idx, exit_idx + 1):
            bar = work.iloc[j]
            if direction == "SHORT":
                hit_sl = bar["High"] >= sl; hit_target = bar["Low"] <= target
            else:
                hit_sl = bar["Low"] <= sl; hit_target = bar["High"] >= target
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

        pnl_r = ((entry - exit_price) / (sl_atr * atr) if direction == "SHORT"
                 else (exit_price - entry) / (sl_atr * atr))
        trades.append({
            "Symbol": symbol, "Direction": direction, "Score": round(score,1),
            "Signal Time": row["Datetime"], "Entry Time": entry_row["Datetime"],
            "Entry": round(entry,2), "SL": round(sl,2), "Target": round(target,2),
            "Exit": round(exit_price,2), "Exit Time": exit_time,
            "Outcome": outcome, "R": round(pnl_r,3),
            "Target_RR": round(target_atr / sl_atr,3),
            "RVOL": round(float(row["RVOL"]),2), "RSI14": round(float(row["RSI14"]),2),
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
    st.subheader("🎯 Precision Strategy Lab — V8")
    st.caption("V8 ka goal high-selectivity setups ko test karna hai. 70–80% win rate target hai, guarantee nahi. Har test next-candle entry, one-position-per-stock aur conservative same-candle SL rule use karta hai.")

    v8_interval = st.selectbox("Validation candle interval", [5,10,15], index=0, key="v8_interval")
    v8_stocks = st.multiselect("Validation stocks", list(STOCKS.keys()), default=list(STOCKS.keys()), key="v8_stocks")
    v8_hold = st.slider("Maximum holding candles", 6, 48, 24, 3, key="v8_hold")
    v8_direction = st.selectbox("Direction", ["SHORT","LONG"], index=0, key="v8_direction")
    v8_min_score = st.slider("Minimum score", 55, 90, 65, 5, key="v8_min_score")
    v8_max_score = st.slider("Maximum score", 55, 100, 69, 1, key="v8_max_score")
    v8_sl = st.selectbox("Stop-loss ATR", [1.0,1.2,1.5], index=1, key="v8_sl")
    v8_target = st.selectbox("Target ATR", [0.8,1.0,1.2,1.5,2.0], index=1, key="v8_target")
    c1,c2,c3 = st.columns(3)
    with c1: v8_regime = st.checkbox("EMA50/200 regime", True, key="v8_regime")
    with c2: v8_structure = st.checkbox("Breakout/breakdown", True, key="v8_structure")
    with c3: v8_vwap = st.checkbox("VWAP alignment", True, key="v8_vwap")
    v8_rvol = st.selectbox("Minimum RVOL", [1.2,1.5,1.8,2.0], index=1, key="v8_rvol")
    v8_rsi = st.checkbox("Add RSI filter", False, key="v8_rsi")
    if v8_rsi:
        if v8_direction == "SHORT":
            r1,r2 = st.slider("SHORT RSI range", 20, 55, (28,45), key="v8_rsi_range")
        else:
            r1,r2 = st.slider("LONG RSI range", 45, 80, (55,72), key="v8_rsi_range")
    else:
        r1=r2=None

    if st.button("🎯 Run V8 Precision Test", type="primary", key="v8_run"):
        if not v8_stocks:
            st.warning("कम से कम एक stock select करो.")
        elif v8_min_score > v8_max_score:
            st.warning("Minimum score maximum score से बड़ा नहीं हो सकता.")
        else:
            rows=[]; errors=[]; progress=st.progress(0); status=st.empty()
            for i,symbol in enumerate(v8_stocks):
                status.write(f"Testing {symbol}...")
                try:
                    data=get_historical_data(STOCKS[symbol], interval=v8_interval, days_back=30)
                    if not data.empty:
                        rows.extend(v8_precision_backtest(
                            data, symbol, direction=v8_direction,
                            min_score=v8_min_score, max_score=v8_max_score,
                            max_hold_bars=v8_hold, sl_atr=v8_sl, target_atr=v8_target,
                            require_regime=v8_regime, require_breakdown=v8_structure,
                            require_vwap=v8_vwap, min_rvol=v8_rvol,
                            rsi_min=r1, rsi_max=r2))
                except Exception as e:
                    errors.append(f"{symbol}: {e}")
                progress.progress((i+1)/len(v8_stocks))
            status.empty()

            if rows:
                dt=pd.DataFrame(rows)
                stt=v7_stats(dt)
                a,b,c,d,e=st.columns(5)
                a.metric("Trades",stt["Trades"]); b.metric("Win Rate",f'{stt["Win Rate %"]}%')
                c.metric("Net R",stt["Net R"]); d.metric("Profit Factor",stt["Profit Factor"]); e.metric("Avg R",stt["Avg R"])
                st.dataframe(dt.sort_values("Signal Time", ascending=False),use_container_width=True,hide_index=True)

                # Chronological validation: first 15 calendar days vs remaining days.
                train,test=v7_walkforward_split(dt,15)
                wr=[]
                for name,x in [("First 15 days",train),("Later days",test)]:
                    z=v7_stats(x); wr.append({"Period":name,**z})
                st.markdown("### Chronological validation")
                st.dataframe(pd.DataFrame(wr),use_container_width=True,hide_index=True)
                st.download_button("⬇️ Download V8 Precision Trades",dt.to_csv(index=False).encode("utf-8"),"intraday_edge_precision_v8.csv","text/csv")
            else:
                st.warning("इन conditions पर कोई qualifying setup नहीं मिला.")
            if errors:
                with st.expander("API / data errors"):
                    for err in errors: st.write(err)

st.divider()
st.caption("Research / scanning only • No automatic order placement")
st.caption("Backtest results are historical simulations, not a guarantee of future performance.")
