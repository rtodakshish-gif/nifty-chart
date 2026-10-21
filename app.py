"""
LIVE NIFTY 50 chart app  ->  local: http://localhost:8001  |  cloud (Render): your https link   (updates every 2 seconds)
-----------------------------------------------------------------------------------------------------------
Data: free Yahoo Finance (^NSEI). It can be slightly delayed. No API key, no broker login, NO orders.
Time frames: 1m, 5m, 15m, 30m, 1h, 1D, 1W (buttons at the top).
Shows: candles (incl. the running candle), Supertrend(10,3), EMA20, and BUY/SELL signals on EVERY time frame:
  BIG BUY/SELL  = Supertrend flip confirmed by close vs EMA20 (the bot's rule)
  small buy/sell = Supertrend flip without EMA20 confirmation (dim)
plus a 1h hedged option-selling TRADE PLAN card (manual orders; optional ntfy phone alerts), a panel with the latest signal of every time frame, and a simulated result (stop-loss %, same-day exit).
Needs: pip install requests pandas numpy      (internet is also needed for the chart library)
Run:   python nifty_chart_server.py       Stop: Ctrl + C
Phone on the same WiFi: open the address printed as "Phone" below.
"""
import datetime as dt
import json
import math
import os
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests

SYMBOL = "%5ENSEI"                       # ^NSEI = Nifty 50 on Yahoo Finance
# time frame -> (yahoo interval, yahoo range, seconds per candle)
TFS = {"1m": ("1m", "5d", 60), "5m": ("5m", "10d", 300), "15m": ("15m", "30d", 900),
       "30m": ("30m", "60d", 1800), "1h": ("60m", "6mo", 3600), "1d": ("1d", "2y", 86400), "1w": ("1wk", "5y", 604800), "1mo": ("1mo", "10y", 2592000)}
ST_PERIOD, ST_MULT, EMA_LEN = 10, 3, 20
SL_PCT = 0.5                             # stop-loss used in the simulation (% of price)
PORT = int(os.environ.get("PORT", 8001))   # cloud hosts give the port in $PORT
CLOUD = "PORT" in os.environ
HOST = "0.0.0.0"                         # "127.0.0.1" = only this PC; "0.0.0.0" = phones on your WiFi too
SHIFT = 19800                            # show IST (UTC+5:30)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# ---- trade-plan settings (1h hedged option selling; you place orders yourself in Kite) ----
LOT = 65                                 # Nifty lot size
STRIKE_STEP = 50
SELL_PREMIUM = 200                       # target premium of the option you SELL (Rs)
HEDGE_PREMIUM = 35                       # target premium of the far-OTM option you BUY (Rs)
PREMIUM_SL_PCT = 30                      # stop-loss = sell premium + 30%
EXPIRY_WEEKDAY = 1                       # 0=Mon,1=Tue ... weekly expiry day (please verify in Kite)
RISK_FREE, FALLBACK_IV = 0.07, 0.14
PLAN_TF = "1h"
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")   # optional phone alerts via ntfy.sh

cache = {}                               # tf -> {"json": str, "t": float}
lock = threading.Lock()
http = requests.Session()                 # keep-alive: no new TLS handshake for every request
tf_locks = {k: threading.Lock() for k in TFS}
active = {}                               # tf -> last time a browser asked for it


# ---------------- data + indicators ----------------
def fetch(tf):
    interval, rng, _ = TFS[tf]
    last_err = None
    for host in ("query1", "query2"):
        try:
            url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{SYMBOL}"
            r = http.get(url, params={"interval": interval, "range": rng, "includePrePost": "false"},
                         headers=HEADERS, timeout=8)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({"time": res["timestamp"], "open": q["open"], "high": q["high"],
                               "low": q["low"], "close": q["close"]}).dropna()
            df["time"] = df["time"].astype(int)
            df = df.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
            if len(df) < 30:
                raise RuntimeError("Yahoo returned too little data")
            df.attrs["tick"] = res.get("meta", {}).get("regularMarketTime")
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Yahoo data error: {last_err}")


def supertrend(df):
    hl2 = (df.high + df.low) / 2
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / ST_PERIOD, adjust=False).mean()
    ub, lb, c = (hl2 + ST_MULT * atr).values, (hl2 - ST_MULT * atr).values, df.close.values
    fub, flb, d = ub.copy(), lb.copy(), np.ones(len(df))
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        d[i] = (-1 if c[i] < flb[i] else 1) if d[i - 1] == 1 else (1 if c[i] > fub[i] else -1)
    return d, np.where(d == 1, flb, fub)


def simulate(df, d, intraday):
    """Enter on a Supertrend flip confirmed by EMA20; exit on opposite flip, SL, or (intraday) end of day."""
    close, high, low, ema = df.close.values, df.high.values, df.low.values, df.ema.values
    day = ((df.time.values + SHIFT) // 86400)
    trades, i, n = [], 1, len(df)
    while i < n:
        flip = d[i] != d[i - 1] and day[i] == day[i - 1] or (d[i] != d[i - 1] and not intraday)
        long_ok = flip and d[i] == 1 and close[i] > ema[i]
        short_ok = flip and d[i] == -1 and close[i] < ema[i]
        if not (long_ok or short_ok):
            i += 1
            continue
        side = 1 if long_ok else -1
        entry = close[i]
        sl = entry * (1 - SL_PCT / 100 * side)
        j, exit_px, why = i + 1, None, None
        while j < n:
            if intraday and day[j] != day[i]:
                exit_px, why, j = close[j - 1], "EOD", j - 1
                break
            hit = low[j] <= sl if side == 1 else high[j] >= sl
            if hit:
                exit_px, why = sl, "SL"
                break
            if d[j] != d[i]:
                exit_px, why = close[j], "FLIP"
                break
            j += 1
        if exit_px is None:
            trades.append(dict(side=side, i=i, entry=entry, j=None, exit=close[-1], why="OPEN"))
            break
        trades.append(dict(side=side, i=i, entry=entry, j=j, exit=exit_px, why=why))
        i = j + 1 if why in ("SL", "EOD") else j
    return trades


def market_status():
    ist = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=SHIFT)
    open_now = ist.weekday() < 5 and dt.time(9, 15) <= ist.time() <= dt.time(15, 30)
    return "OPEN" if open_now else "CLOSED"


def market_structure(df, n_closed, t, N):
    """Swing highs/lows (pivot with N candles each side) and breakouts of them on CLOSED candles.
    BOS = close breaks the last swing in the trend direction; CHoCH = break against the prior trend."""
    high, low, close = df.high.values, df.low.values, df.close.values
    ph = pl = None
    trend, events = 0, []
    for i in range(N, n_closed):
        j = i - N                                   # pivot at j is confirmed only now (no look-ahead)
        if j >= N:
            if high[j] >= high[j - N:j + N + 1].max():
                ph = (j, float(high[j]))
            if low[j] <= low[j - N:j + N + 1].min():
                pl = (j, float(low[j]))
        c = float(close[i])
        if ph and c > ph[1]:
            events.append(dict(kind="CHoCH" if trend == -1 else "BOS", up=True, t0=t[ph[0]], t1=t[i],
                               price=round(ph[1], 2)))
            trend, ph = 1, None
        elif pl and c < pl[1]:
            events.append(dict(kind="CHoCH" if trend == 1 else "BOS", up=False, t0=t[pl[0]], t1=t[i],
                               price=round(pl[1], 2)))
            trend, pl = -1, None
    return dict(events=events[-14:], trend=trend,
                high=dict(t=t[ph[0]], price=round(ph[1], 2)) if ph else None,
                low=dict(t=t[pl[0]], price=round(pl[1], 2)) if pl else None)


def liquidity_zones(df, n_closed, t, N, atr):
    """Liquidity = swing highs/lows where stop-losses sit. EQH/EQL = two swings at (almost) the same level.
    Sweep = a candle pokes beyond a swing level but CLOSES back inside (stop-hunt). Closed candles only."""
    high, low, close = df.high.values, df.low.values, df.close.values
    levels, sweeps = [], []
    for i in range(N, n_closed):
        j = i - N
        if j >= N:
            tol = 0.25 * float(atr[i])
            for side, arr, ok in (("H", high, high[j] >= high[j - N:j + N + 1].max()),
                                  ("L", low, low[j] <= low[j - N:j + N + 1].min())):
                if not ok:
                    continue
                lv = dict(side=side, j=j, price=float(arr[j]), eq=False, done=False)
                for o in levels:                    # equal high / equal low with an earlier untouched swing
                    if o["side"] == side and not o["done"] and abs(o["price"] - lv["price"]) <= tol:
                        o["eq"] = lv["eq"] = True
                levels.append(lv)
            levels = levels[-80:]
        for lv in levels:
            if lv["done"]:
                continue
            if lv["side"] == "H" and high[i] > lv["price"]:
                lv["done"] = True
                if close[i] < lv["price"]:
                    sweeps.append(dict(up=False, eq=lv["eq"], t0=t[lv["j"]], t1=t[i], price=round(lv["price"], 2)))
            elif lv["side"] == "L" and low[i] < lv["price"]:
                lv["done"] = True
                if close[i] > lv["price"]:
                    sweeps.append(dict(up=True, eq=lv["eq"], t0=t[lv["j"]], t1=t[i], price=round(lv["price"], 2)))
    pools = [dict(side=lv["side"], t0=t[lv["j"]], price=round(lv["price"], 2))
             for lv in levels if lv["eq"] and not lv["done"]]
    return dict(sweeps=sweeps[-12:], pools=pools[-6:], end=t[n_closed - 1])


def closed_count(df, sec, now):
    """How many candles are finished (the last one may still be running)."""
    last = int(df.iloc[-1]["time"])
    if sec >= 2592000:                             # monthly candle runs until the month ends
        d0 = dt.datetime.fromtimestamp(last + SHIFT, dt.timezone.utc)
        nxt = (d0.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        running = now < int(nxt.timestamp()) - SHIFT
    elif sec >= 604800:                            # weekly candle runs until Friday close
        running = now < last + 5 * 86400
    else:
        running = market_status() == "OPEN" and last + sec > now
    return len(df) - 1 if running else len(df)


def fmt_sig_time(ts, sec):
    """IST time when the signal candle CLOSED."""
    if sec >= 2592000:                             # monthly: show the month
        return dt.datetime.fromtimestamp(ts + SHIFT, dt.timezone.utc).strftime("%b %Y")
    if sec >= 604800:                              # weekly: show the Friday the week ended
        return dt.datetime.fromtimestamp(ts + 4 * 86400 + SHIFT, dt.timezone.utc).strftime("%d-%b")
    x = dt.datetime.fromtimestamp(ts + sec + SHIFT, dt.timezone.utc)
    if sec < 86400 and (x.hour, x.minute) > (15, 30):      # last 1h candle is only 15 min long
        x = x.replace(hour=15, minute=30)
    return x.strftime("%d-%b") if sec >= 86400 else x.strftime("%d-%b %H:%M")


def find_flips(df, d, n_closed, sec):
    """Every Supertrend flip on a closed candle. ok = also confirmed by close vs EMA20 (bot rule)."""
    out = []
    for i in range(1, n_closed):
        if d[i] != d[i - 1]:
            buy = bool(d[i] == 1)
            c, e = float(df.close[i]), float(df.ema[i])
            out.append(dict(i=i, buy=buy, ok=(c > e) if buy else (c < e), price=round(c, 2),
                            time=fmt_sig_time(int(df.time[i]), sec)))
    return out


raw = {}                                 # tf -> {"sig": dict, "t": float}   (for the all-time-frame panel)
sig_lock = threading.Lock()


def tf_signal(tf):
    """Latest signal of one time frame (data cached 30 s)."""
    c = raw.get(tf)
    if c is None or time.time() - c["t"] > 30:
        interval, rng, sec = TFS[tf]
        df = fetch(tf)
        df["ema"] = df.close.ewm(span=EMA_LEN, adjust=False).mean()
        d, _ = supertrend(df)
        now = int(time.time())
        n_closed = closed_count(df, sec, now)
        flips = find_flips(df, d, n_closed, sec)
        sig = dict(tf=tf, trend="GREEN" if d[-1] == 1 else "RED",
                   signal=(dict(side="BUY" if flips[-1]["buy"] else "SELL", ok=flips[-1]["ok"],
                                price=flips[-1]["price"], time=flips[-1]["time"]) if flips else None))
        c = raw[tf] = {"sig": sig, "t": time.time()}
    return c["sig"]


def all_signals():
    out = []
    for tf in TFS:
        try:
            out.append(tf_signal(tf))
        except Exception as e:
            out.append(dict(tf=tf, trend=None, signal=None, error=str(e)))
    return out


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, iv, call):
    """Black-Scholes estimate (flat IV) - only an approximation of the real premium."""
    if T <= 0:
        return max(0.0, (S - K) if call else (K - S))
    d1 = (math.log(S / K) + (RISK_FREE + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if call:
        return S * _ncdf(d1) - K * math.exp(-RISK_FREE * T) * _ncdf(d2)
    return K * math.exp(-RISK_FREE * T) * _ncdf(-d2) - S * _ncdf(-d1)


def strike_for(S, T, iv, call, target):
    atm = round(S / STRIKE_STEP) * STRIKE_STEP
    best = None
    for k in range(0, 300):
        K = atm + (k if call else -k) * STRIKE_STEP
        p = bs_price(S, K, T, iv, call)
        if best is None or abs(p - target) < abs(best[1] - target):
            best = (K, p)
        if p < target * 0.3:
            break
    return best


vix_cache = {"v": FALLBACK_IV, "t": 0}


def get_iv():
    if time.time() - vix_cache["t"] < 600:
        return vix_cache["v"]
    try:
        r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/%5EINDIAVIX",
                         params={"interval": "1d", "range": "5d"}, headers=HEADERS, timeout=10)
        closes = [c for c in r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"] if c]
        vix_cache["v"] = closes[-1] / 100.0
    except Exception:
        pass
    vix_cache["t"] = time.time()
    return vix_cache["v"]


def next_expiry(now_ist):
    add = (EXPIRY_WEEKDAY - now_ist.weekday()) % 7
    if add == 0 and now_ist.time() >= dt.time(15, 30):
        add = 7
    return now_ist.date() + dt.timedelta(days=add)


plan_cache = {}


def make_plan():
    """Latest 1h signal + an estimated hedged option-selling plan (you place the orders yourself)."""
    c = plan_cache.get("p")
    if c and time.time() - c["t"] < 30:
        return c["v"]
    sec = TFS[PLAN_TF][2]
    df = fetch(PLAN_TF)
    df["ema"] = df.close.ewm(span=EMA_LEN, adjust=False).mean()
    d, _ = supertrend(df)
    now = int(time.time())
    n_closed = closed_count(df, sec, now)
    flips = find_flips(df, d, n_closed, sec)
    spot = float(df.close.iloc[-1])
    out = dict(tf=PLAN_TF, spot=round(spot, 2), trend="GREEN" if d[-1] == 1 else "RED", signal=None, plan=None,
               lot=LOT)
    if flips:
        f = flips[-1]
        close_ts = int(df.time[f["i"]]) + sec
        over = (close_ts + SHIFT) % 86400 - (15 * 3600 + 1800)
        if over > 0:
            close_ts -= over
        age = max(0, (now - close_ts) // 60)
        today = (close_ts + SHIFT) // 86400 == (now + SHIFT) // 86400
        late = (close_ts + SHIFT) % 86400 >= 15 * 3600 + 900
        out["signal"] = dict(side="BUY" if f["buy"] else "SELL", ok=f["ok"], price=f["price"], time=f["time"],
                             age_min=int(age), fresh=bool(f["ok"] and today and age <= 90), late=bool(late))
        if f["ok"]:
            call = not f["buy"]                 # SELL signal -> sell CE ; BUY signal -> sell PE
            now_ist = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=SHIFT)
            exp = next_expiry(now_ist)
            iv = get_iv()
            atm = round(spot / STRIKE_STEP) * STRIKE_STEP
            for _ in range(4):                  # nearest expiry too cheap for the target premium -> next week
                end = dt.datetime.combine(exp, dt.time(15, 30))
                T = max((end - now_ist.replace(tzinfo=None)).total_seconds() / (365 * 86400), 1 / (365 * 24))
                if bs_price(spot, atm, T, iv, call) >= SELL_PREMIUM:
                    break
                exp += dt.timedelta(days=7)
            sK, sp = strike_for(spot, T, iv, call, SELL_PREMIUM)
            hK, hp = strike_for(spot, T, iv, call, HEDGE_PREMIUM)
            if (call and hK <= sK) or ((not call) and hK >= sK):
                hK = sK + STRIKE_STEP * (1 if call else -1)
                hp = bs_price(spot, hK, T, iv, call)
            sp, hp = round(sp / 5) * 5, max(round(hp / 5) * 5, 5)
            out["plan"] = dict(kind="CE" if call else "PE", sell_strike=int(sK), sell_prem=sp,
                               hedge_strike=int(hK), hedge_prem=hp, expiry=exp.strftime("%d-%b-%Y"),
                               sl_prem=round(sp * (1 + PREMIUM_SL_PCT / 100) / 5) * 5,
                               max_loss=int(max(0, abs(hK - sK) - (sp - hp)) * LOT),
                               credit=int((sp - hp) * LOT), iv=round(iv * 100, 1))
    plan_cache["p"] = {"v": out, "t": time.time()}
    return out


def send_ntfy(p):
    s, pl = p["signal"], p["plan"]
    body = (f"{s['side']} signal 1h ({s['time']}) | Nifty {p['spot']}\n"
            f"SELL {pl['sell_strike']} {pl['kind']} (~Rs {pl['sell_prem']}) + BUY {pl['hedge_strike']} {pl['kind']} "
            f"(~Rs {pl['hedge_prem']}) | Expiry {pl['expiry']}\n"
            f"SL: premium Rs {pl['sl_prem']} or 1h Supertrend flip. Check real premiums in Kite.")
    requests.post("https://ntfy.sh/" + NTFY_TOPIC, data=body.encode("utf-8"), timeout=10,
                  headers={"Title": "NIFTY 1h signal", "Priority": "high", "Tags": "chart_with_upwards_trend"})


def alert_loop():
    last, first = None, True
    while True:
        try:
            p = make_plan()
            s = p.get("signal")
            key = (s["side"], s["time"]) if s else None
            if first:
                last, first = key, False
            elif key and key != last:
                last = key
                if s["ok"] and s["fresh"] and not s["late"] and p["plan"]:
                    send_ntfy(p)
        except Exception as e:
            print("alert note:", e)
        time.sleep(60)


def build_json(tf):
    interval, rng, sec = TFS[tf]
    intraday = sec < 86400
    df = fetch(tf)
    tick = df.attrs.get("tick")
    lag = int(time.time() - tick) if (tick and market_status() == "OPEN") else None
    df["ema"] = df.close.ewm(span=EMA_LEN, adjust=False).mean()
    d, line = supertrend(df)

    # a candle is "running" while the market is open and the candle has not finished
    now = int(time.time())
    n_closed = closed_count(df, sec, now)
    closed = df.iloc[:n_closed].reset_index(drop=True)
    trades = simulate(closed, d[:n_closed], intraday)

    t = (df["time"] + SHIFT).tolist()
    candles = [dict(time=t[i], open=round(float(df.open[i]), 2), high=round(float(df.high[i]), 2),
                    low=round(float(df.low[i]), 2), close=round(float(df.close[i]), 2)) for i in range(len(df))]
    ema = [dict(time=t[i], value=round(float(df.ema[i]), 2)) for i in range(len(df))]
    st_up = [dict(time=t[i], value=round(float(line[i]), 2)) if d[i] == 1 else dict(time=t[i]) for i in range(len(df))]
    st_dn = [dict(time=t[i], value=round(float(line[i]), 2)) if d[i] == -1 else dict(time=t[i]) for i in range(len(df))]

    # signals: every Supertrend flip on a closed candle (bright = EMA20-confirmed = bot rule, dim = not confirmed)
    flips = find_flips(df, d, n_closed, sec)
    markers = []
    for f in flips:
        ti = t[f["i"]]
        if f["buy"]:
            markers.append(dict(time=ti, position="belowBar", shape="arrowUp",
                                color="#00e676" if f["ok"] else "#6b8f71", text="BUY" if f["ok"] else "buy"))
        else:
            markers.append(dict(time=ti, position="aboveBar", shape="arrowDown",
                                color="#ff5252" if f["ok"] else "#9a6b6b", text="SELL" if f["ok"] else "sell"))
    for tr in trades:                    # simulated exits that are not a plain flip
        if tr["j"] is not None and tr["why"] in ("SL", "EOD"):
            markers.append(dict(time=t[tr["j"]], position="inBar", color="#b0bec5", shape="circle", text=tr["why"]))
    markers.sort(key=lambda m: m["time"])

    done = [x for x in trades if x["j"] is not None]
    pts = [(x["exit"] - x["entry"]) * x["side"] for x in done]
    pct = [(x["exit"] - x["entry"]) / x["entry"] * 100 * x["side"] for x in done]
    if pts:
        wins = sum(1 for p in pts if p > 0)
        summary = (f"Sim {tf}, SL {SL_PCT}%: {len(pts)} trades, win {wins / len(pts) * 100:.0f}%, "
                   f"total {sum(pts):+.0f} pts ({sum(pct):+.2f}%), no costs")
    else:
        summary = f"Sim {tf}: no completed trades in this period"

    struct = market_structure(df, n_closed, t, 3 if sec >= 604800 else 5)
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    liq = liquidity_zones(df, n_closed, t, 3 if sec >= 604800 else 5, tr.rolling(14, min_periods=1).mean().values)
    last_sig = (dict(side="BUY" if flips[-1]["buy"] else "SELL", ok=flips[-1]["ok"],
                     price=flips[-1]["price"], time=flips[-1]["time"]) if flips else None)

    # change vs previous day's close
    last_day = (df["time"].iloc[-1] + SHIFT) // 86400
    prev = df[((df["time"] + SHIFT) // 86400) < last_day]
    prev_close = float(prev["close"].iloc[-1]) if len(prev) else float(df["close"].iloc[0])
    price = float(df.close.iloc[-1])

    return json.dumps(dict(
        tf=tf, candles=candles, ema=ema, st_up=st_up, st_dn=st_dn, markers=markers,
        last=dict(price=price, chg=price - prev_close, chg_pct=(price - prev_close) / prev_close * 100,
                  st="GREEN" if d[-1] == 1 else "RED", ema=float(df.ema.iloc[-1]),
                  market=market_status(), lag=lag,
                  time=(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=SHIFT)).strftime("%d-%b %H:%M:%S IST")),
        summary=summary, last_signal=last_sig, structure=struct, liquidity=liq))


# ---------------- web page ----------------
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>NIFTY 50 live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#0f1218">
<meta name="apple-mobile-web-app-capable" content="yes">
<link rel="apple-touch-icon" href="/icon-192.png">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
html,body{margin:0;height:100%;background:#0f1218;color:#d1d4dc;font-family:Segoe UI,Arial,sans-serif}
body{display:flex;flex-direction:column}
#top{padding:8px 12px;font-size:14px;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
#name{font-weight:700;color:#ff9800}
#price{font-size:20px}
#tfs button{background:#1c212b;color:#d1d4dc;border:1px solid #2b3139;border-radius:6px;padding:5px 11px;
  margin-right:4px;font-size:13px;cursor:pointer}
#tfs button.on{background:#ff6d00;color:#fff;border-color:#ff6d00}
#chart{flex:1;min-height:0}
#sigs{padding:2px 12px 8px;display:flex;gap:6px;flex-wrap:wrap}
.chip{background:#1c212b;border:1px solid #2b3139;border-radius:6px;padding:4px 9px;font-size:12px;cursor:pointer;line-height:1.35}
.chip.cur{border-color:#ff6d00}
#plan{margin:2px 12px 8px;padding:8px 10px;background:#161b24;border:1px solid #2b3139;border-radius:8px;font-size:13px;line-height:1.5}
#plan:empty{display:none}
.hid{display:none!important}
.tg{background:#1c212b;color:#d1d4dc;border:1px solid #2b3139;border-radius:6px;padding:4px 9px;font-size:12px;cursor:pointer}
.chip b{font-size:13px}
</style></head><body>
<div id="top"><span id="name">NIFTY 50</span><b id="price">loading...</b><span id="chg"></span><span id="tfs"></span>
<span id="st"></span><span id="ema"></span><span id="mkt"></span><span id="lastsig"></span><span id="hides"><button class="tg" id="b_ref" onclick="hardRefresh()">&#8635; Refresh</button> <button class="tg" id="b_plan" onclick="toggleBox('hp')">Hide Plan</button> <button class="tg" id="b_sigs" onclick="toggleBox('hs')">Hide Signals</button> <button class="tg" id="b_str" onclick="toggleStruct()">Hide Structure</button> <button class="tg" id="b_liq" onclick="toggleLiq()">Hide Liquidity</button> <button class="tg" id="b_info" onclick="toggleBox('hi')">Hide Info</button></span></div>
<div id="plan"></div>
<div id="sigs"></div>
<div id="info">
<div style="padding:0 12px 4px;font-size:12px;opacity:.8"><span id="sum"></span> <span id="upd" style="opacity:.7"></span></div>
<div style="padding:0 12px 6px;font-size:11px;opacity:.65">BUY / SELL (big) = Supertrend flip + EMA20 confirmed (bot rule) &nbsp;|&nbsp; buy / sell (dim) = flip without EMA20 confirmation &nbsp;|&nbsp; signals are given only after the candle closes<br>Structure: BOS = close breaks the last swing high/low (trend continues) | CHoCH = break against the earlier trend (possible reversal) | dashed grey lines = latest unbroken swing high/low<br>Liquidity: EQH/EQL (orange) = equal highs/lows where stop-losses pile up | Sweep (cyan) = price poked beyond a swing level but closed back inside (stop-hunt), often a reversal hint</div>
</div>
<div id="chart"></div>
<script>
const TFS = ['1m','5m','15m','30m','1h','1d','1w','1mo'];
const LBL = {'1d':'1D', '1w':'1W', '1mo':'1M'};
let tf = '15m';
try { tf = localStorage.getItem('ntf') || '15m'; } catch(e){}
if (!TFS.includes(tf)) tf = '15m';

const chart = LightweightCharts.createChart(document.getElementById('chart'), {
  autoSize:true,
  layout:{background:{color:'#0f1218'}, textColor:'#d1d4dc'},
  grid:{vertLines:{color:'#1c212b'}, horzLines:{color:'#1c212b'}},
  timeScale:{timeVisible:true, secondsVisible:false}, rightPriceScale:{borderColor:'#2b3139'}});
const candles = chart.addCandlestickSeries({upColor:'#26a69a', downColor:'#ef5350', borderVisible:false,
  wickUpColor:'#26a69a', wickDownColor:'#ef5350'});
const opt = (c, w) => ({color:c, lineWidth:w, priceLineVisible:false, lastValueVisible:false});
const stUp = chart.addLineSeries(opt('#00c853', 2));
const stDn = chart.addLineSeries(opt('#ff1744', 2));
const ema  = chart.addLineSeries(opt('#2196f3', 1));
let first = true, lastT = 0;

const box = document.getElementById('tfs');
TFS.forEach(x => {
  const b = document.createElement('button');
  b.textContent = LBL[x] || x; b.id = 'tf_' + x;
  b.onclick = () => {
    tf = x; try { localStorage.setItem('ntf', x); } catch(e){}
    first = true; lastT = 0; clearStructure(); clearLiquidity(); mark(); tick(); loadSigs();
  };
  box.appendChild(b);
});
function mark(){ TFS.forEach(x => document.getElementById('tf_' + x).className = (x === tf ? 'on' : '')); }
mark();

let structSeries = [], structLines = [], structKey = '', lastStruct = null, showStruct = true;
try { showStruct = localStorage.getItem('st') !== '0'; } catch(e){}
function clearStructure(){
  structSeries.forEach(x => { try { chart.removeSeries(x); } catch(e){} });
  structLines.forEach(x => { try { candles.removePriceLine(x); } catch(e){} });
  structSeries = []; structLines = []; structKey = '';
}
function drawStructure(st){
  lastStruct = st;
  document.getElementById('b_str').textContent = (showStruct ? 'Hide' : 'Show') + ' Structure';
  const key = showStruct ? (tf + JSON.stringify(st)) : 'off';
  if (key === structKey) return;
  clearStructure(); structKey = key;
  if (!showStruct || !st) return;
  st.events.forEach(e => {
    const col = e.kind === 'CHoCH' ? '#ce93d8' : '#ffd54f';
    const s = chart.addLineSeries({color: col, lineWidth: 1, lineStyle: 2, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false});
    s.setData([{time: e.t0, value: e.price}, {time: e.t1, value: e.price}]);
    s.setMarkers([{time: e.t1, position: e.up ? 'belowBar' : 'aboveBar', color: col, shape: 'circle',
      text: e.kind + (e.up ? ' up' : ' down')}]);
    structSeries.push(s);
  });
  [[st.high, 'Swing H'], [st.low, 'Swing L']].forEach(a => {
    if (a[0]) structLines.push(candles.createPriceLine({price: a[0].price, color: '#78909c', lineWidth: 1,
      lineStyle: 2, axisLabelVisible: true, title: a[1]}));
  });
}
function toggleStruct(){
  showStruct = !showStruct;
  try { localStorage.setItem('st', showStruct ? '1' : '0'); } catch(e){}
  structKey = '';
  drawStructure(lastStruct);
}

let liqSeries = [], liqKey = '', lastLiq = null, showLiq = true;
try { showLiq = localStorage.getItem('lq') !== '0'; } catch(e){}
function clearLiquidity(){
  liqSeries.forEach(x => { try { chart.removeSeries(x); } catch(e){} });
  liqSeries = []; liqKey = '';
}
function drawLiquidity(lq){
  lastLiq = lq;
  document.getElementById('b_liq').textContent = (showLiq ? 'Hide' : 'Show') + ' Liquidity';
  const key = showLiq ? (tf + JSON.stringify(lq)) : 'off';
  if (key === liqKey) return;
  clearLiquidity(); liqKey = key;
  if (!showLiq || !lq) return;
  const mk = (t0, t1, price, col, text, pos) => {
    if (!(t1 > t0)) return;
    const s = chart.addLineSeries({color: col, lineWidth: 1, lineStyle: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false});
    s.setData([{time: t0, value: price}, {time: t1, value: price}]);
    s.setMarkers([{time: (pos === 'start' ? t0 : t1), position: 'inBar', color: col, shape: 'square', text: text}]);
    liqSeries.push(s);
  };
  lq.pools.forEach(p => mk(p.t0, lq.end, p.price, '#ffa726', p.side === 'H' ? 'EQH' : 'EQL', 'start'));
  lq.sweeps.forEach(e => mk(e.t0, e.t1, e.price, '#26c6da', (e.eq ? 'EQ ' : '') + 'Sweep ' + (e.up ? 'low' : 'high'), 'end'));
}
function toggleLiq(){
  showLiq = !showLiq;
  try { localStorage.setItem('lq', showLiq ? '1' : '0'); } catch(e){}
  liqKey = '';
  drawLiquidity(lastLiq);
}

const hideState = {hp: '0', hs: '0', hi: '0'};
try { hideState.hp = localStorage.getItem('hp') || '0'; hideState.hs = localStorage.getItem('hs') || '0'; hideState.hi = localStorage.getItem('hi') || '0'; } catch(e){}
function applyHide(){
  [['hp','plan','Plan'], ['hs','sigs','Signals'], ['hi','info','Info']].forEach(a => {
    const off = hideState[a[0]] === '1';
    document.getElementById(a[1]).classList.toggle('hid', off);
    document.getElementById('b_' + a[1]).textContent = (off ? 'Show ' : 'Hide ') + a[2];
  });
}
function toggleBox(k){
  hideState[k] = hideState[k] === '1' ? '0' : '1';
  try { localStorage.setItem(k, hideState[k]); } catch(e){}
  applyHide();
}
applyHide();

async function loadPlan(f){
  try{
    const r = await fetch('/plan?_=' + Date.now() + (f ? '&force=1' : ''));
    const p = await r.json();
    const el = document.getElementById('plan');
    if (p.error){ el.innerHTML = ''; return; }
    const s = p.signal, pl = p.plan;
    let h = '<b>1h trade plan</b> (you place orders in Kite) <span style="opacity:.7">| Nifty ' + p.spot.toLocaleString() + ', lot ' + p.lot + '</span><br>';
    if (!s){ el.innerHTML = h + 'No 1h signal in this data yet.'; return; }
    const buy = s.side === 'BUY';
    h += '1h signal: <b style="color:' + (buy ? '#00e676' : '#ff5252') + '">' + s.side + (s.ok ? ' &#10003;' : ' (no EMA)') + '</b> '
      + s.time + ' (' + s.age_min + ' min ago)';
    if (!s.ok){ el.innerHTML = h + '<br>No EMA20 confirmation, so NO trade by the bot rule.'; return; }
    if (!s.fresh) h += '<br><span style="color:#ffb300">Old signal - do not enter now; wait for the next 1h signal.</span>';
    else if (s.late) h += '<br><span style="color:#ffb300">Signal came at 15:15 or later - better to skip a new entry today.</span>';
    if (pl){
      h += '<br>SELL <b>' + pl.sell_strike + ' ' + pl.kind + '</b> (~Rs ' + pl.sell_prem + ') + BUY hedge <b>' + pl.hedge_strike + ' ' + pl.kind + '</b> (~Rs ' + pl.hedge_prem + ')'
        + '<br>Expiry ' + pl.expiry + ' | credit ~Rs ' + pl.credit.toLocaleString() + ' | max loss ~Rs ' + pl.max_loss.toLocaleString()
        + '<br>Exit: sell premium reaches Rs ' + pl.sl_prem + ', or 1h Supertrend flips against you.'
        + '<br><span style="opacity:.65;font-size:11px">Premiums are model estimates (IV ' + pl.iv + '%). Check real prices and margin in Kite before ordering. Not financial advice.</span>';
    }
    el.innerHTML = h;
  } catch(e){}
}

async function loadSigs(f){
  try{
    const r = await fetch('/signals?_=' + Date.now() + (f ? '&force=1' : ''));
    const arr = await r.json();
    const box = document.getElementById('sigs');
    box.innerHTML = '';
    arr.forEach(s => {
      const el = document.createElement('div');
      el.className = 'chip' + (s.tf === tf ? ' cur' : '');
      const name = LBL[s.tf] || s.tf;
      let h = '<b>' + name + '</b> ';
      if (s.trend) h += '<span style="color:' + (s.trend==='GREEN'?'#00c853':'#ff1744') + '">&#9679;</span> ';
      if (s.signal){
        const buy = s.signal.side === 'BUY';
        h += '<b style="color:' + (buy?'#00e676':'#ff5252') + '">' + s.signal.side + (s.signal.ok ? ' &#10003;' : ' (no EMA)') + '</b>'
          + '<br><span style="opacity:.75">' + s.signal.time + ' @ ' + s.signal.price.toLocaleString() + '</span>';
      } else h += '<span style="opacity:.6">' + (s.error ? 'no data' : 'no signal') + '</span>';
      el.innerHTML = h;
      el.onclick = () => { const b = document.getElementById('tf_' + s.tf); if (b) b.click(); };
      box.appendChild(el);
    });
  } catch(e){}
}

function push(series, arr, base){ for (const p of arr) if (p.time >= base) series.update(p); }

let busy = false, forceNext = false;
async function tick(){
  if (busy) return;
  busy = true;
  try { await tickInner(); } finally { busy = false; }
}
function hardRefresh(){
  forceNext = true; first = true; lastT = 0; clearStructure(); clearLiquidity();
  document.getElementById('upd').textContent = 'refreshing...';
  tick(); loadSigs(true); loadPlan(true);
}
document.addEventListener('visibilitychange', () => { if (!document.hidden){ tick(); loadSigs(); loadPlan(); } });

async function tickInner(){
  const want = tf;
  try{
    const r = await fetch('/data?tf=' + want + '&_=' + Date.now() + (forceNext ? '&force=1' : ''));
    const j = await r.json();
    if (want !== tf) return;
    forceNext = false;
    if (j.error){ document.getElementById('upd').textContent = 'error: ' + j.error; return; }
    if (first){
      chart.applyOptions({timeScale:{timeVisible: !LBL[want]}});
      candles.setData(j.candles); stUp.setData(j.st_up); stDn.setData(j.st_dn); ema.setData(j.ema);
      const n = j.candles.length;
      chart.timeScale().setVisibleLogicalRange({from: Math.max(0, n - 110), to: n + 6});
      first = false;
    } else {
      const base = lastT;
      push(candles, j.candles, base); push(stUp, j.st_up, base); push(stDn, j.st_dn, base); push(ema, j.ema, base);
    }
    lastT = j.candles[j.candles.length - 1].time;
    candles.setMarkers(j.markers);
    drawStructure(j.structure);
    drawLiquidity(j.liquidity);
    const up = j.last.chg >= 0;
    document.getElementById('price').textContent = j.last.price.toLocaleString(undefined,{minimumFractionDigits:2, maximumFractionDigits:2});
    const c = document.getElementById('chg');
    c.textContent = (up ? '+' : '') + j.last.chg.toFixed(2) + ' (' + (up ? '+' : '') + j.last.chg_pct.toFixed(2) + '%)';
    c.style.color = up ? '#00c853' : '#ff1744';
    document.getElementById('st').innerHTML = 'Supertrend: <b style="color:' + (j.last.st==='GREEN'?'#00c853':'#ff1744') + '">' + j.last.st + '</b>';
    document.getElementById('ema').textContent = 'EMA20: ' + j.last.ema.toFixed(2);
    document.getElementById('mkt').innerHTML = 'Market: <b style="color:' + (j.last.market==='OPEN'?'#00c853':'#ffb300') + '">' + j.last.market + '</b>';
    document.getElementById('sum').textContent = j.summary;
    const ls = j.last_signal, lse = document.getElementById('lastsig');
    if (ls) lse.innerHTML = 'Last signal: <b style="color:' + (ls.side==='BUY'?'#00e676':'#ff5252') + '">' + ls.side
      + (ls.ok ? '' : ' (no EMA)') + '</b> ' + ls.time + ' @ ' + ls.price.toLocaleString();
    else lse.textContent = 'Last signal: none';
    document.getElementById('upd').textContent = 'updated ' + j.last.time + (j.last.lag != null ? ' | price is ' + j.last.lag + ' s old at Yahoo' : ' (Yahoo data, may be delayed)');
  } catch(e){ document.getElementById('upd').textContent = 'connection problem...'; }
}
tick(); setInterval(tick, 1000);
loadSigs(); setInterval(loadSigs, 15000);
loadPlan(); setInterval(loadPlan, 20000);
</script></body></html>"""


def refresher():
    """Rebuilds the chart data of every time frame somebody is watching, so a request is answered instantly."""
    while True:
        for tf in list(active):
            if time.time() - active[tf] < 90:
                try:
                    with tf_locks[tf]:
                        cache[tf] = {"json": build_json(tf), "t": time.time()}
                except Exception as e:
                    print("refresh note:", e)
        time.sleep(1.5 if market_status() == "OPEN" else 10)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif u.path == "/data":
            tf = parse_qs(u.query).get("tf", ["15m"])[0]
            if tf not in TFS:
                tf = "15m"
            force = "force" in parse_qs(u.query)
            active[tf] = time.time()
            c = cache.get(tf)
            if force or c is None:
                with tf_locks[tf]:
                    try:
                        c = cache[tf] = {"json": build_json(tf), "t": time.time()}
                    except Exception as e:
                        print("Data error:", e)
                        if c is None:
                            self._send(200, "application/json", json.dumps({"error": str(e)}).encode("utf-8"))
                            return
            body = c["json"].encode("utf-8")
            self._send(200, "application/json", body)
        elif u.path == "/manifest.json":
            m = {"name": "NIFTY 50 Live", "short_name": "NIFTY", "start_url": "/", "display": "standalone",
                 "background_color": "#0f1218", "theme_color": "#0f1218",
                 "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
                           {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"}]}
            self._send(200, "application/manifest+json", json.dumps(m).encode("utf-8"))
        elif u.path in ("/icon-192.png", "/icon-512.png"):
            f = os.path.join(os.path.dirname(os.path.abspath(__file__)), u.path.lstrip("/"))
            try:
                with open(f, "rb") as fh:
                    self._send(200, "image/png", fh.read())
            except OSError:
                self.send_error(404)
        elif u.path == "/healthz":
            self._send(200, "text/plain", b"ok")
        elif u.path == "/plan":
            if "force" in parse_qs(u.query):
                plan_cache.clear()
            try:
                body = json.dumps(make_plan()).encode("utf-8")
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode("utf-8")
            self._send(200, "application/json", body)
        elif u.path == "/signals":
            if "force" in parse_qs(u.query):
                raw.clear()
            with sig_lock:
                body = json.dumps(all_signals()).encode("utf-8")
            self._send(200, "application/json", body)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "PC-IP"


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"NIFTY 50 LIVE chart:  http://localhost:{PORT}")
    if HOST == "0.0.0.0" and not CLOUD:
        print(f"Phone (same WiFi): http://{lan_ip()}:{PORT}")
    print("Stop: Ctrl + C")
    threading.Thread(target=refresher, daemon=True).start()
    if NTFY_TOPIC:
        threading.Thread(target=alert_loop, daemon=True).start()
        print("Phone alerts ON (ntfy topic set)")
    if not CLOUD:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
