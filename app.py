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
import re
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, quote as urlquote, urlparse

import numpy as np
import pandas as pd
import requests

DEFAULT_SYM = "^NSEI"                    # ^NSEI = Nifty 50 on Yahoo Finance (any Yahoo symbol works: RELIANCE.NS, ^NSEBANK ...)
SYM_RE = re.compile(r"^[\^A-Z0-9&\-._]{1,24}$")
known_syms = set()                       # symbols seen so far (capped, so nobody can hammer Yahoo through this server)
glock = threading.Lock()


def clean_sym(x):
    x = (x or "").strip().upper()
    if not SYM_RE.match(x):
        return DEFAULT_SYM
    with glock:
        if x not in known_syms and len(known_syms) >= 60:
            return DEFAULT_SYM
        known_syms.add(x)
    return x
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
tf_locks = {}
active = {}                               # "sym|tf" -> last time a browser asked for it


def lock_for(k):
    with glock:
        return tf_locks.setdefault(k, threading.Lock())


# ---------------- data + indicators ----------------
def fetch(tf, sym=DEFAULT_SYM):
    interval, rng, _ = TFS[tf]
    last_err = None
    for host in ("query1", "query2"):
        try:
            url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{urlquote(sym, safe='')}"
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
    """LIVE liquidity. Levels = swing highs/lows (stop-losses sit there), found on CLOSED candles only.
    EQH/EQL = two swings at (almost) the same level. Sweep = a candle pokes beyond a level but closes back
    inside (stop-hunt); the running candle is checked too (live=True) so you see it as it happens."""
    high, low, close = df.high.values, df.low.values, df.close.values
    n = len(df)
    levels, sweeps = [], []
    for i in range(N, n):
        live = i >= n_closed                           # the candle that is still forming
        if not live:
            j = i - N
            if j >= N:
                tol = 0.25 * float(atr[i])
                for side, arr, ok in (("H", high, high[j] >= high[j - N:j + N + 1].max()),
                                      ("L", low, low[j] <= low[j - N:j + N + 1].min())):
                    if not ok:
                        continue
                    lv = dict(side=side, j=j, price=float(arr[j]), eq=False, done=False)
                    for o in levels:                   # equal high / equal low with an earlier untouched swing
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
                    sweeps.append(dict(up=False, eq=lv["eq"], live=live, t0=t[lv["j"]], t1=t[i],
                                       price=round(lv["price"], 2)))
            elif lv["side"] == "L" and low[i] < lv["price"]:
                lv["done"] = True
                if close[i] > lv["price"]:
                    sweeps.append(dict(up=True, eq=lv["eq"], live=live, t0=t[lv["j"]], t1=t[i],
                                       price=round(lv["price"], 2)))
    # levels that are still waiting to be hit: nearest 2 above, nearest 2 below, and every EQH/EQL close by
    price, lim = float(close[-1]), 12 * float(atr[-1])
    open_lv = [lv for lv in levels if not lv["done"] and abs(lv["price"] - price) <= lim]
    above = sorted([lv for lv in open_lv if lv["side"] == "H" and lv["price"] > price], key=lambda x: x["price"])[:2]
    below = sorted([lv for lv in open_lv if lv["side"] == "L" and lv["price"] < price], key=lambda x: -x["price"])[:2]
    chosen, seen = [], set()
    for lv in above + below + [x for x in open_lv if x["eq"]]:
        k = (lv["side"], round(lv["price"], 1))
        if k not in seen:
            seen.add(k)
            chosen.append(dict(side=lv["side"], eq=lv["eq"], price=round(lv["price"], 2), t0=t[lv["j"]]))
    return dict(sweeps=sweeps[-12:], levels=chosen[:8], end=t[n - 1])


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


def tf_signal(tf, sym=DEFAULT_SYM):
    """Latest signal of one time frame (data cached 30 s)."""
    c = raw.get((sym, tf))
    if c is None or time.time() - c["t"] > 30:
        interval, rng, sec = TFS[tf]
        df = fetch(tf, sym)
        df["ema"] = df.close.ewm(span=EMA_LEN, adjust=False).mean()
        d, _ = supertrend(df)
        now = int(time.time())
        n_closed = closed_count(df, sec, now)
        flips = find_flips(df, d, n_closed, sec)
        sig = dict(tf=tf, trend="GREEN" if d[-1] == 1 else "RED",
                   signal=(dict(side="BUY" if flips[-1]["buy"] else "SELL", ok=flips[-1]["ok"],
                                price=flips[-1]["price"], time=flips[-1]["time"]) if flips else None))
        c = raw[(sym, tf)] = {"sig": sig, "t": time.time()}
    return c["sig"]


def all_signals(sym=DEFAULT_SYM):
    out = []
    for tf in TFS:
        try:
            out.append(tf_signal(tf, sym))
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


def build_json(tf, sym=DEFAULT_SYM):
    interval, rng, sec = TFS[tf]
    intraday = sec < 86400
    df = fetch(tf, sym)
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
                                color="#00c853" if f["ok"] else "#6b8f71", text="BUY" if f["ok"] else "buy"))
        else:
            markers.append(dict(time=ti, position="aboveBar", shape="arrowDown",
                                color="#ff1744" if f["ok"] else "#9a6b6b", text="SELL" if f["ok"] else "sell"))
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
        tf=tf, sym=sym, candles=candles, ema=ema, st_up=st_up, st_dn=st_dn, markers=markers,
        last=dict(price=price, chg=price - prev_close, chg_pct=(price - prev_close) / prev_close * 100,
                  st="GREEN" if d[-1] == 1 else "RED", ema=float(df.ema.iloc[-1]),
                  market=market_status(), lag=lag,
                  time=(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=SHIFT)).strftime("%d-%b %H:%M:%S IST")),
        summary=summary, last_signal=last_sig, structure=struct, liquidity=liq))


# ---------------- web page ----------------
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Nifty Live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#ffffff">
<meta name="apple-mobile-web-app-capable" content="yes">
<link rel="apple-touch-icon" href="/icon-192.png">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#fff;--bg2:#f6f7f9;--tx:#444;--st:#212121;--mu:#8b8f97;--bd:#eceef1;--bl:#387ed1;--up:#2e9e4f;--dn:#df514c;--or:#ff5722;--chip:#f1f3f6}
[data-theme=dark]{--bg:#0f1218;--bg2:#151a22;--tx:#d1d4dc;--st:#fff;--mu:#7d8590;--bd:#2b3139;--bl:#5b9bf0;--up:#26c281;--dn:#ff5252;--chip:#1c212b}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--tx);font-family:Segoe UI,Roboto,Arial,sans-serif;-webkit-tap-highlight-color:transparent}
#app{display:flex;flex-direction:column;height:100%}
#hd{display:flex;align-items:center;gap:12px;padding:8px 12px;border-bottom:1px solid var(--bd)}
#brand{font-weight:700;color:var(--or);white-space:nowrap;font-size:15px}
#strip{display:flex;gap:16px;overflow-x:auto;flex:1;scrollbar-width:none}
#strip::-webkit-scrollbar{display:none}
.tk{white-space:nowrap;cursor:pointer;font-size:11px;line-height:1.25;color:var(--mu)}
.tk b{display:block;font-size:13px;color:var(--st);font-weight:600}
.btn{background:var(--chip);color:var(--tx);border:1px solid var(--bd);border-radius:6px;padding:5px 10px;font-size:12px;cursor:pointer;white-space:nowrap}
.btn.on{background:var(--bl);color:#fff;border-color:var(--bl)}
#main{flex:1;min-height:0;display:flex}
.view{display:none;flex-direction:column;min-width:0;min-height:0;flex:1}
.view.on{display:flex}
#nav{display:flex;border-top:1px solid var(--bd)}
#nav button{flex:1;background:none;border:0;padding:8px 0 10px;color:var(--mu);font-size:11px;cursor:pointer}
#nav button b{display:block;font-size:16px;font-weight:400}
#nav button.on{color:var(--bl);font-weight:600}
@media(min-width:900px){#v_watch{display:flex!important;flex:none;width:340px;border-right:1px solid var(--bd)}#n_watch{display:none}}
/* watchlist */
#wtools{display:flex;gap:6px;padding:8px 12px;border-bottom:1px solid var(--bd)}
#wtools input{flex:1;min-width:0;background:var(--chip);color:var(--tx);border:1px solid var(--bd);border-radius:6px;padding:7px 9px;font-size:13px}
#wl{overflow-y:auto;flex:1}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:11px 12px;border-bottom:1px solid var(--bd);cursor:pointer}
.row.sel{background:var(--bg2)}
.row .l b{display:block;color:var(--st);font-weight:600;font-size:14px}
.row .l span{font-size:11px;color:var(--mu)}
.row .r{text-align:right;font-size:14px}
.row .r small{display:block;font-size:12px}
.x{color:var(--mu);padding:0 4px;font-size:16px}
.up{color:var(--up)}.dn{color:var(--dn)}
/* chart view */
#ch{padding:8px 12px 2px;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
#nm{font-weight:700;color:var(--st);font-size:16px}
#price{font-size:20px;color:var(--st)}
#chg{font-size:13px}
#mkt{font-size:12px;color:var(--mu);margin-left:auto}
.scrollx{display:flex;gap:5px;overflow-x:auto;padding:4px 12px;scrollbar-width:none}
.scrollx::-webkit-scrollbar{display:none}
#stat{padding:0 12px 4px;font-size:12px;color:var(--mu);display:flex;gap:12px;flex-wrap:wrap}
#chart{flex:1;min-height:180px}
#foot{padding:4px 12px 6px;font-size:11px;color:var(--mu)}
/* signals / plan */
.pad{padding:12px;overflow-y:auto;flex:1}
#sigs{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.chip{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;padding:8px 10px;font-size:12px;cursor:pointer;line-height:1.4}
.chip.cur{border-color:var(--or)}
.chip b{font-size:13px}
.how{margin-top:14px;font-size:12px;line-height:1.55;color:var(--mu)}
.how p{margin:0 0 8px}
#plan{background:var(--bg2);border:1px solid var(--bd);border-radius:10px;padding:12px;font-size:14px;line-height:1.6}
h3{margin:0 0 10px;font-size:15px;color:var(--st)}
</style></head><body>
<div id="app">
<div id="hd"><span id="brand">&#9650; Nifty Live</span><div id="strip"></div><button class="btn" id="b_theme" onclick="toggleTheme()">Dark</button></div>
<div id="main">
  <section class="view" id="v_watch">
    <div id="wtools"><input id="addsym" placeholder="Add stock e.g. TATAMOTORS" autocapitalize="characters" onkeydown="if(event.key==='Enter')addSym()"><button class="btn" onclick="addSym()">Add</button><button class="btn" id="b_edit" onclick="toggleEdit()">Edit</button></div>
    <div id="wl"></div>
  </section>
  <section class="view" id="v_chart">
    <div id="ch"><span id="nm">NIFTY 50</span><b id="price">loading...</b><span id="chg"></span><span id="mkt"></span></div>
    <div class="scrollx" id="tfs"></div>
    <div class="scrollx"><button class="btn" id="b_ref" onclick="hardRefresh()">&#8635; Refresh</button><button class="btn" id="b_str" onclick="toggleStruct()">Structure</button><button class="btn" id="b_liq" onclick="toggleLiq()">Liquidity</button></div>
    <div id="stat"><span id="st"></span><span id="ema"></span><span id="lastsig"></span></div>
    <div id="chart"></div>
    <div id="foot"><span id="sum"></span> <span id="upd"></span></div>
  </section>
  <section class="view" id="v_sigs"><div class="pad">
    <h3>Signals on every time frame</h3>
    <div id="sigs"></div>
    <div class="how">
      <p><b>BUY / SELL (big)</b> = Supertrend flip + EMA20 confirmed (the bot rule). <b>buy / sell (dim)</b> = flip without EMA20 confirmation. Signals come only after the candle closes.</p>
      <p><b>Structure:</b> BOS = close breaks the last swing high/low (trend continues). CHoCH = break against the earlier trend (possible reversal). Dashed grey lines = latest unbroken swing high/low.</p>
      <p><b>Liquidity (live):</b> BSL = stop-losses above a swing high, SSL = below a swing low, EQH/EQL (orange) = equal highs/lows (strongest). Labels are on the price axis. Sweep (cyan) = price poked beyond a level but closed back inside (stop-hunt); yellow LIVE Sweep = happening on the running candle right now.</p>
      <p>Data: free Yahoo Finance, can be delayed. This app shows charts only, it never places orders.</p>
    </div>
  </div></section>
  <section class="view" id="v_plan"><div class="pad"><div id="plan"></div></div></section>
</div>
<div id="nav">
  <button id="n_watch" onclick="setView('watch')"><b>&#9733;</b>Watchlist</button>
  <button id="n_chart" onclick="setView('chart')"><b>&#9698;</b>Chart</button>
  <button id="n_sigs" onclick="setView('sigs')"><b>&#9673;</b>Signals</button>
  <button id="n_plan" onclick="setView('plan')"><b>&#9636;</b>Trade plan</button>
</div>
</div>
<script>
const TFS = ['1m','5m','15m','30m','1h','1d','1w','1mo'];
const LBL = {'1d':'1D', '1w':'1W', '1mo':'1M'};
const NAMES = {'^NSEI':'NIFTY 50', '^NSEBANK':'BANK NIFTY', '^BSESN':'SENSEX', '^INDIAVIX':'INDIA VIX'};
const DEFWL = ['^NSEI','^NSEBANK','^BSESN','^INDIAVIX','RELIANCE.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','TCS.NS','SBIN.NS','ITC.NS','LT.NS'];
const STRIP = ['^NSEI','^NSEBANK','^BSESN'];
const NIFTY = '^NSEI';
const TH = {light:{bg:'#ffffff', tx:'#444444', grid:'#f1f2f4', bd:'#e3e5e8'}, dark:{bg:'#0f1218', tx:'#d1d4dc', grid:'#1c212b', bd:'#2b3139'}};
function nm(s){ return NAMES[s] || s.replace('.NS', ''); }
function ls(k, d){ try { const v = localStorage.getItem(k); return v === null ? d : v; } catch(e){ return d; } }
function ss(k, v){ try { localStorage.setItem(k, v); } catch(e){} }
function $(id){ return document.getElementById(id); }
function fmt(x){ return x.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2}); }
const wide = () => window.matchMedia('(min-width: 900px)').matches;

let tf = ls('ntf', '15m'); if (!TFS.includes(tf)) tf = '15m';
let sym = ls('nsym', NIFTY);
let wl = null; try { wl = JSON.parse(ls('wl', 'null')); } catch(e){}
if (!Array.isArray(wl) || !wl.length) wl = DEFWL.slice();
let dark = ls('dark', '0') === '1';
let view = ls('view', 'chart');
let editMode = false, quotes = {};

const chartBox = $('chart');
const T0 = dark ? TH.dark : TH.light;
const chart = LightweightCharts.createChart(chartBox, {
  autoSize:true,
  layout:{background:{color:T0.bg}, textColor:T0.tx},
  grid:{vertLines:{color:T0.grid}, horzLines:{color:T0.grid}},
  timeScale:{timeVisible:true, secondsVisible:false, borderColor:T0.bd}, rightPriceScale:{borderColor:T0.bd}});
const candles = chart.addCandlestickSeries({upColor:'#26a69a', downColor:'#ef5350', borderVisible:false,
  wickUpColor:'#26a69a', wickDownColor:'#ef5350'});
const opt = (c, w) => ({color:c, lineWidth:w, priceLineVisible:false, lastValueVisible:false});
const stUp = chart.addLineSeries(opt('#00c853', 2));
const stDn = chart.addLineSeries(opt('#ff1744', 2));
const ema  = chart.addLineSeries(opt('#2196f3', 1));
let first = true, lastT = 0, lastN = 0, rangeDirty = false;

function applyTheme(){
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  const t = dark ? TH.dark : TH.light;
  chart.applyOptions({layout:{background:{color:t.bg}, textColor:t.tx}, grid:{vertLines:{color:t.grid}, horzLines:{color:t.grid}},
    rightPriceScale:{borderColor:t.bd}, timeScale:{borderColor:t.bd}});
  $('b_theme').textContent = dark ? 'Light' : 'Dark';
  const m = document.querySelector('meta[name=theme-color]'); if (m) m.setAttribute('content', t.bg);
}
function toggleTheme(){ dark = !dark; ss('dark', dark ? '1' : '0'); applyTheme(); }

/* ---------- views ---------- */
function fixRange(){
  if (rangeDirty && chartBox.clientWidth > 0){
    chart.timeScale().setVisibleLogicalRange({from: Math.max(0, lastN - 110), to: lastN + 6});
    rangeDirty = false;
  }
}
function setView(v){
  if (wide() && v === 'watch') v = 'chart';
  view = v; ss('view', v);
  ['watch','chart','sigs','plan'].forEach(x => {
    $('v_' + x).classList.toggle('on', x === v);
    $('n_' + x).classList.toggle('on', x === v);
  });
  if (v === 'chart') setTimeout(fixRange, 80);
}

/* ---------- time frames ---------- */
const tbox = $('tfs');
TFS.forEach(x => {
  const b = document.createElement('button');
  b.className = 'btn'; b.textContent = LBL[x] || x; b.id = 'tf_' + x;
  b.onclick = () => {
    tf = x; ss('ntf', x);
    first = true; lastT = 0; clearStructure(); clearLiquidity(); mark(); tick(); loadSigs();
  };
  tbox.appendChild(b);
});
function mark(){ TFS.forEach(x => $('tf_' + x).className = 'btn' + (x === tf ? ' on' : '')); }
mark();

/* ---------- market structure ---------- */
let structSeries = [], structLines = [], structKey = '', lastStruct = null, showStruct = ls('st', '1') !== '0';
function clearStructure(){
  structSeries.forEach(x => { try { chart.removeSeries(x); } catch(e){} });
  structLines.forEach(x => { try { candles.removePriceLine(x); } catch(e){} });
  structSeries = []; structLines = []; structKey = '';
}
function drawStructure(st){
  lastStruct = st;
  $('b_str').className = 'btn' + (showStruct ? ' on' : '');
  const key = showStruct ? (sym + tf + JSON.stringify(st)) : 'off';
  if (key === structKey) return;
  clearStructure(); structKey = key;
  if (!showStruct || !st) return;
  st.events.forEach(e => {
    const col = e.kind === 'CHoCH' ? '#ab47bc' : '#f9a825';
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
function toggleStruct(){ showStruct = !showStruct; ss('st', showStruct ? '1' : '0'); structKey = ''; drawStructure(lastStruct); }

/* ---------- liquidity ---------- */
let liqSeries = [], liqLines = [], liqKey = '', lastLiq = null, showLiq = ls('lq', '1') !== '0';
function clearLiquidity(){
  liqSeries.forEach(x => { try { chart.removeSeries(x); } catch(e){} });
  liqLines.forEach(x => { try { candles.removePriceLine(x); } catch(e){} });
  liqSeries = []; liqLines = []; liqKey = '';
}
function drawLiquidity(lq){
  lastLiq = lq;
  $('b_liq').className = 'btn' + (showLiq ? ' on' : '');
  const key = showLiq ? (sym + tf + JSON.stringify(lq)) : 'off';
  if (key === liqKey) return;
  clearLiquidity(); liqKey = key;
  if (!showLiq || !lq) return;
  lq.levels.forEach(l => {
    const name = l.eq ? (l.side === 'H' ? 'EQH' : 'EQL') : (l.side === 'H' ? 'BSL' : 'SSL');
    liqLines.push(candles.createPriceLine({price: l.price, color: l.eq ? '#ffa726' : '#90a4ae', lineWidth: 1,
      lineStyle: 3, axisLabelVisible: true, title: name}));
  });
  lq.sweeps.forEach(e => {
    if (!(e.t1 > e.t0)) return;
    const col = e.live ? '#fbc02d' : '#26c6da';
    const s = chart.addLineSeries({color: col, lineWidth: 1, lineStyle: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false});
    s.setData([{time: e.t0, value: e.price}, {time: e.t1, value: e.price}]);
    s.setMarkers([{time: e.t1, position: e.up ? 'belowBar' : 'aboveBar', color: col, shape: 'square',
      text: (e.live ? 'LIVE ' : '') + (e.eq ? 'EQ ' : '') + 'Sweep ' + (e.up ? 'low' : 'high')}]);
    liqSeries.push(s);
  });
}
function toggleLiq(){ showLiq = !showLiq; ss('lq', showLiq ? '1' : '0'); liqKey = ''; drawLiquidity(lastLiq); }

/* ---------- trade plan (Nifty only) ---------- */
async function loadPlan(f){
  const el = $('plan');
  if (sym !== NIFTY){
    el.innerHTML = '<h3>Trade plan</h3>The option trade plan is made only for NIFTY 50 (1h signals, hedged option selling). Pick NIFTY 50 in the watchlist to see it.';
    return;
  }
  try{
    const r = await fetch('/plan?_=' + Date.now() + (f ? '&force=1' : ''));
    const p = await r.json();
    if (p.error){ el.innerHTML = '<h3>Trade plan</h3>Could not load the plan (' + p.error + ').'; return; }
    const s = p.signal, pl = p.plan;
    let h = '<h3>1h trade plan</h3><span style="color:var(--mu)">You place the orders yourself in your broker app | Nifty ' + fmt(p.spot) + ', lot ' + p.lot + '</span><br>';
    if (!s){ el.innerHTML = h + 'No 1h signal in this data yet.'; return; }
    const buy = s.side === 'BUY';
    h += '1h signal: <b class="' + (buy ? 'up' : 'dn') + '">' + s.side + (s.ok ? ' &#10003;' : ' (no EMA)') + '</b> '
      + s.time + ' (' + s.age_min + ' min ago)';
    if (!s.ok){ el.innerHTML = h + '<br>No EMA20 confirmation, so NO trade by the bot rule.'; return; }
    if (!s.fresh) h += '<br><span style="color:#f9a825">Old signal - do not enter now; wait for the next 1h signal.</span>';
    else if (s.late) h += '<br><span style="color:#f9a825">Signal came at 15:15 or later - better to skip a new entry today.</span>';
    if (pl){
      h += '<br>SELL <b>' + pl.sell_strike + ' ' + pl.kind + '</b> (~Rs ' + pl.sell_prem + ') + BUY hedge <b>' + pl.hedge_strike + ' ' + pl.kind + '</b> (~Rs ' + pl.hedge_prem + ')'
        + '<br>Expiry ' + pl.expiry + ' | credit ~Rs ' + pl.credit.toLocaleString() + ' | max loss ~Rs ' + pl.max_loss.toLocaleString()
        + '<br>Exit: sell premium reaches Rs ' + pl.sl_prem + ', or 1h Supertrend flips against you.'
        + '<br><span style="color:var(--mu);font-size:12px">Premiums are model estimates (IV ' + pl.iv + '%). Check real prices and margin in your broker app before ordering. Not financial advice.</span>';
    }
    el.innerHTML = h;
  } catch(e){}
}

/* ---------- signals of every time frame ---------- */
async function loadSigs(f){
  const ws = sym;
  try{
    const r = await fetch('/signals?sym=' + encodeURIComponent(ws) + '&_=' + Date.now() + (f ? '&force=1' : ''));
    const arr = await r.json();
    if (ws !== sym) return;
    const box = $('sigs');
    box.innerHTML = '';
    arr.forEach(s => {
      const el = document.createElement('div');
      el.className = 'chip' + (s.tf === tf ? ' cur' : '');
      let h = '<b>' + (LBL[s.tf] || s.tf) + '</b> ';
      if (s.trend) h += '<span class="' + (s.trend === 'GREEN' ? 'up' : 'dn') + '">&#9679;</span> ';
      if (s.signal){
        const buy = s.signal.side === 'BUY';
        h += '<b class="' + (buy ? 'up' : 'dn') + '">' + s.signal.side + (s.signal.ok ? ' &#10003;' : ' (no EMA)') + '</b>'
          + '<br><span style="color:var(--mu)">' + s.signal.time + ' @ ' + s.signal.price.toLocaleString() + '</span>';
      } else h += '<span style="color:var(--mu)">' + (s.error ? 'no data' : 'no signal') + '</span>';
      el.innerHTML = h;
      el.onclick = () => { $('tf_' + s.tf).click(); setView('chart'); };
      box.appendChild(el);
    });
  } catch(e){}
}

/* ---------- watchlist ---------- */
function pctCls(q){ return q && q.chg >= 0 ? 'up' : 'dn'; }
function renderWatch(){
  const box = $('wl');
  box.innerHTML = '';
  wl.forEach(s => {
    const q = quotes[s];
    const row = document.createElement('div');
    row.className = 'row' + (s === sym ? ' sel' : '');
    const l = document.createElement('div'); l.className = 'l';
    const b = document.createElement('b'); b.textContent = nm(s);
    const sp = document.createElement('span'); sp.textContent = s.charAt(0) === '^' ? 'INDEX' : 'NSE';
    l.appendChild(b); l.appendChild(sp);
    const r = document.createElement('div'); r.className = 'r';
    if (q){
      r.innerHTML = '<span class="' + pctCls(q) + '">' + fmt(q.price) + '</span><small class="' + pctCls(q) + '">'
        + (q.chg >= 0 ? '+' : '') + q.chg.toFixed(2) + ' (' + (q.chg >= 0 ? '+' : '') + q.pct.toFixed(2) + '%)</small>';
    } else r.innerHTML = '<span style="color:var(--mu)">...</span>';
    row.appendChild(l); row.appendChild(r);
    if (editMode){
      const x = document.createElement('span'); x.className = 'x'; x.innerHTML = '&#10005;';
      x.onclick = ev => { ev.stopPropagation(); wl = wl.filter(y => y !== s); if (!wl.length) wl = [NIFTY]; ss('wl', JSON.stringify(wl)); renderWatch(); };
      row.appendChild(x);
    }
    row.onclick = () => { if (!editMode) selectSym(s); };
    box.appendChild(row);
  });
}
function renderStrip(){
  const box = $('strip');
  box.innerHTML = '';
  STRIP.forEach(s => {
    const q = quotes[s];
    const d = document.createElement('div'); d.className = 'tk';
    d.innerHTML = nm(s) + '<b></b>';
    d.querySelector('b').innerHTML = q ? ('<span class="' + pctCls(q) + '">' + fmt(q.price) + ' <small>' + (q.chg >= 0 ? '+' : '') + q.pct.toFixed(2) + '%</small></span>') : '...';
    d.onclick = () => selectSym(s);
    box.appendChild(d);
  });
}
async function loadQuotes(){
  try{
    const list = Array.from(new Set(wl.concat(STRIP, [sym])));
    const r = await fetch('/quotes?syms=' + encodeURIComponent(list.join(',')) + '&_=' + Date.now());
    const arr = await r.json();
    arr.forEach(q => { if (!q.error) quotes[q.sym] = q; });
    renderWatch(); renderStrip();
  } catch(e){}
}
function toggleEdit(){ editMode = !editMode; $('b_edit').className = 'btn' + (editMode ? ' on' : ''); $('b_edit').textContent = editMode ? 'Done' : 'Edit'; renderWatch(); }
async function addSym(){
  const el = $('addsym');
  let s = el.value.trim().toUpperCase().replace(/ /g, '');
  if (!s) return;
  if (s.charAt(0) !== '^' && s.indexOf('.') < 0) s += '.NS';
  if (wl.indexOf(s) >= 0){ el.value = ''; return; }
  try{
    const r = await fetch('/quotes?syms=' + encodeURIComponent(s));
    const a = await r.json();
    if (!a.length || a[0].error || a[0].sym !== s){ alert('Symbol not found on Yahoo Finance: ' + s); return; }
  } catch(e){ alert('Could not check the symbol, try again.'); return; }
  wl.push(s); ss('wl', JSON.stringify(wl)); el.value = ''; loadQuotes();
}
function selectSym(s){
  sym = s; ss('nsym', s);
  first = true; lastT = 0; clearStructure(); clearLiquidity();
  $('nm').textContent = nm(s); $('price').textContent = '...'; $('chg').textContent = '';
  renderWatch(); setView('chart');
  tick(); loadSigs(); loadPlan();
}

/* ---------- live chart data ---------- */
function push(series, arr, base){ for (const p of arr) if (p.time >= base) series.update(p); }

let busy = false, forceNext = false;
async function tick(){
  if (busy) return;
  busy = true;
  try { await tickInner(); } finally { busy = false; }
}
function hardRefresh(){
  forceNext = true; first = true; lastT = 0; clearStructure(); clearLiquidity();
  $('upd').textContent = 'refreshing...';
  tick(); loadSigs(true); loadPlan(true); loadQuotes();
}
document.addEventListener('visibilitychange', () => { if (!document.hidden){ tick(); loadSigs(); loadPlan(); loadQuotes(); } });

async function tickInner(){
  const want = tf, ws = sym;
  try{
    const r = await fetch('/data?tf=' + want + '&sym=' + encodeURIComponent(ws) + '&_=' + Date.now() + (forceNext ? '&force=1' : ''));
    const j = await r.json();
    if (want !== tf || ws !== sym) return;
    forceNext = false;
    if (j.error){ $('upd').textContent = 'error: ' + j.error; return; }
    if (first){
      chart.applyOptions({timeScale:{timeVisible: !LBL[want]}});
      candles.setData(j.candles); stUp.setData(j.st_up); stDn.setData(j.st_dn); ema.setData(j.ema);
      lastN = j.candles.length;
      if (chartBox.clientWidth > 0) chart.timeScale().setVisibleLogicalRange({from: Math.max(0, lastN - 110), to: lastN + 6});
      else rangeDirty = true;
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
    $('nm').textContent = nm(ws);
    $('price').textContent = fmt(j.last.price);
    const c = $('chg');
    c.textContent = (up ? '+' : '') + j.last.chg.toFixed(2) + ' (' + (up ? '+' : '') + j.last.chg_pct.toFixed(2) + '%)';
    c.className = up ? 'up' : 'dn';
    $('st').innerHTML = 'Supertrend: <b class="' + (j.last.st === 'GREEN' ? 'up' : 'dn') + '">' + j.last.st + '</b>';
    $('ema').textContent = 'EMA20: ' + j.last.ema.toFixed(2);
    $('mkt').innerHTML = 'Market <b class="' + (j.last.market === 'OPEN' ? 'up' : '') + '">' + j.last.market + '</b>';
    $('sum').textContent = j.summary;
    const ls_ = j.last_signal, lse = $('lastsig');
    if (ls_) lse.innerHTML = 'Last signal: <b class="' + (ls_.side === 'BUY' ? 'up' : 'dn') + '">' + ls_.side
      + (ls_.ok ? '' : ' (no EMA)') + '</b> ' + ls_.time + ' @ ' + ls_.price.toLocaleString();
    else lse.textContent = 'Last signal: none';
    $('upd').textContent = '| updated ' + j.last.time + (j.last.lag != null ? ' | price is ' + j.last.lag + ' s old at Yahoo' : ' (Yahoo data, may be delayed)');
    document.title = nm(ws) + ' ' + fmt(j.last.price);
  } catch(e){ $('upd').textContent = '| connection problem...'; }
}

applyTheme();
if (wide() && view === 'watch') view = 'chart';
if (['watch','chart','sigs','plan'].indexOf(view) < 0) view = 'chart';
$('nm').textContent = nm(sym);
setView(view);
renderWatch(); renderStrip();
tick(); setInterval(tick, 1000);
loadSigs(); setInterval(loadSigs, 15000);
loadPlan(); setInterval(loadPlan, 20000);
loadQuotes(); setInterval(loadQuotes, 5000);
</script></body></html>"""


quote_cache = {}


def quote(sym):
    """Last price + change vs previous close (cached 8 s per symbol)."""
    c = quote_cache.get(sym)
    if c and time.time() - c["t"] < 8:
        return c["v"]
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urlquote(sym, safe='')}"
        r = http.get(url, params={"interval": "1d", "range": "5d", "includePrePost": "false"},
                     headers=HEADERS, timeout=8)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        meta = res["meta"]
        price = float(meta["regularMarketPrice"])
        rt = int(meta.get("regularMarketTime") or time.time())
        today = (rt + SHIFT) // 86400
        prev = None
        for a, b in zip(res.get("timestamp") or [], res["indicators"]["quote"][0]["close"]):
            if b is not None and (a + SHIFT) // 86400 < today:
                prev = float(b)
        if prev is None:
            prev = float(meta.get("chartPreviousClose") or price)
        v = dict(sym=sym, price=round(price, 2), chg=round(price - prev, 2),
                 pct=round((price - prev) / prev * 100, 2) if prev else 0.0)
    except Exception as e:
        v = dict(sym=sym, error=str(e)[:80])
    quote_cache[sym] = {"v": v, "t": time.time()}
    return v


def refresher():
    """Rebuilds the chart data of every time frame somebody is watching, so a request is answered instantly."""
    while True:
        for k in list(active):
            if time.time() - active[k] < 90:
                try:
                    sym, tf = k.rsplit("|", 1)
                    with lock_for(k):
                        cache[k] = {"json": build_json(tf, sym), "t": time.time()}
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
            q = parse_qs(u.query)
            tf = q.get("tf", ["15m"])[0]
            if tf not in TFS:
                tf = "15m"
            sym = clean_sym(q.get("sym", [DEFAULT_SYM])[0])
            k = f"{sym}|{tf}"
            force = "force" in q
            active[k] = time.time()
            c = cache.get(k)
            if force or c is None:
                with lock_for(k):
                    try:
                        c = cache[k] = {"json": build_json(tf, sym), "t": time.time()}
                    except Exception as e:
                        print("Data error:", e)
                        if c is None:
                            self._send(200, "application/json", json.dumps({"error": str(e)}).encode("utf-8"))
                            return
            body = c["json"].encode("utf-8")
            self._send(200, "application/json", body)
        elif u.path == "/manifest.json":
            m = {"name": "Nifty Live", "short_name": "Nifty Live", "start_url": "/", "display": "standalone",
                 "background_color": "#ffffff", "theme_color": "#ffffff",
                 "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
                           {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"}]}
            self._send(200, "application/manifest+json", json.dumps(m).encode("utf-8"))
        elif u.path in ("/icon-192.png", "/icon-512.png", "/favicon.ico"):
            name = "icon-192.png" if u.path == "/favicon.ico" else u.path.lstrip("/")
            f = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
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
            q = parse_qs(u.query)
            if "force" in q:
                raw.clear()
            sym = clean_sym(q.get("sym", [DEFAULT_SYM])[0])
            with sig_lock:
                body = json.dumps(all_signals(sym)).encode("utf-8")
            self._send(200, "application/json", body)
        elif u.path == "/quotes":
            names = [clean_sym(x) for x in parse_qs(u.query).get("syms", [""])[0].split(",") if x.strip()][:30]
            names = list(dict.fromkeys(names))
            with ThreadPoolExecutor(max_workers=8) as ex:
                out = list(ex.map(quote, names))
            self._send(200, "application/json", json.dumps(out).encode("utf-8"))
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
