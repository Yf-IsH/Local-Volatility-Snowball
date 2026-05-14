import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


INDEX_UNIVERSE: Dict[str, Dict[str, str]] = {
    "etf50": {"name": "50ETF", "code": "510050", "secid": "1.510050", "option_prefix": "510050"},
    "etf300": {"name": "300ETF", "code": "510300", "secid": "1.510300", "option_prefix": "510300"},
    "etf500": {"name": "500ETF", "code": "510500", "secid": "1.510500", "option_prefix": "510500"},
    "kc50": {"name": "科创50ETF", "code": "588000", "secid": "1.588000", "option_prefix": "588000"},
    "csi300": {"name": "沪深300", "code": "000300", "secid": "1.000300"},
    "csi500": {"name": "中证500", "code": "000905", "secid": "1.000905"},
    "csi1000": {"name": "中证1000", "code": "000852", "secid": "1.000852"},
    "sse50": {"name": "上证50", "code": "000016", "secid": "1.000016"},
}


@dataclass
class MarketSnapshot:
    key: str
    name: str
    code: str
    spot: float
    close: List[float]
    dates: List[str]
    annual_vol: float
    source: str
    asof: str


def _cache_path(index_key: str) -> Path:
    return CACHE_DIR / f"{index_key}_daily.csv"


def _sample_base(index_key: str):
    return {
        "etf50": 2.7,
        "etf300": 3.9,
        "etf500": 6.0,
        "kc50": 0.9,
        "csi300": 3900.0,
        "csi500": 5600.0,
        "csi1000": 5700.0,
        "sse50": 2700.0,
    }.get(index_key, 100.0)


def _sample_vol(index_key: str):
    return {
        "etf50": 0.18,
        "etf300": 0.19,
        "etf500": 0.23,
        "kc50": 0.28,
        "csi300": 0.18,
        "csi500": 0.22,
        "csi1000": 0.25,
        "sse50": 0.17,
    }.get(index_key, 0.2)


def _parse_eastmoney_klines(payload, index_key: str):
    rows = payload.get("data", {}).get("klines") or []
    if not rows:
        raise ValueError("empty kline response")

    parsed = []
    for line in rows:
        fields = line.split(",")
        parsed.append(
            {
                "date": fields[0],
                "open": float(fields[1]),
                "close": float(fields[2]),
                "high": float(fields[3]),
                "low": float(fields[4]),
                "volume": float(fields[5]),
                "amount": float(fields[6]),
            }
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _cache_path(index_key).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(parsed[0].keys()))
        writer.writeheader()
        writer.writerows(parsed)
    return parsed


def _fetch_eastmoney(index_key: str, lookback_days: int):
    meta = INDEX_UNIVERSE[index_key]
    end = date.today()
    begin = end - timedelta(days=max(lookback_days * 2, 420))
    params = {
        "secid": meta["secid"],
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": begin.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
    }
    session = requests.Session()
    session.trust_env = False
    last_error = None
    for _ in range(3):
        try:
            response = session.get(
                "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                params=params,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            break
        except Exception as exc:
            last_error = exc
            time.sleep(0.3)
    else:
        raise last_error
    payload = response.json()
    if payload.get("rc") != 0:
        raise ValueError(json.dumps(payload, ensure_ascii=False)[:300])
    return _parse_eastmoney_klines(payload, index_key)


def _load_cache(index_key: str):
    path = _cache_path(index_key)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return [
            {
                "date": row["date"],
                "open": float(row["open"]),
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "volume": float(row["volume"]),
                "amount": float(row["amount"]),
            }
            for row in csv.DictReader(f)
        ]


def _sample_history(index_key: str, days: int):
    rng = np.random.default_rng(abs(hash(index_key)) % (2**32))
    close = [_sample_base(index_key)]
    for _ in range(days - 1):
        ret = rng.normal(-0.02 / 252.0, _sample_vol(index_key) / math.sqrt(252.0))
        close.append(close[-1] * math.exp(ret))
    end = date.today()
    dates = [(end - timedelta(days=days - i - 1)).isoformat() for i in range(days)]
    return [
        {
            "date": dt,
            "open": px,
            "close": px,
            "high": px,
            "low": px,
            "volume": 0.0,
            "amount": 0.0,
        }
        for dt, px in zip(dates, close)
    ]


def load_index_history(index_key: str, lookback_days: int = 520):
    if index_key not in INDEX_UNIVERSE:
        raise ValueError(f"unknown index key: {index_key}")
    try:
        rows = _fetch_eastmoney(index_key, lookback_days)
        source = "东方财富 push2his 实时接口"
    except Exception:
        try:
            rows = _load_cache(index_key)
            source = "本地缓存"
        except Exception:
            rows = _sample_history(index_key, lookback_days)
            source = "离线样例数据"
    return rows[-lookback_days:], source


def realized_volatility(close, window=252):
    close = np.asarray(close, dtype=float)
    if len(close) < 3:
        return 0.2
    returns = np.diff(np.log(close[-(window + 1) :]))
    vol = np.std(returns, ddof=1) * np.sqrt(252.0)
    return float(np.clip(vol, 0.05, 0.8))


def load_market_snapshot(index_key: str, lookback_days: int = 520) -> MarketSnapshot:
    rows, source = load_index_history(index_key, lookback_days)
    meta = INDEX_UNIVERSE[index_key]
    close = [row["close"] for row in rows]
    dates = [row["date"] for row in rows]
    return MarketSnapshot(
        key=index_key,
        name=meta["name"],
        code=meta["code"],
        spot=float(close[-1]),
        close=close,
        dates=dates,
        annual_vol=realized_volatility(close),
        source=source,
        asof=dates[-1] if dates else datetime.now().date().isoformat(),
    )
