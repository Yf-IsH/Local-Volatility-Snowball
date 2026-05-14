import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

import numpy as np
import requests

from core.vol_surface import VolSurface
from data.market_data import CACHE_DIR, INDEX_UNIVERSE


ETF_OPTION_UNIVERSE: Dict[str, Dict[str, str]] = {
    key: meta for key, meta in INDEX_UNIVERSE.items() if "option_prefix" in meta
}


@dataclass
class OptionIVSurface:
    surface: VolSurface
    source: str
    asof: str
    raw_points: int
    filtered_points: int


def _previous_dates(days=12):
    today = date.today()
    return [today - timedelta(days=i) for i in range(days)]


def _cache_path(etf_key: str, trade_date: str) -> Path:
    return CACHE_DIR / f"{etf_key}_sse_option_iv_{trade_date}.csv"


def _request_sse_risk(trade_date: date):
    session = requests.Session()
    session.trust_env = False
    params = {
        "sqlId": "SSE_ZQPZ_YSP_GGQQZSXT_YSHQ_QQFXZB_DATE_L",
        "tradeDate": trade_date.strftime("%Y%m%d"),
        "pageHelp.pageSize": "5000",
        "pageHelp.pageNo": "1",
        "pageHelp.beginPage": "1",
        "pageHelp.endPage": "1",
    }
    response = session.get(
        "http://query.sse.com.cn/commonQuery.do",
        params=params,
        timeout=10,
        headers={"Referer": "http://www.sse.com.cn/", "User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    return response.json().get("result") or []


def _write_cache(etf_key: str, trade_date: str, rows):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(etf_key, trade_date)
    if not rows:
        return
    fields = sorted({field for row in rows for field in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_latest_cache(etf_key: str):
    files = sorted(CACHE_DIR.glob(f"{etf_key}_sse_option_iv_*.csv"), reverse=True)
    for path in files:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            trade_date = path.stem.rsplit("_", 1)[-1]
            return rows, trade_date
    raise FileNotFoundError(f"no cached option IV for {etf_key}")


def load_sse_option_rows(etf_key: str):
    if etf_key not in ETF_OPTION_UNIVERSE:
        raise ValueError(f"{etf_key} has no SSE ETF option IV source")

    prefix = ETF_OPTION_UNIVERSE[etf_key]["option_prefix"]
    for dt in _previous_dates():
        try:
            rows = _request_sse_risk(dt)
            rows = [row for row in rows if str(row.get("CONTRACT_ID", "")).startswith(prefix)]
            if rows:
                trade_date = rows[0].get("TRADE_DATE") or dt.isoformat()
                _write_cache(etf_key, trade_date.replace("-", ""), rows)
                return rows, trade_date, "上交所期权风险指标 IMPLC_VOLATLTY"
        except Exception:
            continue

    rows, trade_date = _read_latest_cache(etf_key)
    return rows, trade_date, "本地缓存：上交所期权风险指标"


def _last_wednesday(year: int, month: int):
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != 2:
        d -= timedelta(days=1)
    return d


def _parse_contract(contract_id: str):
    match = re.search(r"([CP])(\d{2})(\d{2})M(\d+)", contract_id)
    if not match:
        return None
    cp, yy, mm, strike_raw = match.groups()
    year = 2000 + int(yy)
    month = int(mm)
    strike = int(strike_raw) / 1000.0
    expiry = _last_wednesday(year, month)
    return cp, expiry, strike


def _clean_points(rows, spot: float, trade_date: str):
    base_date = datetime.strptime(trade_date[:10], "%Y-%m-%d").date() if "-" in trade_date else datetime.strptime(
        trade_date[:8], "%Y%m%d"
    ).date()
    grouped: Dict[Tuple[float, float], List[float]] = {}
    raw_count = 0

    for row in rows:
        parsed = _parse_contract(str(row.get("CONTRACT_ID", "")))
        if not parsed:
            continue
        _, expiry, strike = parsed
        tau = (expiry - base_date).days / 365.0
        if tau < 7 / 365.0 or tau > 2.25:
            continue
        try:
            iv = float(row.get("IMPLC_VOLATLTY", 0.0))
        except (TypeError, ValueError):
            continue
        raw_count += 1
        if not (0.03 <= iv <= 1.2):
            continue
        if not (0.65 <= strike / spot <= 1.45):
            continue
        grouped.setdefault((strike, tau), []).append(iv)

    points = [(K, T, median(vols)) for (K, T), vols in grouped.items()]
    return points, raw_count


def _grid_surface(points, spot: float):
    if len(points) < 12:
        raise ValueError("not enough option IV points after filtering")

    strikes_raw = np.array([p[0] for p in points], dtype=float)
    maturities_raw = np.array([p[1] for p in points], dtype=float)
    vols_raw = np.array([p[2] for p in points], dtype=float)

    strike_ratios = np.array([0.7, 0.8, 0.9, 0.97, 1.0, 1.03, 1.1, 1.2, 1.35])
    strikes = spot * strike_ratios
    maturities = np.array([1 / 12, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    maturities = maturities[(maturities >= max(1 / 365, maturities_raw.min())) & (maturities <= maturities_raw.max())]
    if len(maturities) < 4:
        unique_t = np.unique(np.round(maturities_raw, 6))
        maturities = unique_t[: max(4, min(len(unique_t), 7))]

    iv = np.zeros((len(strikes), len(maturities)))
    scale_k = max(spot * 0.08, 1e-6)
    scale_t = 0.35
    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            dist2 = ((strikes_raw - K) / scale_k) ** 2 + ((maturities_raw - T) / scale_t) ** 2
            weights = 1.0 / np.maximum(dist2, 1e-6)
            nearest = np.argsort(dist2)[: min(12, len(dist2))]
            iv[i, j] = np.sum(weights[nearest] * vols_raw[nearest]) / np.sum(weights[nearest])

    return VolSurface(strikes, maturities, np.clip(iv, 0.03, 1.2))


def load_sse_option_iv_surface(etf_key: str, spot: float) -> OptionIVSurface:
    rows, trade_date, source = load_sse_option_rows(etf_key)
    points, raw_count = _clean_points(rows, spot, trade_date)
    surface = _grid_surface(points, spot)
    return OptionIVSurface(
        surface=surface,
        source=source,
        asof=trade_date,
        raw_points=raw_count,
        filtered_points=len(points),
    )
