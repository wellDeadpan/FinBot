from __future__ import annotations
import os
import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =========================
# Data models
# =========================

@dataclass
class Signal:
    name: str
    value: float
    unit: str = ""
    z: Optional[float] = None
    direction: Optional[str] = None  # "risk_on" / "risk_off" / "neutral"
    note: str = ""


@dataclass
class DailyContext:
    date: dt.date
    fx: Dict[str, pd.Series]        # symbol -> price series (e.g., last 60d)
    rates: Dict[str, pd.Series]
    equity: Dict[str, pd.Series]
    portfolio: pd.DataFrame         # positions table


# =========================
# Providers (pluggable)
# =========================

class MarketDataProvider:
    """Interface: return price history for each symbol."""
    def get_history(self, symbols: List[str], lookback_days: int = 90) -> Dict[str, pd.Series]:
        raise NotImplementedError


class CSVPortfolioProvider:
    def __init__(self, positions_path: str):
        self.positions_path = positions_path

    def load_positions(self) -> pd.DataFrame:
        df = pd.read_csv(self.positions_path)
        needed = {"symbol", "qty", "avg_cost"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"positions.csv missing columns: {missing}")
        return df


# =========================
# Utilities
# =========================

def pct_change(series: pd.Series, n: int = 1) -> float:
    s = series.dropna()
    if len(s) < n + 1:
        return float("nan")
    return (s.iloc[-1] / s.iloc[-(n+1)] - 1.0) * 100.0


def zscore_of_last(series: pd.Series, window: int = 60) -> float:
    s = series.dropna()
    if len(s) < window:
        window = len(s)
    w = s.iloc[-window:]
    mu = w.mean()
    sd = w.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        return float("nan")
    return (w.iloc[-1] - mu) / sd


# =========================
# Analyzers
# =========================

class FXAnalyzer:
    def analyze(self, fx_hist: Dict[str, pd.Series]) -> List[Signal]:
        out: List[Signal] = []
        # Example: DXY up + USDJPY up => risk-off-ish (simplified)
        dxy = fx_hist.get("DXY")
        uj = fx_hist.get("USDJPY")
        if dxy is not None:
            out.append(Signal("DXY 1D %", pct_change(dxy, 1), "%", z=zscore_of_last(dxy)))
        if uj is not None:
            out.append(Signal("USDJPY 1D %", pct_change(uj, 1), "%", z=zscore_of_last(uj)))

        # Simple composite risk score (you can refine later)
        score = 0.0
        if dxy is not None:
            score += 0.6 * (pct_change(dxy, 1))
        if uj is not None:
            score += 0.4 * (pct_change(uj, 1))

        direction = "neutral"
        if score > 0.15:
            direction = "risk_off"
        elif score < -0.15:
            direction = "risk_on"

        out.append(Signal("FX Risk Score", score, "pts", direction=direction,
                          note=">0 tends risk-off (USD strength), <0 tends risk-on (USD weakness)"))
        return out


class RatesAnalyzer:
    def analyze(self, rates_hist: Dict[str, pd.Series]) -> List[Signal]:
        out: List[Signal] = []
        us2 = rates_hist.get("US2Y")
        us10 = rates_hist.get("US10Y")
        if us2 is not None:
            out.append(Signal("US2Y 1D chg", us2.iloc[-1] - us2.iloc[-2], "bp", z=zscore_of_last(us2)))
        if us10 is not None:
            out.append(Signal("US10Y 1D chg", us10.iloc[-1] - us10.iloc[-2], "bp", z=zscore_of_last(us10)))

        if us2 is not None and us10 is not None and len(us2.dropna()) > 2 and len(us10.dropna()) > 2:
            curve = (us10 - us2)
            out.append(Signal("10Y-2Y", curve.iloc[-1], "bp", z=zscore_of_last(curve)))
            out.append(Signal("10Y-2Y 1D chg", curve.iloc[-1] - curve.iloc[-2], "bp"))
        return out


class EquityAnalyzer:
    def analyze(self, eq_hist: Dict[str, pd.Series]) -> List[Signal]:
        out: List[Signal] = []
        for k in ["SPY", "QQQ", "IWM", "VIX"]:
            s = eq_hist.get(k)
            if s is not None:
                out.append(Signal(f"{k} 1D %", pct_change(s, 1), "%", z=zscore_of_last(s)))
        return out


class PortfolioAnalyzer:
    def analyze(self, positions: pd.DataFrame, last_prices: Dict[str, float]) -> List[Signal]:
        # Minimal: top concentration + unrealized PnL
        df = positions.copy()
        df["last"] = df["symbol"].map(last_prices).astype(float)
        df["mkt_value"] = df["qty"] * df["last"]
        total = df.loc[df["symbol"] != "CASH", "mkt_value"].sum()
        if total == 0:
            return [Signal("Portfolio", 0, note="No non-cash positions.")]
        df["weight"] = df["mkt_value"] / total

        top = df[df["symbol"] != "CASH"].sort_values("weight", ascending=False).head(5)
        top_note = "; ".join([f"{r.symbol}:{r.weight:.1%}" for r in top.itertuples()])
        out = [Signal("Top weights", top["weight"].max() * 100, "%", note=top_note)]
        return out


# =========================
# Reporter (LLM optional)
# =========================

class MarkdownReporter:
    def build(self, date: dt.date, fx: List[Signal], rates: List[Signal], eq: List[Signal], pf: List[Signal]) -> str:
        def fmt(sig: Signal) -> str:
            z = "" if sig.z is None or math.isnan(sig.z) else f" (z={sig.z:.2f})"
            d = "" if not sig.direction else f" [{sig.direction}]"
            unit = f" {sig.unit}".rstrip()
            note = f" — {sig.note}" if sig.note else ""
            return f"- **{sig.name}**: {sig.value:.2f}{unit}{z}{d}{note}"

        lines = []
        lines.append(f"# Daily Review — {date.isoformat()}\n")
        lines.append("## FX (汇)\n" + "\n".join(fmt(s) for s in fx) + "\n")
        lines.append("## Rates (债)\n" + "\n".join(fmt(s) for s in rates) + "\n")
        lines.append("## Equity (股)\n" + "\n".join(fmt(s) for s in eq) + "\n")
        lines.append("## Portfolio (持仓)\n" + "\n".join(fmt(s) for s in pf) + "\n")
        lines.append("## Actions\n- (auto) If risk-off score↑ + rates↑ + equities↓, consider reduce beta / add hedge.\n")
        return "\n".join(lines)


# =========================
# Orchestrator
# =========================

def run_daily(config: dict, mkt: MarketDataProvider, pfprov: CSVPortfolioProvider) -> str:
    today = dt.date.today()

    universe = config["universe"]
    fx_syms = universe["fx"]
    rate_syms = universe["rates"]
    eq_syms = universe["equity_index"] + universe["equity_sectors"] + universe["vol"]

    fx_hist = mkt.get_history(fx_syms, lookback_days=120)
    rates_hist = mkt.get_history(rate_syms, lookback_days=120)
    eq_hist = mkt.get_history(eq_syms, lookback_days=120)

    positions = pfprov.load_positions()

    # last prices for portfolio symbols
    last_prices = {}
    for sym in positions["symbol"].unique():
        s = eq_hist.get(sym) or fx_hist.get(sym) or rates_hist.get(sym)
        if s is not None and len(s.dropna()) > 0:
            last_prices[sym] = float(s.dropna().iloc[-1])
        elif sym == config["portfolio"].get("cash_symbol", "CASH"):
            last_prices[sym] = 1.0

    fx_sig = FXAnalyzer().analyze(fx_hist)
    rates_sig = RatesAnalyzer().analyze(rates_hist)
    eq_sig = EquityAnalyzer().analyze(eq_hist)
    pf_sig = PortfolioAnalyzer().analyze(positions, last_prices)

    md = MarkdownReporter().build(today, fx_sig, rates_sig, eq_sig, pf_sig)

    outdir = config["report"]["output_dir"]
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"daily_report_{today.isoformat()}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path
