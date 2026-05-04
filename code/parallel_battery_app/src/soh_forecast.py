"""soh_forecast.py
SOH (State of Health) estimation and time-series forecasting.

SOH definition:
    SOH_k = Q_k / Q_nominal * 100 %

Pipeline
--------
1. estimate_soh_per_cycle  – compute SOH per cycle from Coulomb-counted
   discharge capacity (or approximate cycles from current sign changes).
2. forecast_soh  – ensemble of linear / polynomial / exponential-decay
   models; returns point forecast + 95 % CI.
3. estimate_rul  – infer End-of-Life (SOH < threshold) and RUL in cycles.
4. plot_* helpers – Plotly figures ready for Streamlit.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# ---------------------------------------------------------------------------
# SOH estimation
# ---------------------------------------------------------------------------

def estimate_soh_per_cycle(
    timeseries_df: pd.DataFrame,
    cell_current_cols: List[str],
    time_col: Optional[str] = None,
    cycle_col: Optional[str] = None,
    nominal_capacity_ah: Optional[float] = None,
) -> pd.DataFrame:
    """
    Estimate per-cycle SOH for each cell.

    Returns a DataFrame with columns:
        cycle, soh_<col>, ..., soh_mean, soh_min, soh_max, soh_spread
    All SOH values are normalised so that cycle-1 = 100 %.
    """
    df = timeseries_df.copy()

    # --- time axis --------------------------------------------------------
    if time_col and time_col in df.columns:
        t_arr = pd.to_numeric(df[time_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    else:
        t_arr = np.arange(len(df), dtype=float)
    dt = np.maximum(np.diff(t_arr, prepend=t_arr[0]), 0.0)

    # --- resolve cycle grouping -------------------------------------------
    group_col: str
    if cycle_col and cycle_col in df.columns:
        group_col = cycle_col
    else:
        # detect charge/discharge cycles via sign changes of total current
        total_i = np.zeros(len(df))
        for col in cell_current_cols:
            total_i += pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=float)

        sign_vec = np.sign(total_i)
        sign_vec[sign_vec == 0] = 1
        change_pts = np.where(np.diff(sign_vec) != 0)[0] + 1

        # pair consecutive sign-change points into one full cycle
        boundaries = np.concatenate([[0], change_pts, [len(df)]])
        # merge every 2 adjacent segments into one cycle
        cycle_ids = np.zeros(len(df), dtype=int)
        cycle_num = 1
        seg_count = 0
        for k in range(len(boundaries) - 1):
            s, e = boundaries[k], boundaries[k + 1]
            cycle_ids[s:e] = cycle_num
            seg_count += 1
            if seg_count >= 2:
                cycle_num += 1
                seg_count = 0
        df["_cycle"] = cycle_ids
        group_col = "_cycle"

    # --- per-cycle capacity -----------------------------------------------
    first_caps: dict = {}
    records = []

    for cycle_id, grp in df.groupby(group_col, sort=True):
        iloc_idx = grp.index
        global_pos = np.where(df.index.isin(iloc_idx))[0]
        dt_grp = dt[global_pos] if len(global_pos) else np.ones(len(grp))

        row: dict = {"cycle": int(cycle_id)}
        capacities: List[float] = []

        for col in cell_current_cols:
            i_arr = pd.to_numeric(grp[col], errors="coerce").fillna(0).to_numpy(dtype=float)
            n_g = min(len(i_arr), len(dt_grp))
            i_arr = i_arr[:n_g]
            dts   = dt_grp[:n_g]

            # prefer discharge half-cycle capacity
            dis_mask = i_arr < 0
            if dis_mask.sum() > 0:
                cap = float(np.sum(np.abs(i_arr[dis_mask]) * dts[dis_mask]) / 3600.0)
            else:
                cap = float(np.sum(np.abs(i_arr) * dts) / 3600.0)

            if col not in first_caps:
                first_caps[col] = max(cap, 1e-6)

            q_nom = nominal_capacity_ah if (nominal_capacity_ah and nominal_capacity_ah > 0) else first_caps[col]
            soh_val = float(np.clip(cap / q_nom * 100.0, 0.0, 120.0))
            row[f"soh_{col}"] = soh_val
            capacities.append(soh_val)

        row["soh_mean"]   = float(np.nanmean(capacities))  if capacities else np.nan
        row["soh_min"]    = float(np.nanmin(capacities))   if capacities else np.nan
        row["soh_max"]    = float(np.nanmax(capacities))   if capacities else np.nan
        row["soh_spread"] = row["soh_max"] - row["soh_min"]
        records.append(row)

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # normalise so that cycle-1 reference = 100 %
    soh_cols = [c for c in result.columns if c.startswith("soh_")]
    for col in soh_cols:
        first_valid = result[col].dropna()
        if not first_valid.empty and first_valid.iloc[0] > 0:
            result[col] = result[col] / first_valid.iloc[0] * 100.0

    return result


# ---------------------------------------------------------------------------
# SOH forecasting
# ---------------------------------------------------------------------------

def forecast_soh(
    soh_df: pd.DataFrame,
    target_col: str = "soh_mean",
    horizon: int = 50,
    method: str = "ensemble",
) -> pd.DataFrame:
    """
    Forecast SOH for `horizon` future cycles.

    method : 'linear' | 'polynomial' | 'ensemble'

    Returns DataFrame: cycle, soh_forecast, lower_ci, upper_ci
    """
    df = soh_df.dropna(subset=[target_col]).copy()
    if len(df) < 3:
        return pd.DataFrame(columns=["cycle", "soh_forecast", "lower_ci", "upper_ci"])

    x = df["cycle"].to_numpy(dtype=float).reshape(-1, 1)
    y = df[target_col].to_numpy(dtype=float)
    last_cycle = int(x.max())
    future_x = np.arange(last_cycle + 1, last_cycle + horizon + 1, dtype=float).reshape(-1, 1)

    preds: dict = {}

    # linear
    lin = LinearRegression().fit(x, y)
    preds["linear"] = lin.predict(future_x)
    y_lin_train = lin.predict(x)

    # polynomial degree-2
    try:
        poly = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg",  Ridge(alpha=1.0)),
        ]).fit(x, y)
        preds["poly2"] = poly.predict(future_x)
        y_poly_train = poly.predict(x)
    except Exception:
        preds["poly2"] = preds["linear"]
        y_poly_train = y_lin_train

    # exponential decay  SOH = a * exp(-b * cycle)
    try:
        from scipy.optimize import curve_fit  # type: ignore
        def _exp(c, a, b): return a * np.exp(-b * c)
        p0 = [100.0, max(1e-6, (y[0] - y[-1]) / (100 * (x[-1, 0] - x[0, 0] + 1)))]
        popt, _ = curve_fit(_exp, x.ravel(), y, p0=p0, maxfev=8000, bounds=([0, 0], [200, 1]))
        preds["exp"] = _exp(future_x.ravel(), *popt)
        y_exp_train = _exp(x.ravel(), *popt)
    except Exception:
        preds["exp"] = preds["linear"]
        y_exp_train = y_lin_train

    # ensemble
    W = {"linear": 0.25, "poly2": 0.40, "exp": 0.35}
    ensemble_future = sum(W[k] * preds[k] for k in W)
    ensemble_train  = 0.25 * y_lin_train + 0.40 * y_poly_train + 0.35 * y_exp_train

    forecast_map = {
        "linear":     (preds["linear"],  y_lin_train),
        "polynomial": (preds["poly2"],   y_poly_train),
        "ensemble":   (ensemble_future,  ensemble_train),
    }
    forecast_vals, y_train_hat = forecast_map.get(method, forecast_map["ensemble"])

    std_res = float(np.std(y - y_train_hat[:len(y)]))
    forecast_vals = np.clip(forecast_vals, 0.0, 100.0)

    return pd.DataFrame({
        "cycle":        future_x.ravel().astype(int),
        "soh_forecast": forecast_vals,
        "lower_ci":     np.clip(forecast_vals - 1.96 * std_res, 0.0, 100.0),
        "upper_ci":     np.clip(forecast_vals + 1.96 * std_res, 0.0, 100.0),
    })


# ---------------------------------------------------------------------------
# RUL estimation
# ---------------------------------------------------------------------------

def estimate_rul(
    soh_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target_col: str = "soh_mean",
    threshold: float = 80.0,
) -> dict:
    """
    Return dict with keys:
        threshold_pct, current_soh, eol_cycle, rul_cycles, already_degraded
    """
    result = {
        "threshold_pct": threshold,
        "current_soh":   np.nan,
        "eol_cycle":     None,
        "rul_cycles":    None,
        "already_degraded": False,
    }

    if not soh_df.empty and target_col in soh_df.columns:
        valid = soh_df[target_col].dropna()
        if not valid.empty:
            result["current_soh"] = float(valid.iloc[-1])

    last_cycle = int(soh_df["cycle"].max()) if not soh_df.empty else 0

    # Check if already below threshold in history
    if not soh_df.empty and target_col in soh_df.columns:
        below = soh_df[soh_df[target_col].fillna(100.0) < threshold]
        if not below.empty:
            eol = int(below["cycle"].iloc[0])
            result["eol_cycle"]        = eol
            result["rul_cycles"]       = max(0, eol - last_cycle)
            result["already_degraded"] = True
            return result

    # Check forecast
    if not forecast_df.empty and "soh_forecast" in forecast_df.columns:
        below = forecast_df[forecast_df["soh_forecast"] < threshold]
        if not below.empty:
            eol = int(below["cycle"].iloc[0])
            result["eol_cycle"]  = eol
            result["rul_cycles"] = max(0, eol - last_cycle)

    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_soh_forecast(
    soh_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    cell_cols: Optional[List[str]] = None,
    target_col: str = "soh_mean",
    rul_info: Optional[dict] = None,
):
    """Plotly figure: historical SOH + per-cell lines + forecast ribbon + EOL."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    COLORS = ["#60a5fa", "#34d399", "#f87171", "#a78bfa", "#fbbf24"]
    fig = go.Figure()

    # per-cell historical
    if cell_cols:
        for idx, col in enumerate(cell_cols):
            sc = f"soh_{col}"
            if sc in soh_df.columns:
                lbl = col.replace("current_a_cell", "Cell").replace("_", " ").strip()
                fig.add_trace(go.Scatter(
                    x=soh_df["cycle"], y=soh_df[sc],
                    mode="lines",
                    name=f"{lbl} (hist.)",
                    line=dict(width=1.5, color=COLORS[idx % len(COLORS)], dash="dot"),
                    opacity=0.55,
                ))

    # mean historical
    if target_col in soh_df.columns:
        fig.add_trace(go.Scatter(
            x=soh_df["cycle"], y=soh_df[target_col],
            mode="lines+markers",
            name="SOH Mean (hist.)",
            line=dict(width=2.5, color="#38bdf8"),
            marker=dict(size=5),
        ))

    # spread band
    if "soh_min" in soh_df.columns and "soh_max" in soh_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([soh_df["cycle"], soh_df["cycle"].iloc[::-1]]),
            y=pd.concat([soh_df["soh_max"], soh_df["soh_min"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(56,189,248,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Cell spread",
        ))

    # forecast
    if not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df["cycle"], y=forecast_df["soh_forecast"],
            mode="lines",
            name="SOH Forecast",
            line=dict(width=2.5, color="#f97316", dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["cycle"], forecast_df["cycle"].iloc[::-1]]),
            y=pd.concat([forecast_df["upper_ci"], forecast_df["lower_ci"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(249,115,22,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95 % CI",
        ))

    # EOL threshold
    fig.add_hline(y=80.0, line_dash="dash", line_color="#ef4444",
                  annotation_text="EOL threshold (80 %)",
                  annotation_position="bottom right",
                  annotation_font_color="#ef4444")

    # EOL vertical line
    if rul_info and rul_info.get("eol_cycle"):
        fig.add_vline(x=rul_info["eol_cycle"], line_dash="dot", line_color="#facc15",
                      annotation_text=f"EOL ≈ cycle {rul_info['eol_cycle']}",
                      annotation_position="top right",
                      annotation_font_color="#facc15")

    fig.update_layout(
        title="SOH Time-Series Forecast & Remaining Useful Life",
        xaxis_title="Cycle",
        yaxis_title="State of Health (%)",
        yaxis=dict(range=[0, 112]),
        paper_bgcolor="#111827",
        plot_bgcolor="#1f2937",
        font=dict(color="#e5e7eb"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#374151"),
        height=520,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def plot_rul_gauge(rul_info: dict):
    """Plotly gauge: current SOH with EOL threshold and RUL annotation."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    soh_val = float(np.nan_to_num(rul_info.get("current_soh", 100.0), nan=100.0))
    rul_cycles = rul_info.get("rul_cycles")
    subtitle = (
        f"RUL ≈ {int(rul_cycles)} cycles remaining"
        if rul_cycles is not None
        else "EOL not predicted within forecast horizon"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=soh_val,
        title={"text": "Current SOH (%)", "font": {"size": 16, "color": "#e5e7eb"}},
        delta={"reference": 100.0,
               "decreasing": {"color": "#ef4444"},
               "increasing": {"color": "#22c55e"}},
        number={"font": {"size": 36, "color": "#f1f5f9"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
            "bar":  {"color": "#3b82f6"},
            "bgcolor": "#1f2937",
            "borderwidth": 2,
            "bordercolor": "#374151",
            "steps": [
                {"range": [0, 60],   "color": "#7f1d1d"},
                {"range": [60, 80],  "color": "#78350f"},
                {"range": [80, 90],  "color": "#1e3a5f"},
                {"range": [90, 100], "color": "#14532d"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 4},
                "thickness": 0.8,
                "value": 80.0,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#111827",
        font={"color": "#e5e7eb"},
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=[dict(
            text=subtitle,
            showarrow=False,
            x=0.5, y=-0.08,
            xref="paper", yref="paper",
            font=dict(size=13, color="#a0aec0"),
        )],
    )
    return fig


def plot_soh_spread(
    soh_df: pd.DataFrame,
    cell_cols: List[str],
):
    """Box-plot of SOH distribution across cells per cycle window."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    COLORS = ["#60a5fa", "#34d399", "#f87171", "#a78bfa"]
    fig = go.Figure()
    for idx, col in enumerate(cell_cols):
        sc = f"soh_{col}"
        if sc not in soh_df.columns:
            continue
        lbl = col.replace("current_a_cell", "Cell").replace("_", " ").strip()
        fig.add_trace(go.Box(
            y=soh_df[sc].dropna(),
            name=lbl,
            marker_color=COLORS[idx % len(COLORS)],
            boxmean="sd",
        ))

    fig.update_layout(
        title="SOH Distribution per Cell (all cycles)",
        yaxis_title="SOH (%)",
        paper_bgcolor="#111827",
        plot_bgcolor="#1f2937",
        font=dict(color="#e5e7eb"),
        height=400,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
