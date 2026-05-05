from __future__ import annotations

from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_CELL_COLORS = ["#60a5fa", "#34d399", "#f87171", "#a78bfa", "#fbbf24", "#fb923c"]
_DELTA_COLORS = [
    "#f97316", "#e879f9", "#2dd4bf", "#facc15",
    "#f43f5e", "#a3e635", "#38bdf8",
]
_DARK_BG   = "#111827"
_DARK_PLOT = "#1f2937"
_FONT_CLR  = "#e5e7eb"

# Legend style: white box, black text — readable on both dark chart + light sidebar
_LEGEND = dict(
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="#d1d5db",
    borderwidth=1,
    font=dict(color="#1a1a1a", size=12),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dark_layout(fig, title: str, xaxis_title: str = "",
                 yaxis_title: str = "", height: int = 420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=_FONT_CLR)),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_PLOT,
        font=dict(color=_FONT_CLR),
        xaxis=dict(title=xaxis_title, gridcolor="#374151", zerolinecolor="#4b5563"),
        yaxis=dict(title=yaxis_title, gridcolor="#374151", zerolinecolor="#4b5563"),
        legend=_LEGEND,
        height=height,
        margin=dict(l=50, r=20, t=55, b=45),
    )
    return fig


def _short(col: str) -> str:
    """Turn 'current_a_cell_1' -> 'C1', 'temperature_c_cell_2' -> 'T2'."""
    c = col.lower()
    for pre in ("current_a_cell_", "temperature_c_cell_",
                "current_a_", "temperature_c_", "temp_cell_", "cell_"):
        if c.startswith(pre):
            suffix = c[len(pre):]
            prefix_char = "C" if "current" in pre else "T"
            return f"{prefix_char}{suffix}"
    return col


def _hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    """
    Convert a 6-character hex color (#rrggbb) to an rgba() string.
    Plotly does NOT accept 8-character hex (#rrggbbaa) — always use rgba().
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(96,165,250,{alpha})"   # safe fallback (blue)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Unchanged helpers
# ---------------------------------------------------------------------------

def plot_numeric_distribution(df, col, color=None, title=None):
    if col not in df.columns:
        return None
    return px.histogram(df, x=col, color=color, marginal="box",
                        nbins=30, title=title or col)


def plot_categorical_distribution(df, col):
    if col not in df.columns:
        return None
    counts = df[col].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = [col, "count"]
    return px.bar(counts, x=col, y="count", title=f"Distribution: {col}")


def plot_ocv_curves(df, x_col, y_cols):
    if x_col not in df.columns:
        return None
    fig = go.Figure()
    for y in y_cols:
        if y in df.columns:
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y], mode="lines", name=y))
    fig.update_layout(title="OCV curves", legend=_LEGEND)
    return fig


def plot_timeseries(df, time_col, y_cols, title):
    fig = go.Figure()
    for idx, y in enumerate(y_cols):
        if y in df.columns:
            fig.add_trace(go.Scatter(
                x=df[time_col], y=df[y], mode="lines", name=y,
                line=dict(width=1.8, color=_CELL_COLORS[idx % len(_CELL_COLORS)]),
            ))
    _dark_layout(fig, title, xaxis_title=time_col)
    return fig


def plot_missing_values(df):
    if df.empty:
        return None
    miss = df.isna().mean().sort_values(ascending=False).head(30)
    return px.bar(x=miss.index, y=miss.values, title="Missing value ratio (top 30)")


def plot_correlation_heatmap(df, columns, title):
    cols = [c for c in columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        return None
    return px.imshow(df[cols].corr(numeric_only=True), text_auto=True, title=title)


def plot_actual_vs_predicted(pred_df, title):
    if pred_df.empty:
        return None
    return px.scatter(pred_df, x="actual", y="predicted", trendline="ols", title=title)


def plot_residuals(pred_df, title):
    if pred_df.empty:
        return None
    tmp = pred_df.copy()
    tmp["residual"] = tmp["actual"] - tmp["predicted"]
    return px.scatter(tmp, x="predicted", y="residual", title=title)


def plot_feature_importance(feat_df, title):
    if feat_df is None or feat_df.empty:
        return None
    return px.bar(feat_df.head(20), x="importance", y="feature",
                  orientation="h", title=title)


def plot_risk_gauge(value, title="Risk"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text": title},
        gauge={"axis": {"range": [0, 100]}},
    ))
    return fig


def plot_lifetime_index(value, title="Relative lifetime index"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text": title},
        gauge={"axis": {"range": [0, 100]}},
    ))
    return fig


def scenario_comparison_bar(df, x_col, y_cols, title):
    if df.empty:
        return None
    return px.bar(df, x=x_col, y=y_cols[0], title=title)


# ---------------------------------------------------------------------------
# Delta / imbalance plots
# ---------------------------------------------------------------------------

def plot_cell_deviation_from_mean(
    df: pd.DataFrame,
    time_col: str,
    cell_cols: List[str],
    title: str = "Deviation from module mean  (cell_i - mean) x k",
    scale_factor: float = 1.0,
    unit_label: str = "",
) -> go.Figure:
    """
    For each cell: deviation_i = (cell_i − mean_of_all_cells) × scale_factor.
    Fill area between the trace and y=0 to emphasise the imbalance region.
    """
    valid = [c for c in cell_cols if c in df.columns]
    if not valid or time_col not in df.columns:
        return go.Figure()

    mat = df[valid].apply(pd.to_numeric, errors="coerce")
    mu  = mat.mean(axis=1)
    fig = go.Figure()
    fig.add_hline(y=0.0, line_dash="dot", line_color="#6b7280", line_width=1)

    for idx, col in enumerate(valid):
        dev        = (mat[col] - mu) * scale_factor
        lbl        = _short(col)
        color      = _CELL_COLORS[idx % len(_CELL_COLORS)]
        fill_color = _hex_to_rgba(color, alpha=0.13)
        fig.add_trace(go.Scatter(
            x=df[time_col], y=dev,
            mode="lines",
            name=f"D-{lbl}",
            line=dict(width=1.8, color=color),
            fill="tozeroy",
            fillcolor=fill_color,
        ))

    ylab = f"(cell - mean) x {scale_factor:.3g}  {unit_label}".strip()
    _dark_layout(fig, title, xaxis_title=time_col, yaxis_title=ylab, height=380)
    return fig


def plot_pairwise_delta(
    df: pd.DataFrame,
    time_col: str,
    cell_cols: List[str],
    title: str = "Pairwise delta between cells  (cell_i - cell_j) x k",
    scale_factor: float = 1.0,
    unit_label: str = "",
    show_abs: bool = False,
) -> go.Figure:
    """
    C(N,2) pair traces.  4 cells → 6 pairs: C1-C2, C1-C3, C1-C4, C2-C3, C2-C4, C3-C4.
    show_abs=True plots |Δ_ij| so all curves stay positive.
    """
    valid = [c for c in cell_cols if c in df.columns]
    if len(valid) < 2 or time_col not in df.columns:
        return go.Figure()

    mat   = df[valid].apply(pd.to_numeric, errors="coerce")
    pairs = list(combinations(range(len(valid)), 2))
    fig   = go.Figure()
    fig.add_hline(y=0.0, line_dash="dot", line_color="#6b7280", line_width=1)

    for pidx, (i, j) in enumerate(pairs):
        d   = (mat[valid[i]] - mat[valid[j]]) * scale_factor
        if show_abs:
            d = d.abs()
        li  = _short(valid[i])
        lj  = _short(valid[j])
        lbl = f"|{li}-{lj}|" if show_abs else f"{li}-{lj}"
        fig.add_trace(go.Scatter(
            x=df[time_col], y=d,
            mode="lines", name=lbl,
            line=dict(width=1.6, color=_DELTA_COLORS[pidx % len(_DELTA_COLORS)]),
        ))

    ylab = f"delta x {scale_factor:.3g}  {unit_label}".strip()
    _dark_layout(fig, title, xaxis_title=time_col, yaxis_title=ylab, height=400)
    return fig


def plot_rolling_imbalance(
    df: pd.DataFrame,
    time_col: str,
    cell_cols: List[str],
    window: int = 50,
    title: str = "Rolling imbalance index  sigma(cells) x k",
    scale_factor: float = 1.0,
    unit_label: str = "",
) -> go.Figure:
    """Rolling cross-cell σ — shows whether imbalance is growing over time."""
    valid = [c for c in cell_cols if c in df.columns]
    if not valid or time_col not in df.columns:
        return go.Figure()

    mat     = df[valid].apply(pd.to_numeric, errors="coerce")
    sigma   = mat.std(axis=1) * scale_factor
    roll_mu = sigma.rolling(window=window, min_periods=1).mean()
    roll_mx = sigma.rolling(window=window, min_periods=1).max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[time_col], y=sigma,
        mode="lines", name="sigma (instant)",
        line=dict(width=0.9, color="#6b7280"), opacity=0.4,
    ))
    fig.add_trace(go.Scatter(
        x=df[time_col], y=roll_mu,
        mode="lines", name=f"rolling mean (w={window})",
        line=dict(width=2.2, color="#38bdf8"),
    ))
    fig.add_trace(go.Scatter(
        x=df[time_col], y=roll_mx,
        mode="lines", name=f"rolling peak (w={window})",
        line=dict(width=1.4, color="#f97316", dash="dash"),
    ))

    ylab = f"sigma x {scale_factor:.3g}  {unit_label}".strip()
    _dark_layout(fig, title, xaxis_title=time_col, yaxis_title=ylab, height=360)
    return fig


def plot_imbalance_dashboard(
    df: pd.DataFrame,
    time_col: str,
    cell_cols: List[str],
    scale_factor: float = 1.0,
    window: int = 50,
    unit_label: str = "",
    main_title: str = "Imbalance Dashboard",
) -> go.Figure:
    """
    4-panel subplot:
      [1,1] Raw signals        [1,2] Deviation from mean × k
      [2,1] Pairwise |delta|   [2,2] Rolling sigma × k
    """
    valid = [c for c in cell_cols if c in df.columns]
    if not valid or time_col not in df.columns:
        return go.Figure()

    mat   = df[valid].apply(pd.to_numeric, errors="coerce")
    mu    = mat.mean(axis=1)
    pairs = list(combinations(range(len(valid)), 2))
    t     = df[time_col]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Raw signals per cell",
            f"Deviation from mean  x {scale_factor:.3g}",
            f"Pairwise |delta|  x {scale_factor:.3g}",
            f"Rolling sigma (w={window})  x {scale_factor:.3g}",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.09,
    )

    # panel 1: raw
    for idx, col in enumerate(valid):
        fig.add_trace(go.Scatter(
            x=t, y=mat[col], mode="lines",
            name=_short(col),
            line=dict(width=1.6, color=_CELL_COLORS[idx % len(_CELL_COLORS)]),
            legendgroup=f"r{idx}",
        ), row=1, col=1)

    # panel 2: deviation from mean
    for idx, col in enumerate(valid):
        dev        = (mat[col] - mu) * scale_factor
        color      = _CELL_COLORS[idx % len(_CELL_COLORS)]
        fill_color = _hex_to_rgba(color, alpha=0.13)
        fig.add_trace(go.Scatter(
            x=t, y=dev, mode="lines",
            name=f"D-{_short(col)}",
            line=dict(width=1.6, color=color),
            fill="tozeroy",
            fillcolor=fill_color,
            legendgroup=f"d{idx}", showlegend=True,
        ), row=1, col=2)

    # panel 3: pairwise |delta|
    for pidx, (i, j) in enumerate(pairs):
        d  = (mat[valid[i]] - mat[valid[j]]).abs() * scale_factor
        li = _short(valid[i])
        lj = _short(valid[j])
        fig.add_trace(go.Scatter(
            x=t, y=d, mode="lines",
            name=f"|{li}-{lj}|",
            line=dict(width=1.5, color=_DELTA_COLORS[pidx % len(_DELTA_COLORS)]),
            legendgroup=f"p{pidx}",
        ), row=2, col=1)

    # panel 4: rolling sigma
    sigma  = mat.std(axis=1) * scale_factor
    roll_s = sigma.rolling(window=window, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=t, y=sigma, mode="lines", name="sigma instant",
        line=dict(width=0.8, color="#6b7280"), opacity=0.4, showlegend=False,
        legendgroup="s0",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=t, y=roll_s, mode="lines", name=f"sigma rolling w={window}",
        line=dict(width=2.2, color="#38bdf8"),
        legendgroup="s1",
    ), row=2, col=2)

    fig.update_layout(
        title=dict(text=main_title, font=dict(size=15, color=_FONT_CLR)),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_PLOT,
        font=dict(color=_FONT_CLR),
        height=720,
        margin=dict(l=55, r=20, t=85, b=45),
        legend=_LEGEND,
    )
    for ax in fig.layout:
        if ax.startswith(("xaxis", "yaxis")):
            fig.layout[ax].update(gridcolor="#374151", zerolinecolor="#4b5563")
    return fig


def plot_delta_stats_bar(
    df: pd.DataFrame,
    cell_cols: List[str],
    scale_factor: float = 1.0,
    unit_label: str = "",
) -> go.Figure:
    """Summary bar: mean ± std and max |Δ_ij| for every cell pair."""
    valid = [c for c in cell_cols if c in df.columns]
    if len(valid) < 2:
        return go.Figure()

    mat    = df[valid].apply(pd.to_numeric, errors="coerce")
    pairs  = list(combinations(range(len(valid)), 2))
    labels, means, maxes, stds = [], [], [], []

    for i, j in pairs:
        d = (mat[valid[i]] - mat[valid[j]]).abs() * scale_factor
        li, lj = _short(valid[i]), _short(valid[j])
        labels.append(f"|{li}-{lj}|")
        means.append(float(d.mean()))
        maxes.append(float(d.max()))
        stds.append(float(d.std()))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Mean |delta|", x=labels, y=means,
        marker_color="#38bdf8",
        error_y=dict(type="data", array=stds, visible=True),
    ))
    fig.add_trace(go.Bar(
        name="Max |delta|", x=labels, y=maxes,
        marker_color="#f97316", opacity=0.75,
    ))

    u = f" [{unit_label}]" if unit_label else ""
    _dark_layout(
        fig,
        title=f"Pairwise imbalance stats  (x{scale_factor:.3g}){u}",
        xaxis_title="Cell pair",
        yaxis_title=f"|delta|{u}",
        height=360,
    )
    fig.update_layout(barmode="group")
    return fig
