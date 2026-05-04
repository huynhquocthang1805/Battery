"""graph_model.py
Graph-based analysis for parallel battery modules.

4 cells = 4 nodes.
Edges are weighted by the combined similarity:
  - Current imbalance  (ΔI between cells, weight alpha_current)
  - Thermal coupling   (ΔT between cells, weight alpha_thermal)
  - SOC deviation      (ΔSOC between cells, weight alpha_soc)

w_ij = exp( -D_ij )   where D_ij is normalised composite distance.

A lightweight numpy Message-Passing step (GCN-style) aggregates
neighbour features into richer node embeddings used downstream
for SOH forecasting.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float_mat(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    """Return (T, N) float matrix, forward-filled then zero-filled."""
    mat = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # forward fill
    for j in range(mat.shape[1]):
        col = mat[:, j]
        mask = np.isnan(col)
        if mask.any() and not mask.all():
            idx = np.where(~mask)[0]
            col[mask] = np.interp(np.where(mask)[0], idx, col[idx])
        col[np.isnan(col)] = 0.0
        mat[:, j] = col
    return mat


def _pairwise_mean_abs_diff(mat: np.ndarray) -> np.ndarray:
    """(T, N) -> (N, N) mean |x_i - x_j| matrix."""
    n = mat.shape[1]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = float(np.nanmean(np.abs(mat[:, i] - mat[:, j])))
    return D


def _coulomb_soc_col(t_arr: np.ndarray, i_arr: np.ndarray) -> np.ndarray:
    dt = np.diff(t_arr, prepend=t_arr[0])
    dt = np.maximum(dt, 0.0)
    cap = max(float(np.sum(np.abs(i_arr) * dt) / 3600.0), 1e-6)
    return 1.0 - np.cumsum(i_arr * dt / 3600.0) / cap


def _normalise(D: np.ndarray) -> np.ndarray:
    mx = D.max()
    return D / mx if mx > 0 else D


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_battery_graph(
    timeseries_df: pd.DataFrame,
    cell_current_cols: List[str],
    cell_temp_cols: Optional[List[str]] = None,
    time_col: Optional[str] = None,
    alpha_current: float = 0.50,
    alpha_thermal: float = 0.30,
    alpha_soc: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build weighted (N x N) adjacency matrix for the N-cell battery graph.

    Returns
    -------
    adj          : (N, N) similarity matrix  (0 = unrelated, 1 = identical)
    node_features: (N, 5) matrix  [mean_I, std_I, mean_T, std_T, mean_SoC]
    labels       : human-readable cell names
    """
    n = len(cell_current_cols)
    labels = [
        c.replace("current_a_cell", "Cell")
         .replace("current_a_", "Cell")
         .replace("_", " ")
         .strip()
        for c in cell_current_cols
    ]

    # --- time axis --------------------------------------------------------
    if time_col and time_col in timeseries_df.columns:
        t_arr = pd.to_numeric(timeseries_df[time_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    else:
        t_arr = np.arange(len(timeseries_df), dtype=float)

    # --- current matrix ---------------------------------------------------
    curr_mat = _to_float_mat(timeseries_df, cell_current_cols)  # (T, N)
    D_current = _pairwise_mean_abs_diff(curr_mat)

    # --- thermal matrix ---------------------------------------------------
    D_thermal = np.zeros((n, n))
    if cell_temp_cols and len(cell_temp_cols) >= n:
        temp_mat = _to_float_mat(timeseries_df, cell_temp_cols[:n])
        D_thermal = _pairwise_mean_abs_diff(temp_mat)
    elif cell_temp_cols and len(cell_temp_cols) > 0:
        temp_mat = _to_float_mat(timeseries_df, cell_temp_cols)
        # broadcast to n columns
        temp_full = np.zeros((len(timeseries_df), n))
        for k in range(n):
            temp_full[:, k] = temp_mat[:, k % temp_mat.shape[1]]
        D_thermal = _pairwise_mean_abs_diff(temp_full)

    # --- SOC matrix -------------------------------------------------------
    soc_cols_list: List[np.ndarray] = []
    D_soc = np.zeros((n, n))
    try:
        for col in cell_current_cols:
            i_arr = pd.to_numeric(timeseries_df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
            soc_cols_list.append(_coulomb_soc_col(t_arr, i_arr))
        soc_mat = np.column_stack(soc_cols_list)  # (T, N)
        D_soc = _pairwise_mean_abs_diff(soc_mat)
    except Exception:
        soc_mat = np.full((len(timeseries_df), n), 0.5)

    # --- combine ----------------------------------------------------------
    D_combined = (
        alpha_current * _normalise(D_current)
        + alpha_thermal * _normalise(D_thermal)
        + alpha_soc    * _normalise(D_soc)
    )
    adj = np.exp(-D_combined)
    np.fill_diagonal(adj, 0.0)

    # --- node features  [mean_I, std_I, mean_T, std_T, mean_SoC] ---------
    node_features_list = []
    for idx, col in enumerate(cell_current_cols):
        i_arr = curr_mat[:, idx]
        if cell_temp_cols and idx < len(cell_temp_cols):
            t2 = _to_float_mat(timeseries_df, [cell_temp_cols[idx]])[:, 0]
        else:
            t2 = np.full(len(i_arr), 25.0)
        soc_arr = soc_cols_list[idx] if idx < len(soc_cols_list) else np.full(len(i_arr), 0.5)
        node_features_list.append([
            float(np.nanmean(i_arr)),
            float(np.nanstd(i_arr)),
            float(np.nanmean(t2)),
            float(np.nanstd(t2)),
            float(np.nanmean(soc_arr)),
        ])

    return adj, np.array(node_features_list), labels


def message_passing_aggregate(
    adj: np.ndarray,
    node_features: np.ndarray,
    n_iter: int = 2,
) -> np.ndarray:
    """
    GCN-style symmetric normalised message passing (numpy).

        H_new = D^{-1/2} (A + I) D^{-1/2}  H

    Returns enriched node feature matrix (N, F).
    """
    A = adj.copy()
    np.fill_diagonal(A, 1.0)  # self-loop
    deg = A.sum(axis=1)
    inv_sqrt_deg = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
    A_norm = inv_sqrt_deg @ A @ inv_sqrt_deg
    H = node_features.copy()
    for _ in range(n_iter):
        H = A_norm @ H
    return H


def compute_graph_metrics(
    adj: np.ndarray,
    labels: List[str],
) -> pd.DataFrame:
    """Return per-node DataFrame with graph-theoretic health indicators."""
    n = adj.shape[0]

    # Weighted degree
    degree = adj.sum(axis=1)

    # Weighted clustering coefficient
    clustering = []
    for i in range(n):
        nbrs = np.where(adj[i] > 0)[0]
        k = len(nbrs)
        if k < 2:
            clustering.append(0.0)
            continue
        s = 0.0
        for a in nbrs:
            for b in nbrs:
                if a != b:
                    s += adj[i, a] * adj[a, b] * adj[b, i]
        denom = k * (k - 1)
        clustering.append(float(s / denom) if denom > 0 else 0.0)

    # PageRank (power iteration, alpha=0.85)
    pr = np.ones(n) / n
    col_sum = adj.sum(axis=0)
    col_sum[col_sum == 0] = 1.0
    A_col = adj / col_sum[np.newaxis, :]
    for _ in range(60):
        pr = 0.85 * A_col @ pr + 0.15 / n

    # Fiedler value (algebraic connectivity) from graph Laplacian
    L = np.diag(degree) - adj
    try:
        eigvals = np.linalg.eigvalsh(L)
        fiedler = float(sorted(eigvals)[1]) if n > 1 else 0.0
    except Exception:
        fiedler = 0.0

    # Degradation propagation risk score = degree * clustering
    risk = degree * np.array(clustering)

    rows = []
    for i, label in enumerate(labels):
        rows.append({
            "node": label,
            "degree_weighted": float(degree[i]),
            "clustering_coeff": float(clustering[i]),
            "pagerank": float(pr[i]),
            "degradation_propagation_risk": float(risk[i]),
            "fiedler_value": round(fiedler, 6),
        })
    return pd.DataFrame(rows)


def plot_battery_graph(
    adj: np.ndarray,
    node_features: np.ndarray,
    labels: List[str],
    graph_metrics_df: pd.DataFrame,
    title: str = "Battery Module Graph – 4-Cell Network",
):
    """Interactive Plotly network diagram of the 4-cell battery graph."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {i: (float(np.cos(a)), float(np.sin(a))) for i, a in enumerate(angles)}

    # --- edges ------------------------------------------------------------
    traces = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(adj[i, j])
            if w < 0.02:
                continue
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=max(0.8, w * 7), color=f"rgba(99,179,237,{min(0.85, w)})"),
                hoverinfo="none",
                showlegend=False,
            ))
            # edge-weight label
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            traces.append(go.Scatter(
                x=[mx], y=[my],
                mode="text",
                text=[f"{w:.3f}"],
                textfont=dict(size=9, color="#94a3b8"),
                hoverinfo="none",
                showlegend=False,
            ))

    # --- nodes ------------------------------------------------------------
    pr_arr  = graph_metrics_df["pagerank"].to_numpy()
    risk_arr = graph_metrics_df["degradation_propagation_risk"].to_numpy()
    sizes = 40 + 60 * pr_arr / max(pr_arr.max(), 1e-9)

    hover_texts = [
        (
            f"<b>{row['node']}</b><br>"
            f"Degree: {row['degree_weighted']:.3f}<br>"
            f"Clustering: {row['clustering_coeff']:.3f}<br>"
            f"PageRank: {row['pagerank']:.4f}<br>"
            f"Degrad. Prop. Risk: {row['degradation_propagation_risk']:.4f}<br>"
            f"Fiedler: {row['fiedler_value']:.4f}<br>"
            f"Mean I: {node_features[i, 0]:.3f} A | Std I: {node_features[i, 1]:.3f} A<br>"
            f"Mean T: {node_features[i, 2]:.1f} °C | Std T: {node_features[i, 3]:.2f} °C<br>"
            f"Mean SoC: {node_features[i, 4]:.2%}"
        )
        for i, (_, row) in enumerate(graph_metrics_df.iterrows())
    ]

    traces.append(go.Scatter(
        x=[pos[i][0] for i in range(n)],
        y=[pos[i][1] for i in range(n)],
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=12, color="#f1f5f9"),
        marker=dict(
            size=list(sizes),
            color=list(risk_arr),
            colorscale="RdYlGn_r",
            cmin=0.0,
            colorbar=dict(title="Degrad.<br>Prop. Risk", thickness=14),
            line=dict(width=2, color="white"),
        ),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#f1f5f9")),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#e5e7eb"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.6, 1.6]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.6, 1.6]),
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


def plot_adjacency_heatmap(adj: np.ndarray, labels: List[str]):
    """Plotly heatmap of the (N x N) cell-coupling similarity matrix."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    text_vals = [[f"{adj[i, j]:.3f}" for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(
        z=adj,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=text_vals,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Similarity", thickness=14),
        zmin=0.0,
        zmax=1.0,
    ))
    fig.update_layout(
        title="Cell-to-Cell Coupling Similarity Matrix",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#e5e7eb"),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_gcn_features(
    node_features_raw: np.ndarray,
    node_features_gcn: np.ndarray,
    labels: List[str],
    feature_names: Optional[List[str]] = None,
):
    """Bar chart comparing raw vs GCN-aggregated node features."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if feature_names is None:
        feature_names = ["mean_I", "std_I", "mean_T", "std_T", "mean_SoC"]

    COLORS = ["#60a5fa", "#34d399", "#f87171", "#a78bfa"]
    fig = go.Figure()
    for ni, label in enumerate(labels):
        fig.add_trace(go.Bar(
            name=f"{label} raw",
            x=feature_names,
            y=list(node_features_raw[ni]),
            marker_color=COLORS[ni % len(COLORS)],
            opacity=0.5,
        ))
        fig.add_trace(go.Bar(
            name=f"{label} GCN",
            x=feature_names,
            y=list(node_features_gcn[ni]),
            marker_color=COLORS[ni % len(COLORS)],
            opacity=1.0,
        ))

    fig.update_layout(
        barmode="group",
        title="Raw vs GCN-Aggregated Node Features",
        xaxis_title="Feature",
        yaxis_title="Value",
        paper_bgcolor="#111827",
        plot_bgcolor="#1f2937",
        font=dict(color="#e5e7eb"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
