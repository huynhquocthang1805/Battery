from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.cache_manager import (
    cache_clear_all, cache_get, cache_list, cache_set,
    cache_size_mb, fingerprint_dfs, fingerprint_path, get_cache_dir,
)
from src.data_loader import classify_tables, concat_tables, load_dataset_bundle
from src.explainability import auto_explanation_text, summarize_feature_effects
from src.feature_engineering import (
    build_feature_table_from_timeseries, build_risk_scores,
    integrate_characterization_features,
)
from src.graph_model import (
    build_battery_graph, compute_graph_metrics, message_passing_aggregate,
    plot_adjacency_heatmap, plot_battery_graph, plot_gcn_features,
)
from src.inference import rule_based_recommendations
from src.modeling import (
    ModelingResult, MultiOutputModelingResult,
    get_gpu_info, save_model, train_regression_model, train_multi_output_model,
)
from src.preprocessing import prepare_data
from src.soh_forecast import (
    estimate_rul, estimate_soh_per_cycle, forecast_soh,
    plot_rul_gauge, plot_soh_forecast, plot_soh_spread,
)
from src.utils import html_report
from src.visualization import (
    plot_actual_vs_predicted, plot_cell_deviation_from_mean,
    plot_delta_stats_bar, plot_feature_importance, plot_imbalance_dashboard,
    plot_lifetime_index, plot_missing_values, plot_numeric_distribution,
    plot_ocv_curves, plot_pairwise_delta, plot_residuals, plot_risk_gauge,
    plot_rolling_imbalance, plot_timeseries,
)

st.set_page_config(
    page_title="Parallel Battery Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #0f172a 0%, #111827 60%, #0b1220 100%); }
      .block-container { padding-top: 1rem; }
      h1, h2, h3, p, li, label, div { color: #e5e7eb; }
      div[data-testid="stMetricValue"] { font-size: 1.1rem; }

      section[data-testid="stSidebar"] { background: #ffffff !important; }
      section[data-testid="stSidebar"] *,
      section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4,
      section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div,
      section[data-testid="stSidebar"] li, section[data-testid="stSidebar"] a,
      section[data-testid="stSidebar"] .stMarkdown,
      section[data-testid="stSidebar"] .stCaption { color: #1a1a1a !important; }
      section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea {
        color: #1a1a1a !important; background: #f8f9fa !important; border-color: #d1d5db !important;
      }
      section[data-testid="stSidebar"] details summary p { color: #1a1a1a !important; }
      section[data-testid="stSidebar"] code { color: #374151 !important; background: #f3f4f6 !important; }
      section[data-testid="stSidebar"] hr { border-color: #d1d5db !important; }

      div[data-baseweb="select"] > div,
      div[data-baseweb="select"] > div:focus-within {
        background-color: #ffffff !important; color: #1a1a1a !important; border-color: #d1d5db !important;
      }
      div[data-baseweb="select"] span,
      div[data-baseweb="select"] div[class*="ValueContainer"] *,
      div[data-baseweb="select"] div[class*="singleValue"],
      div[data-baseweb="select"] div[class*="placeholder"] { color: #1a1a1a !important; }
      ul[data-baseweb="menu"] li, ul[data-baseweb="menu"] {
        background-color: #ffffff !important; color: #1a1a1a !important;
      }
      ul[data-baseweb="menu"] li:hover { background-color: #f0f4ff !important; }
      div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input {
        background-color: #ffffff !important; color: #1a1a1a !important; border-color: #d1d5db !important;
      }
      .stRadio label p, .stCheckbox label p { color: #e5e7eb; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Constants
# ============================================================

_LEGEND = dict(
    bgcolor="rgba(255,255,255,0.95)", bordercolor="#d1d5db",
    borderwidth=1, font=dict(color="#1a1a1a", size=12),
)
_COLORS = ["#60a5fa","#34d399","#f87171","#a78bfa","#fbbf24","#fb923c","#38bdf8","#f97316"]
_DARK_BG, _DARK_PLOT, _FONT = "#111827", "#1f2937", "#e5e7eb"

_MODEL_DESCRIPTIONS = {
    "Random Forest": (
        "🌲 **Random Forest** — ensemble 300 cây quyết định, mỗi cây train trên random subset. "
        "Kết quả = trung bình tất cả cây → giảm variance, tránh overfit. "
        "Xử lý tốt quan hệ phi tuyến. Feature importance = tổng giảm impurity qua tất cả cây."
    ),
    "XGBoost": (
        "⚡ **XGBoost** — gradient boosting theo trình tự, mỗi cây sửa lỗi cây trước. "
        "Mạnh nhất trong 4 thuật toán, xử lý tốt missing values & feature interactions. "
        "Hỗ trợ GPU (device=cuda). Config: n_estimators=300, max_depth=4, lr=0.05."
    ),
    "Ridge": (
        "📐 **Ridge** — hồi quy tuyến tính + regularization L2 (α·‖w‖²). "
        "Tránh overfit khi features tương quan cao. Nhanh nhất. "
        "Multi-output: mỗi target có bộ coeff riêng chạy song song."
    ),
    "Linear Regression": (
        "📏 **Linear Regression** cổ điển — minimize MSE, hoàn toàn tuyến tính. "
        "Dễ diễn giải nhất, dùng làm baseline. "
        "Multi-output: mỗi target độc lập, chạy song song (MultiOutputRegressor)."
    ),
}

# ============================================================
# Helpers
# ============================================================

def ensure_session_key(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


def get_loaded_tables_by_names(bundle, names: List[str]):
    out, names_set = [], set(names)
    for table in bundle.tables:
        if f"{table.source_file}::{table.table_name}" in names_set:
            out.append(table)
    return out


def filter_useful_timeseries_tables(tables):
    useful = []
    for t in tables:
        cols = {str(c).lower() for c in t.df.columns}
        fn, tn = t.source_file.lower(), t.table_name.lower()
        if tn == "data" and (fn.startswith("m1_") or fn.startswith("m2_")):
            useful.append(t); continue
        if {"test_time_s","current_a","voltage_v"} <= cols:
            useful.append(t); continue
        if any(c.startswith("current_a_cell") for c in cols) or any(c.startswith("temperature_c_cell") for c in cols):
            useful.append(t)
    return useful


def filter_useful_characterization_tables(tables):
    useful = []
    for t in tables:
        name = f"{t.source_file}_{t.table_name}".lower()
        if any(k in name for k in ["hppc","multisine","ocv","capacity","char","dis"]):
            useful.append(t)
    return useful


def get_feature_targets(feature_df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        "current": [c for c in ["sigma_i_start","sigma_i_mid","sigma_i_end","delta_soc_max","delta_soc_end","delta_t_max","sigma_t_mean","ttsb"] if c in feature_df.columns],
        "thermal": [c for c in ["sigma_t_start","sigma_t_mid","sigma_t_end","sigma_t_mean","delta_t_start","delta_t_mid","delta_t_end","delta_t_max","temp_peak","module_temp_gradient_series_auc","ttsb"] if c in feature_df.columns],
        "soh":     [c for c in feature_df.columns if c.lower() in {"soh","rul","remaining_useful_life","capacity_retention"}],
    }


def build_scenario_row(feature_df: pd.DataFrame, controls: Dict[str, object]) -> pd.DataFrame:
    skip = {"degradation_risk_score","relative_lifetime_index","estimated_cycle_life_band","risk_model_features_used"}
    base: Dict[str, object] = {}
    for col in feature_df.columns:
        if col in skip: continue
        if pd.api.types.is_numeric_dtype(feature_df[col]) and feature_df[col].notna().any():
            base[col] = float(feature_df[col].median())
        else:
            mode = feature_df[col].mode(dropna=True)
            base[col] = mode.iloc[0] if not mode.empty else "unknown"
    alias = {"operating_temperature":["operating_temperature","ambient_temperature","test_temperature"],"interconnection_resistance":["interconnection_resistance","branch_resistance"],"chemistry":["chemistry"],"ageing":["ageing","aging"],"ambient_temperature":["ambient_temperature","operating_temperature"]}
    for k, v in controls.items():
        for col in alias.get(k, [k]):
            if col in base: base[col] = v
    return pd.DataFrame([base])


# ============================================================
# Cached loading
# ============================================================

@st.cache_data(show_spinner=False)
def _st_load_bundle(path_str: str):
    fp = fingerprint_path(path_str)
    obj = cache_get("bundle", fp)
    if obj is not None: return obj, True
    bundle = load_dataset_bundle(path_str)
    cache_set("bundle", bundle, fp)
    return bundle, False


@st.cache_data(show_spinner=False)
def _st_prepare_engineer(ts_df: pd.DataFrame, char_df: pd.DataFrame):
    fp = fingerprint_dfs(ts_df, char_df)
    prepared = cache_get("prepared_data", fp)
    if prepared is None:
        prepared = prepare_data(timeseries_df=ts_df, characterization_df=char_df)
        cache_set("prepared_data", prepared, fp)
    if prepared.timeseries_df.empty:
        return prepared, pd.DataFrame(), False
    feature_df = cache_get("feature_df", fp)
    from_cache = feature_df is not None
    if feature_df is None:
        feature_df = build_feature_table_from_timeseries(prepared.timeseries_df)
        if not prepared.characterization_df.empty and not feature_df.empty:
            feature_df = integrate_characterization_features(feature_df, prepared.characterization_df)
        if not feature_df.empty:
            feature_df = build_risk_scores(feature_df)
        cache_set("feature_df", feature_df, fp, use_parquet=True)
    return prepared, feature_df, from_cache


# ============================================================
# Model cache helpers
# ============================================================

def _model_cache_key(model_type: str, target: str, model_name: str) -> str:
    return f"model_{model_type}_{target}_{model_name}".replace(" ","_")


def save_model_to_cache(result, model_type: str) -> None:
    tgt = result.target_col if hasattr(result, "target_col") else "_".join(result.target_cols)
    key = _model_cache_key(model_type, tgt, result.model_name)
    cache_set(key, result, fingerprint=f"{tgt}_{result.model_name}_{result.trained_at}")


def load_model_from_cache(model_type: str, target: str, model_name: str):
    key = _model_cache_key(model_type, target, model_name)
    if not any(m["key"] == key for m in cache_list()): return None
    from src.cache_manager import _load_meta, _read_object
    d = get_cache_dir(); meta = _load_meta(d); info = meta.get(key, {})
    return _read_object(d, key, use_parquet=info.get("parquet", False)) if info else None


# ============================================================
# Sidebar
# ============================================================

def _render_sidebar_info():
    gpu = get_gpu_info()
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚡ Accelerator")
        if gpu["has_cuda"]:  st.success(f"🟢 {gpu['info']}")
        elif gpu["has_mps"]: st.info(f"🔵 {gpu['info']}")
        else:                st.warning(f"🟡 {gpu['info']}")
        st.markdown("### 💾 Disk Cache")
        st.caption(f"Location: `{get_cache_dir().resolve()}`")
        st.caption(f"Size: **{cache_size_mb()} MB**")
        entries = cache_list()
        if entries:
            with st.expander(f"Cached objects ({len(entries)})", expanded=False):
                for e in entries:
                    st.markdown(f"- `{e['key']}`  {e['size_kb']} KB · {e['saved_at']}")
        if st.button("🗑 Clear ALL cache", key="clear_disk_cache"):
            cache_clear_all(); st.cache_data.clear()
            st.success("Cache cleared."); st.rerun()


# ============================================================
# Chart helpers for per-target section
# ============================================================

def _make_scatter_fig(ya, yp, tgt: str, color: str) -> go.Figure:
    """Actual vs Predicted scatter với đường diagonal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ya, y=yp, mode="markers", name="samples",
        marker=dict(size=6, color=color, opacity=0.70,
                    line=dict(width=0.5, color="white")),
    ))
    mn = float(min(ya.min(), yp.min()))
    mx = float(max(ya.max(), yp.max()))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(color="#6b7280", width=1.5, dash="dash"),
        name="ideal (y=x)", showlegend=True,
    ))
    # annotate R²
    r2 = float(np.corrcoef(ya, yp)[0,1] ** 2) if len(ya) > 1 else 0.0
    fig.add_annotation(
        text=f"R²={r2:.3f}", xref="paper", yref="paper",
        x=0.05, y=0.95, showarrow=False,
        font=dict(size=12, color="#f1f5f9"),
        bgcolor="rgba(0,0,0,0.4)", bordercolor=color, borderwidth=1,
    )
    fig.update_layout(
        title=dict(text=f"Actual vs Predicted — <b>{tgt}</b>", font=dict(size=13, color=_FONT)),
        xaxis_title="Actual", yaxis_title="Predicted",
        paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
        font=dict(color=_FONT),
        xaxis=dict(gridcolor="#374151", zerolinecolor="#4b5563"),
        yaxis=dict(gridcolor="#374151", zerolinecolor="#4b5563"),
        legend=_LEGEND, height=340, margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def _make_residual_fig(ya, yp, tgt: str, color: str) -> go.Figure:
    """Residuals (actual − predicted) vs predicted."""
    residuals = ya - yp
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", line_width=1)
    fig.add_trace(go.Scatter(
        x=yp, y=residuals, mode="markers", name="residuals",
        marker=dict(size=5, color=color, opacity=0.60),
    ))
    fig.update_layout(
        title=dict(text=f"Residuals — <b>{tgt}</b>", font=dict(size=13, color=_FONT)),
        xaxis_title="Predicted", yaxis_title="Residual (actual − predicted)",
        paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
        font=dict(color=_FONT),
        xaxis=dict(gridcolor="#374151", zerolinecolor="#4b5563"),
        yaxis=dict(gridcolor="#374151", zerolinecolor="#4b5563"),
        legend=_LEGEND, height=280, margin=dict(l=50, r=20, t=45, b=40),
    )
    return fig


def _make_fi_fig(fi_df: pd.DataFrame, tgt: str, top_k: int, color: str) -> go.Figure:
    """Horizontal bar chart of feature importance for one target."""
    df = fi_df.head(top_k).copy()
    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"],
        orientation="h",
        marker=dict(color=color, opacity=0.82,
                    line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
        text=[f"{v:.4f}" for v in df["importance"]],
        textposition="outside",
        textfont=dict(color=_FONT, size=10),
    ))
    fig.update_layout(
        title=dict(text=f"Feature importance — <b>{tgt}</b>", font=dict(size=13, color=_FONT)),
        xaxis_title="Importance", yaxis_title="",
        yaxis=dict(autorange="reversed", gridcolor="#374151"),
        xaxis=dict(gridcolor="#374151"),
        paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
        font=dict(color=_FONT),
        height=max(280, top_k * 26),
        margin=dict(l=180, r=60, t=45, b=40),
    )
    return fig


# ============================================================
# Multi-output tab renderer
# ============================================================

def _render_multi_output_tab(
    tab_key: str,
    tab_title: str,
    target_list: List[str],
    feature_df: pd.DataFrame,
    tab_description: str,
    target_meaning: str,
):
    """
    Train ALL targets at once (1 call to train_multi_output_model).
    Display results EXPLICITLY per-target:
      • summary table (all targets)
      • importance heatmap (all targets)
      • per-target section: metrics cards + actual-vs-pred + residuals + importance bar
    """
    if feature_df.empty:
        st.warning("Feature table đang rỗng."); return
    if not target_list:
        st.warning(f"Không tìm thấy target columns cho {tab_title}.")
        st.code(str(feature_df.columns.tolist())); return

    # ── Description ──────────────────────────────────────────────────────
    st.markdown(tab_description)
    st.info(f"**{len(target_list)} targets sẽ được train cùng lúc:** `{'`  `'.join(target_list)}`")

    # ── Controls ─────────────────────────────────────────────────────────
    col_m, col_k, col_g = st.columns([2, 1, 1])
    model_name = col_m.selectbox(
        "Thuật toán", ["Random Forest","XGBoost","Ridge","Linear Regression"],
        key=f"{tab_key}_model",
    )
    top_k = col_k.slider("Top-K features", 5, 25, 12, key=f"{tab_key}_topk")
    grp   = col_g.selectbox(
        "Group split",
        ["<none>"] + [c for c in ["test_id","module_id","source_file"] if c in feature_df.columns],
        key=f"{tab_key}_group",
    )

    # ── Model explanation ─────────────────────────────────────────────────
    with st.expander("📖 Giải thích thuật toán", expanded=False):
        st.markdown(_MODEL_DESCRIPTIONS.get(model_name, ""))
        st.markdown(f"""
**Pipeline:**
```
feature_df  →  ColumnTransformer (Impute + Scale + OHE)
            →  {model_name}  [fit(X, Y_matrix)]   ← 1 lần cho tất cả targets
            →  predict(X_test)  →  Y_pred matrix
            →  per-target: MAE, RMSE, R², feature importance
```
{target_meaning}
        """)

    # ── Cache auto-load ───────────────────────────────────────────────────
    ck_tgt = "_".join(target_list)
    if st.session_state.get(f"{tab_key}_result") is None:
        cr = load_model_from_cache(tab_key, ck_tgt, model_name)
        if cr is not None:
            st.session_state[f"{tab_key}_result"] = cr
            st.info(f"💾 Loaded from cache — {cr.trained_at} | {cr.model_name}")

    # ── Train button ──────────────────────────────────────────────────────
    col_btn, col_force = st.columns([3, 1])
    train_clicked = col_btn.button(
        f"🚀 Train ALL {len(target_list)} targets — 1 click",
        key=f"train_{tab_key}", type="primary",
    )
    force_retrain = col_force.checkbox("Force retrain", key=f"force_{tab_key}")

    if train_clicked:
        if not force_retrain:
            cr = load_model_from_cache(tab_key, ck_tgt, model_name)
            if cr is not None:
                st.session_state[f"{tab_key}_result"] = cr
                st.success(f"💾 From cache ({cr.trained_at}). Tick 'Force retrain' để train lại.")
                st.rerun()
        with st.spinner(f"Training {model_name} × {len(target_list)} targets..."):
            try:
                t0  = time.perf_counter()
                res = train_multi_output_model(
                    feature_df,
                    target_cols=target_list,
                    model_name=model_name,
                    group_col=None if grp == "<none>" else grp,
                    exclude_cols=["estimated_cycle_life_band","risk_model_features_used"],
                )
                st.session_state[f"{tab_key}_result"] = res
                save_model_to_cache(res, tab_key)
                elapsed = time.perf_counter() - t0
                gpu_tag = " ⚡ GPU" if res.gpu_used else " CPU"
                st.success(f"✅ Done {elapsed:.2f}s{gpu_tag} | {len(res.target_cols)} targets | cached.")
            except Exception as exc:
                st.error(str(exc)); return

    res: Optional[MultiOutputModelingResult] = st.session_state.get(f"{tab_key}_result")
    if res is None:
        st.caption("Bấm **Train** để bắt đầu."); return

    # ════════════════════════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════════════════════════

    gpu_badge = "⚡ GPU" if res.gpu_used else "CPU"
    st.markdown("---")
    st.markdown(
        f"**Model:** {res.model_name}  &nbsp;|&nbsp;  "
        f"**Trained:** {res.trained_at}  &nbsp;|&nbsp;  {gpu_badge}"
    )

    # ── A. Summary metrics table (all targets in one view) ────────────────
    st.markdown("### 📊 Tổng quan — metrics tất cả targets")
    rows = []
    for tgt, m in res.metrics_per_target.items():
        rows.append({
            "Target": tgt,
            "MAE":  round(m.get("MAE",  np.nan), 5),
            "RMSE": round(m.get("RMSE", np.nan), 5),
            "R²":   round(m.get("R2",   np.nan), 4),
            "Train time (s)": round(m.get("train_time_s", 0), 2),
        })
    mdf = pd.DataFrame(rows).set_index("Target")
    st.dataframe(
        mdf.style
            .background_gradient(cmap="RdYlGn", subset=["R²"], vmin=0, vmax=1)
            .background_gradient(cmap="Reds_r",  subset=["MAE","RMSE"])
            .format({"MAE":"{:.5f}","RMSE":"{:.5f}","R²":"{:.4f}","Train time (s)":"{:.2f}"}),
        use_container_width=True,
    )

    # ── B. Feature importance heatmap (all targets × top features) ────────
    if res.feat_imp_per_target:
        st.markdown("### 🔥 Feature importance heatmap — tất cả targets")
        top_feats = res.feature_importance_df["feature"].head(top_k).tolist()
        hm_data = {
            tgt: [
                res.feat_imp_per_target[tgt].set_index("feature")["importance"].get(f, 0.0)
                for f in top_feats
            ]
            for tgt in res.target_cols if tgt in res.feat_imp_per_target
        }
        hm_df = pd.DataFrame(hm_data, index=top_feats)
        fig_hm = go.Figure(go.Heatmap(
            z=hm_df.values,
            x=hm_df.columns.tolist(), y=hm_df.index.tolist(),
            colorscale="Blues",
            text=[[f"{v:.4f}" for v in row] for row in hm_df.values],
            texttemplate="%{text}", showscale=True,
            colorbar=dict(title="Importance", thickness=14),
        ))
        fig_hm.update_layout(
            title=f"Top-{top_k} features × {len(res.target_cols)} targets",
            xaxis_title="Target", yaxis_title="Feature",
            paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
            font=dict(color=_FONT),
            height=max(340, top_k * 24),
            margin=dict(l=220, r=20, t=55, b=40),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── C. Per-target explicit sections ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Kết quả chi tiết từng target")

    for idx, tgt in enumerate(res.target_cols):
        color = _COLORS[idx % len(_COLORS)]
        m     = res.metrics_per_target.get(tgt, {})
        col_a = f"actual_{tgt}"; col_p = f"predicted_{tgt}"
        has_pred = col_a in res.predictions_df.columns

        # Section header with colored pill
        st.markdown(
            f"<div style='background:{color}22;border-left:4px solid {color};"
            f"padding:8px 14px;border-radius:0 6px 6px 0;margin:18px 0 10px 0;'>"
            f"<span style='color:{color};font-weight:600;font-size:15px;'>{tgt}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Metrics cards (4 columns)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mae_v  = m.get("MAE",  np.nan)
        rmse_v = m.get("RMSE", np.nan)
        r2_v   = m.get("R2",   np.nan)
        tt_v   = m.get("train_time_s", 0.0)

        mc1.metric("MAE",  f"{mae_v:.5f}"  if not np.isnan(mae_v)  else "N/A")
        mc2.metric("RMSE", f"{rmse_v:.5f}" if not np.isnan(rmse_v) else "N/A")

        # R² with delta from 1.0
        if not np.isnan(r2_v):
            delta_r2 = round(r2_v - 1.0, 4)
            mc3.metric("R²", f"{r2_v:.4f}", delta=f"{delta_r2:.4f}",
                       delta_color="inverse" if delta_r2 < 0 else "normal")
        else:
            mc3.metric("R²", "N/A")

        mc4.metric("Train time", f"{tt_v:.2f} s")

        if not has_pred:
            st.caption("(không có dự báo cho target này)")
            continue

        ya = res.predictions_df[col_a].to_numpy()
        yp = res.predictions_df[col_p].to_numpy()

        # 2 columns: scatter + feature importance
        left_col, right_col = st.columns(2)
        with left_col:
            st.plotly_chart(_make_scatter_fig(ya, yp, tgt, color), use_container_width=True)
        with right_col:
            fi_df = res.feat_imp_per_target.get(tgt)
            if fi_df is not None:
                st.plotly_chart(_make_fi_fig(fi_df, tgt, top_k, color), use_container_width=True)

        # Residuals (expandable)
        with st.expander(f"📉 Residuals — {tgt}", expanded=False):
            st.plotly_chart(_make_residual_fig(ya, yp, tgt, color), use_container_width=True)
            # residual stats
            res_arr = ya - yp
            rs1, rs2, rs3, rs4 = st.columns(4)
            rs1.metric("Mean residual",  f"{float(res_arr.mean()):.5f}")
            rs2.metric("Std residual",   f"{float(res_arr.std()):.5f}")
            rs3.metric("Max |residual|", f"{float(np.abs(res_arr).max()):.5f}")
            rs4.metric("Samples",        f"{len(res_arr)}")

    # ── D. Download ───────────────────────────────────────────────────────
    st.markdown("---")
    st.download_button(
        "⬇ Download metrics CSV",
        mdf.reset_index().to_csv(index=False).encode(),
        file_name=f"{tab_key}_metrics.csv", mime="text/csv",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    st.title("Parallel-Connected Multi-Battery Analytics Dashboard")
    st.caption("4-cell module · imbalance · thermal · Graph GCN · SOH Forecast & RUL · 💾 Cache · ⚡ GPU")

    for key in ["bundle","bundle_error","dataset_path","soh_hist","soh_forecast","rul_info"]:
        ensure_session_key(key, None if key != "dataset_path" else "")

    with st.sidebar:
        st.header("Dataset Configuration")
        dataset_path = st.text_input("Dataset path", value=st.session_state.get("dataset_path",""))
        if st.button("Load dataset", type="primary"):
            st.session_state["dataset_path"] = dataset_path
            st.session_state["bundle_error"] = None
            try:
                with st.spinner("Loading..."):
                    bundle_obj, from_cache = _st_load_bundle(dataset_path)
                    st.session_state["bundle"] = bundle_obj
                    st.success("✅ From cache." if from_cache else "✅ Loaded & cached.")
            except Exception as exc:
                st.session_state["bundle"] = None; st.session_state["bundle_error"] = str(exc)
        if st.button("Clear Streamlit cache"):
            st.cache_data.clear()
            for k in ["bundle","bundle_error"]: st.session_state.pop(k, None)
            st.rerun()

    _render_sidebar_info()

    if st.session_state.get("bundle_error"):
        st.error(f"Error: {st.session_state['bundle_error']}"); return
    bundle = st.session_state.get("bundle")
    if bundle is None:
        st.info("Nhập dataset path và bấm **Load dataset**."); return
    if bundle.catalog.empty:
        st.error("Không tìm thấy bảng dữ liệu.")
        if bundle.errors: st.write(bundle.errors)
        return

    grouped = classify_tables(bundle)
    grouped["timeseries"]       = filter_useful_timeseries_tables(grouped.get("timeseries",[])) or grouped.get("timeseries",[])
    grouped["characterization"] = filter_useful_characterization_tables(grouped.get("characterization",[])) or grouped.get("characterization",[])
    ts_options   = [f"{t.source_file}::{t.table_name}" for t in grouped.get("timeseries",[])]
    char_options = [f"{t.source_file}::{t.table_name}" for t in grouped.get("characterization",[])]

    with st.sidebar:
        st.subheader("Table Selection")
        use_all_ts   = st.checkbox("Use all timeseries tables",      value=True)
        use_all_char = st.checkbox("Use all characterization tables", value=True)
        sel_ts   = ts_options   if use_all_ts   else st.multiselect("Timeseries",       ts_options,   ts_options[:min(10,len(ts_options))])
        sel_char = char_options if use_all_char else st.multiselect("Characterization", char_options, char_options[:min(10,len(char_options))])

    ts_df   = concat_tables(get_loaded_tables_by_names(bundle, sel_ts),   add_source_cols=True)
    char_df = concat_tables(get_loaded_tables_by_names(bundle, sel_char), add_source_cols=True)

    try:
        with st.spinner("Preprocessing & feature engineering..."):
            t0 = time.perf_counter()
            prepared, feature_df, feat_cache = _st_prepare_engineer(ts_df, char_df)
            elapsed = time.perf_counter() - t0
        st.toast(f"{'💾 Cache' if feat_cache else '⚙️ Done'} ({elapsed:.2f}s)")
    except Exception as exc:
        st.error(f"Error: {exc}"); st.stop()

    targets = get_feature_targets(feature_df)
    schema  = prepared.schema_timeseries

    tabs = st.tabs([
        "📋 Overview","🔬 Cell Characterization","⚡ Current Imbalance",
        "🌡️ Forecast Thermal","📈 Forecast Imbalance",
        "🩺 SoH / Risk","📡 Graph Network Analysis","🔮 SOH Forecast & RUL",
        "🔍 Explainability","🎯 Scenario Simulator","📤 Export",
    ])

    # ── 0: Overview ──────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Dataset Overview")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Tables",         f"{len(bundle.catalog)}")
        c2.metric("Timeseries rows",f"{len(prepared.timeseries_df):,}")
        c3.metric("Char rows",      f"{len(prepared.characterization_df):,}")
        c4.metric("Feature rows",   f"{len(feature_df):,}")
        st.dataframe(bundle.catalog, use_container_width=True)
        if bundle.errors: st.warning("Parse errors:"); st.write(bundle.errors)
        if prepared.notes: st.info("\n".join(prepared.notes))
        st.dataframe(prepared.timeseries_df.head(20), use_container_width=True)
        with st.expander("Debug"):
            st.write("TS shape:", prepared.timeseries_df.shape)
            st.write("Feature cols:", feature_df.columns.tolist())
        fig = plot_missing_values(prepared.timeseries_df)
        if fig: st.plotly_chart(fig, use_container_width=True)

    # ── 1: Characterization ──────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Cell Characterization")
        if prepared.characterization_df.empty:
            st.warning("Chưa có characterization table.")
        else:
            num_cols = prepared.characterization_df.select_dtypes(include=[np.number]).columns.tolist()
            cap_cols = [c for c in num_cols if "capacity" in c]
            res_cols = [c for c in num_cols if any(k in c for k in ["resistance","r0","ohmic"])]
            ocv_cols = [c for c in num_cols if "ocv" in c]
            cc = st.selectbox("Color by", ["<none>"]+[c for c in ["ageing","chemistry"] if c in prepared.characterization_df.columns], key="char_color")
            cc = None if cc == "<none>" else cc
            if cap_cols: st.plotly_chart(plot_numeric_distribution(prepared.characterization_df, cap_cols[0], color=cc, title="Capacity distribution"), use_container_width=True)
            if res_cols: st.plotly_chart(plot_numeric_distribution(prepared.characterization_df, res_cols[0], color=cc, title="Resistance distribution"), use_container_width=True)
            if ocv_cols:
                xc = [c for c in prepared.characterization_df.columns if c not in ocv_cols][:5]
                if xc: st.plotly_chart(plot_ocv_curves(prepared.characterization_df.head(500), xc[0], ocv_cols[:6]), use_container_width=True)
            st.dataframe(prepared.characterization_df.head(50), use_container_width=True)

    # ── 2: Current Imbalance ─────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Current Imbalance Analysis")
        if prepared.timeseries_df.empty or schema is None or not schema.cell_current_cols:
            st.warning("Không có timeseries hoặc cell current cols.")
        else:
            grp_c = [c for c in ["test_id","module_id","source_file","source_table","synthetic_test_id"] if c in prepared.timeseries_df.columns]
            case_df = prepared.timeseries_df.copy()
            if grp_c:
                sc = grp_c[0]
                sv = st.selectbox("Test condition", case_df[sc].astype(str).unique().tolist(), key="analysis_case")
                case_df = case_df[case_df[sc].astype(str) == sv].copy()
            ck1,ck2,ck3 = st.columns(3)
            k_i = ck1.number_input("Scale k (current)", 0.01, 1000.0, 1.0, 0.1, format="%.3f", key="k_current")
            k_t = ck2.number_input("Scale k (thermal)", 0.01, 1000.0, 1.0, 0.1, format="%.3f", key="k_thermal")
            rw  = ck3.number_input("Rolling window", 2, 5000, 50, 10, key="roll_win")
            sa  = st.toggle("Show |delta|", value=True, key="show_abs")
            st.markdown("---"); st.markdown("### Current")
            for fig in [
                plot_timeseries(case_df, schema.time_col, schema.cell_current_cols, "Raw current [A]"),
                plot_cell_deviation_from_mean(case_df, schema.time_col, schema.cell_current_cols, f"Deviation × {k_i:.3g} [A]", k_i, "A"),
                plot_pairwise_delta(case_df, schema.time_col, schema.cell_current_cols, f"Pairwise delta × {k_i:.3g} [A]", k_i, "A", sa),
                plot_rolling_imbalance(case_df, schema.time_col, schema.cell_current_cols, int(rw), f"Rolling σ × {k_i:.3g} [A]", k_i, "A"),
                plot_delta_stats_bar(case_df, schema.cell_current_cols, k_i, "A"),
            ]:
                if fig: st.plotly_chart(fig, use_container_width=True)
            with st.expander("📊 4-panel Dashboard", expanded=False):
                st.plotly_chart(plot_imbalance_dashboard(case_df, schema.time_col, schema.cell_current_cols, k_i, int(rw), "A", "Current Dashboard"), use_container_width=True)
            if schema.cell_temp_cols:
                st.markdown("---"); st.markdown("### Temperature")
                for fig in [
                    plot_timeseries(case_df, schema.time_col, schema.cell_temp_cols, "Raw temperature [°C]"),
                    plot_cell_deviation_from_mean(case_df, schema.time_col, schema.cell_temp_cols, f"Deviation × {k_t:.3g} [°C]", k_t, "°C"),
                    plot_pairwise_delta(case_df, schema.time_col, schema.cell_temp_cols, f"Pairwise delta × {k_t:.3g} [°C]", k_t, "°C", sa),
                    plot_rolling_imbalance(case_df, schema.time_col, schema.cell_temp_cols, int(rw), f"Rolling σ × {k_t:.3g} [°C]", k_t, "°C"),
                    plot_delta_stats_bar(case_df, schema.cell_temp_cols, k_t, "°C"),
                ]:
                    if fig: st.plotly_chart(fig, use_container_width=True)
                with st.expander("🌡️ 4-panel Thermal Dashboard", expanded=False):
                    st.plotly_chart(plot_imbalance_dashboard(case_df, schema.time_col, schema.cell_temp_cols, k_t, int(rw), "°C", "Thermal Dashboard"), use_container_width=True)

    # ── 3: Forecast Thermal ──────────────────────────────────────────────
    with tabs[3]:
        st.subheader("🌡️ Forecast Thermal")
        _render_multi_output_tab(
            tab_key="thermal", tab_title="Thermal",
            target_list=targets["thermal"], feature_df=feature_df,
            tab_description=(
                "Train **một lần** trên toàn bộ thermal targets. "
                "Kết quả hiển thị đầy đủ **tường minh từng target**: "
                "metrics, actual vs predicted, residuals, feature importance."
            ),
            target_meaning=(
                "**Ý nghĩa các target:**\n"
                "- `sigma_t_start/mid/end` — std nhiệt độ giữa các cell ở đầu/giữa/cuối chu kỳ\n"
                "- `delta_t_start/mid/end/max` — chênh lệch nhiệt độ tối đa giữa cell nóng/lạnh nhất\n"
                "- `temp_peak` — nhiệt độ đỉnh trong toàn bộ test\n"
                "- `module_temp_gradient_series_auc` — AUC của gradient nhiệt (tổng năng lượng nhiệt bất đối xứng)\n"
                "- `sigma_t_mean` — trung bình std nhiệt độ\n"
                "- `ttsb` — time-to-steady-balance nhiệt độ"
            ),
        )

    # ── 4: Forecast Imbalance ────────────────────────────────────────────
    with tabs[4]:
        st.subheader("📈 Forecast Current Imbalance")
        _render_multi_output_tab(
            tab_key="imbalance", tab_title="Current Imbalance",
            target_list=targets["current"], feature_df=feature_df,
            tab_description=(
                "Train **một lần** trên toàn bộ current imbalance targets. "
                "Kết quả hiển thị đầy đủ **tường minh từng target**: "
                "metrics, actual vs predicted, residuals, feature importance."
            ),
            target_meaning=(
                "**Ý nghĩa các target:**\n"
                "- `sigma_i_start/mid/end` — std dòng điện giữa 4 cell (đo mức độ mất cân bằng)\n"
                "- `delta_soc_max` — khoảng lệch SoC lớn nhất trong chu kỳ\n"
                "- `delta_soc_end` — lệch SoC cuối chu kỳ (tích lũy qua nhiều vòng)\n"
                "- `delta_t_max` — chênh lệch nhiệt độ tối đa giữa các cell\n"
                "- `sigma_t_mean` — mean std nhiệt độ (thermal imbalance tổng thể)\n"
                "- `ttsb` — time-to-steady-balance dòng điện"
            ),
        )

    # ── 5: SoH / Risk ────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("SoH / Degradation Risk")
        if feature_df.empty: st.warning("Feature table rỗng.")
        else:
            c1, c2 = st.columns(2)
            if "degradation_risk_score" in feature_df.columns:
                c1.plotly_chart(plot_risk_gauge(float(feature_df["degradation_risk_score"].mean()), "Avg degradation risk"), use_container_width=True)
            if "relative_lifetime_index" in feature_df.columns:
                c2.plotly_chart(plot_lifetime_index(float(feature_df["relative_lifetime_index"].mean()), "Avg lifetime index"), use_container_width=True)

    # ── 6: Graph ─────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("📡 Graph Network Analysis")
        if prepared.timeseries_df.empty or schema is None or not schema.cell_current_cols:
            st.warning("Cần time-series với cell current cols.")
        else:
            grp_c = [c for c in ["test_id","module_id","source_file","source_table","synthetic_test_id"] if c in prepared.timeseries_df.columns]
            gdf = prepared.timeseries_df.copy()
            if grp_c:
                sc = grp_c[0]
                sv = st.selectbox("Test condition (graph)", gdf[sc].astype(str).unique().tolist(), key="graph_case")
                gdf = gdf[gdf[sc].astype(str) == sv].copy()
            ca,cb,cc2 = st.columns(3)
            ai = ca.slider("α current",0.0,1.0,0.50,0.05,key="g_ai")
            at = cb.slider("α thermal",0.0,1.0,0.30,0.05,key="g_at")
            as_= cc2.slider("α SOC",   0.0,1.0,0.20,0.05,key="g_as")
            ni = st.slider("GCN iterations",1,5,2,key="gcn_iter")
            with st.spinner("Building graph..."):
                try:
                    adj,nf,labels = build_battery_graph(gdf,schema.cell_current_cols,schema.cell_temp_cols or [],schema.time_col,ai,at,as_)
                    nf_gcn = message_passing_aggregate(adj,nf,n_iter=ni)
                    gm = compute_graph_metrics(adj,labels)
                except Exception as exc: st.error(f"Graph error: {exc}"); st.stop()
            st.markdown("#### Node Metrics")
            st.dataframe(gm.set_index("node").style.format("{:.4f}"), use_container_width=True)
            st.caption(f"Fiedler value: {gm['fiedler_value'].iloc[0] if not gm.empty else 0:.4f}")
            for fig,hdr in [(plot_battery_graph(adj,nf,labels,gm),"#### Network Graph"),(plot_adjacency_heatmap(adj,labels),"#### Coupling Matrix"),(plot_gcn_features(nf,nf_gcn,labels),"#### Raw vs GCN Features")]:
                if fig: st.markdown(hdr); st.plotly_chart(fig, use_container_width=True)

    # ── 7: SOH Forecast ───────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("🔮 SOH Forecast & RUL")
        if prepared.timeseries_df.empty or schema is None or not schema.cell_current_cols:
            st.warning("Cần time-series với cell current cols.")
        else:
            c1,c2,c3 = st.columns(3)
            soh_thr = c1.slider("EOL threshold (%)",60.0,90.0,80.0,1.0,key="rul_thr")
            horizon = c2.slider("Forecast horizon",10,500,100,10,key="rul_horizon")
            method  = c3.selectbox("Method",["ensemble","linear","polynomial"],key="rul_method")
            nom_cap = st.number_input("Nominal cap (Ah), 0=auto",0.0,1000.0,0.0,0.1,key="nominal_cap")
            grp_c = [c for c in ["test_id","module_id","source_file","source_table","synthetic_test_id"] if c in prepared.timeseries_df.columns]
            soh_raw = prepared.timeseries_df.copy()
            if grp_c:
                sc=grp_c[0]; sv=st.selectbox("Test condition (SOH)",soh_raw[sc].astype(str).unique().tolist(),key="soh_case")
                soh_raw=soh_raw[soh_raw[sc].astype(str)==sv].copy()
            soh_fp = fingerprint_dfs(soh_raw)
            if st.session_state.get("soh_hist") is None:
                cs2 = cache_get("soh_hist", soh_fp)
                if cs2 is not None: st.session_state["soh_hist"]=cs2; st.info("💾 SOH from cache.")
            if st.button("Compute SOH & Forecast", key="run_soh"):
                with st.spinner("Estimating..."):
                    try:
                        cyc = next((c for c in ["cycle","cycle_number","cycle_id"] if c in soh_raw.columns),None)
                        sh  = estimate_soh_per_cycle(soh_raw,schema.cell_current_cols,schema.time_col,cyc,nom_cap if nom_cap>0 else None)
                        st.session_state["soh_hist"]=sh; cache_set("soh_hist",sh,soh_fp,use_parquet=True)
                    except Exception as exc: st.error(str(exc)); st.stop()
                with st.spinner("Forecasting..."):
                    try:
                        sf=forecast_soh(sh,"soh_mean",horizon,method)
                        rl=estimate_rul(sh,sf,"soh_mean",soh_thr)
                        st.session_state["soh_forecast"]=sf; st.session_state["rul_info"]=rl
                    except Exception as exc: st.error(str(exc)); st.stop()
            sh=st.session_state.get("soh_hist"); sf=st.session_state.get("soh_forecast"); rl=st.session_state.get("rul_info")
            if sh is not None:
                g1,g2,g3,g4=st.columns(4)
                cs3=rl.get("current_soh",np.nan) if rl else np.nan
                g1.metric("Current SOH",f"{cs3:.1f}%" if not np.isnan(cs3) else "N/A")
                g2.metric("Cycles",f"{int(sh['cycle'].max())}" if not sh.empty else "0")
                g3.metric("EOL",f"Cycle {rl.get('eol_cycle')}" if rl and rl.get('eol_cycle') else "—")
                g4.metric("RUL",f"{int(rl['rul_cycles'])}" if rl and rl.get('rul_cycles') is not None else "—")
                if rl:
                    fg=plot_rul_gauge(rl)
                    if fg: st.plotly_chart(fg,use_container_width=True)
                ff=plot_soh_forecast(sh,sf or pd.DataFrame(),schema.cell_current_cols,"soh_mean",rl)
                if ff: st.plotly_chart(ff,use_container_width=True)
                fb=plot_soh_spread(sh,schema.cell_current_cols)
                if fb: st.plotly_chart(fb,use_container_width=True)
                with st.expander("Per-cycle table"): st.dataframe(sh,use_container_width=True)
                if sf is not None and not sf.empty:
                    comb=pd.concat([sh.assign(type="historical"),sf.rename(columns={"soh_forecast":"soh_mean"}).assign(type="forecast")],ignore_index=True)
                    st.download_button("⬇ Download SOH CSV",comb.to_csv(index=False).encode(),"soh_forecast.csv","text/csv")
            else:
                st.info("Bấm **Compute SOH & Forecast** để bắt đầu.")

    # ── 8: Explainability ─────────────────────────────────────────────────
    with tabs[8]:
        st.subheader("Explainability")
        choice = st.selectbox("Model",["thermal_result","imbalance_result"], key="explain_choice")
        result = st.session_state.get(choice)
        if result is None: st.warning("Hãy train ít nhất một model trước.")
        else:
            fi = result.feature_importance_df
            fg = plot_feature_importance(fi, "Mean feature importance")
            if fg: st.plotly_chart(fg, use_container_width=True)

    # ── 9: Scenario ───────────────────────────────────────────────────────
    with tabs[9]:
        st.subheader("Scenario Simulator")
        if feature_df.empty: st.warning("Feature table rỗng.")
        else:
            a,b,c=st.columns(3)
            controls={"operating_temperature":a.slider("Operating temp (°C)",0.0,60.0,25.0,1.0),"interconnection_resistance":b.slider("Interconnect R (mΩ)",0.0,5.0,1.0,0.1),"chemistry":c.selectbox("Chemistry",["NMC","NCA","Mixed"])}
            d,e=st.columns(2); controls["ageing"]=d.selectbox("Ageing",["unaged","aged"]); controls["ambient_temperature"]=e.slider("Ambient temp (°C)",0.0,60.0,25.0,1.0)
            sc_df=build_scenario_row(feature_df,controls); sc_df=build_risk_scores(sc_df)
            st.dataframe(sc_df,use_container_width=True)
            for rec in rule_based_recommendations(sc_df.iloc[0]): st.write(f"- {rec}")

    # ── 10: Export ────────────────────────────────────────────────────────
    with tabs[10]:
        st.subheader("Export")
        st.download_button("⬇ Features CSV",feature_df.to_csv(index=False).encode(),"features.csv","text/csv")
        report=html_report("Battery Report",{"Overview":bundle.catalog.to_html(index=False),"Notes":"<br>".join(prepared.notes) if prepared.notes else "—"})
        st.download_button("⬇ HTML Report",report.encode(),"report.html","text/html")


if __name__ == "__main__":
    main()
