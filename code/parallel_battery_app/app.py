from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.cache_manager import (
    cache_clear_all,
    cache_get,
    cache_list,
    cache_set,
    cache_size_mb,
    fingerprint_dfs,
    fingerprint_path,
    get_cache_dir,
)
from src.data_loader import classify_tables, concat_tables, load_dataset_bundle
from src.explainability import auto_explanation_text, summarize_feature_effects
from src.feature_engineering import (
    build_feature_table_from_timeseries,
    build_risk_scores,
    integrate_characterization_features,
)
from src.graph_model import (
    build_battery_graph,
    compute_graph_metrics,
    message_passing_aggregate,
    plot_adjacency_heatmap,
    plot_battery_graph,
    plot_gcn_features,
)
from src.inference import rule_based_recommendations
from src.modeling import ModelingResult, get_gpu_info, save_model, train_regression_model
from src.preprocessing import prepare_data
from src.soh_forecast import (
    estimate_rul,
    estimate_soh_per_cycle,
    forecast_soh,
    plot_rul_gauge,
    plot_soh_forecast,
    plot_soh_spread,
)
from src.utils import html_report
from src.visualization import (
    plot_actual_vs_predicted,
    plot_cell_deviation_from_mean,
    plot_delta_stats_bar,
    plot_feature_importance,
    plot_imbalance_dashboard,
    plot_lifetime_index,
    plot_missing_values,
    plot_numeric_distribution,
    plot_ocv_curves,
    plot_pairwise_delta,
    plot_residuals,
    plot_risk_gauge,
    plot_rolling_imbalance,
    plot_timeseries,
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Helpers
# ============================================================

def ensure_session_key(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


def display_model_metrics(metrics: Dict[str, float], cv_scores: Optional[List[float]], gpu_used: bool = False) -> None:
    cols = st.columns(4)
    cols[0].metric("MAE",  f"{metrics.get('MAE',  np.nan):.4f}")
    cols[1].metric("RMSE", f"{metrics.get('RMSE', np.nan):.4f}")
    cols[2].metric("R²",   f"{metrics.get('R2',   np.nan):.4f}")
    train_t = metrics.get("train_time_s")
    gpu_tag  = " ⚡ GPU" if gpu_used else ""
    cols[3].metric("Train time", f"{train_t:.2f} s{gpu_tag}" if train_t else "—")
    if cv_scores:
        st.caption(f"CV RMSE: mean={np.mean(cv_scores):.4f}, std={np.std(cv_scores):.4f}")


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
        if {"test_time_s", "current_a", "voltage_v"} <= cols:
            useful.append(t); continue
        if any(c.startswith("current_a_cell") for c in cols) or any(c.startswith("temperature_c_cell") for c in cols):
            useful.append(t)
    return useful


def filter_useful_characterization_tables(tables):
    useful = []
    for t in tables:
        name = f"{t.source_file}_{t.table_name}".lower()
        if any(k in name for k in ["hppc", "multisine", "ocv", "capacity", "char", "dis"]):
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
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            base[col] = float(feature_df[col].median()) if feature_df[col].notna().any() else 0.0
        else:
            mode = feature_df[col].mode(dropna=True)
            base[col] = mode.iloc[0] if not mode.empty else "unknown"
    alias = {"operating_temperature":["operating_temperature","ambient_temperature","test_temperature"],"interconnection_resistance":["interconnection_resistance","branch_resistance"],"chemistry":["chemistry"],"ageing":["ageing","aging"],"ambient_temperature":["ambient_temperature","operating_temperature"]}
    for k, v in controls.items():
        for col in alias.get(k, [k]):
            if col in base: base[col] = v
    return pd.DataFrame([base])


# ============================================================
# Cached data loading with DISK CACHE
# ============================================================

@st.cache_data(show_spinner=False)
def _st_load_bundle(path_str: str):
    """Streamlit in-memory cache wrapping disk cache."""
    fp = fingerprint_path(path_str)
    obj = cache_get("bundle", fp)
    if obj is not None:
        return obj, True          # (bundle, from_cache)
    bundle = load_dataset_bundle(path_str)
    cache_set("bundle", bundle, fp)
    return bundle, False


@st.cache_data(show_spinner=False)
def _st_prepare_engineer(timeseries_df: pd.DataFrame, characterization_df: pd.DataFrame):
    fp = fingerprint_dfs(timeseries_df, characterization_df)

    # prepared data
    prepared = cache_get("prepared_data", fp)
    if prepared is None:
        prepared = prepare_data(timeseries_df=timeseries_df, characterization_df=characterization_df)
        cache_set("prepared_data", prepared, fp)

    if prepared.timeseries_df.empty:
        return prepared, pd.DataFrame(), False

    # feature table
    feature_df = cache_get("feature_df", fp, )
    from_cache  = feature_df is not None
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
    return f"model_{model_type}_{target}_{model_name}".replace(" ", "_")


def save_model_to_cache(result: ModelingResult, model_type: str) -> None:
    key = _model_cache_key(model_type, result.target_col, result.model_name)
    fp  = f"{result.target_col}_{result.model_name}_{result.trained_at}"
    cache_set(key, result, fingerprint=fp)


def load_model_from_cache(model_type: str, target: str, model_name: str) -> Optional[ModelingResult]:
    key = _model_cache_key(model_type, target, model_name)
    meta_list = cache_list()
    entry = next((m for m in meta_list if m["key"] == key), None)
    if entry is None:
        return None
    fp = f"{target}_{model_name}_"    # partial match — just load whatever is saved
    # use a wildcard: read meta directly
    from src.cache_manager import _load_meta, get_cache_dir, _read_object
    d     = get_cache_dir()
    meta  = _load_meta(d)
    info  = meta.get(key, {})
    if not info:
        return None
    return _read_object(d, key, use_parquet=info.get("parquet", False))


# ============================================================
# Sidebar GPU + Cache panel
# ============================================================

def _render_sidebar_info():
    gpu = get_gpu_info()

    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚡ Accelerator")
        if gpu["has_cuda"]:
            st.success(f"🟢 {gpu['info']}")
        elif gpu["has_mps"]:
            st.info(f"🔵 {gpu['info']}")
        else:
            st.warning(f"🟡 {gpu['info']}")

        st.markdown("### 💾 Disk Cache")
        cache_mb = cache_size_mb()
        st.caption(f"Location: `{get_cache_dir().resolve()}`")
        st.caption(f"Size: **{cache_mb} MB**")

        entries = cache_list()
        if entries:
            with st.expander(f"Cached objects ({len(entries)})", expanded=False):
                for e in entries:
                    st.markdown(f"- `{e['key']}`  {e['size_kb']} KB  ·  {e['saved_at']}")

        if st.button("🗑 Clear ALL cache", key="clear_disk_cache"):
            n = cache_clear_all()
            st.cache_data.clear()
            st.success(f"Removed {n} cached files.")
            st.rerun()


# ============================================================
# MAIN
# ============================================================

def main():
    st.title("Parallel-Connected Multi-Battery Analytics Dashboard")
    st.caption(
        "Phân tích module song song 4 cell — current imbalance · thermal · "
        "Graph (GCN) · SOH Forecast & RUL · Explainability · Scenario · "
        "💾 Persistent cache · ⚡ GPU acceleration"
    )

    for key in ["current_model_result","thermal_model_result","soh_model_result","bundle","bundle_error","dataset_path"]:
        ensure_session_key(key, None if key != "dataset_path" else "")

    # ---- sidebar -------------------------------------------------------
    with st.sidebar:
        st.header("Dataset Configuration")
        dataset_path = st.text_input("Dataset path", value=st.session_state.get("dataset_path",""), help="Trỏ tới thư mục hoặc file CSV/XLSX/MAT.")
        if st.button("Load dataset", type="primary"):
            st.session_state["dataset_path"] = dataset_path
            st.session_state["bundle_error"] = None
            try:
                with st.spinner("Đang đọc dữ liệu..."):
                    bundle_obj, from_cache = _st_load_bundle(dataset_path)
                    st.session_state["bundle"] = bundle_obj
                    if from_cache:
                        st.success("✅ Dataset loaded from disk cache (instant).")
                    else:
                        st.success("✅ Dataset loaded & cached to disk.")
            except Exception as exc:
                st.session_state["bundle"] = None
                st.session_state["bundle_error"] = str(exc)

        if st.button("Clear Streamlit cache"):
            st.cache_data.clear()
            for k in ["bundle","bundle_error","current_model_result","thermal_model_result","soh_model_result"]:
                st.session_state.pop(k, None)
            st.rerun()

    _render_sidebar_info()

    if st.session_state.get("bundle_error"):
        st.error(f"Không thể load dataset: {st.session_state['bundle_error']}")
        return

    bundle = st.session_state.get("bundle")
    if bundle is None:
        st.info("Nhập dataset path ở sidebar và bấm **Load dataset** để bắt đầu.")
        return
    if bundle.catalog.empty:
        st.error("Không tìm thấy bảng dữ liệu hợp lệ.")
        if bundle.errors: st.write(bundle.errors)
        return

    grouped = classify_tables(bundle)
    grouped["timeseries"]      = filter_useful_timeseries_tables(grouped.get("timeseries",[])) or grouped.get("timeseries",[])
    grouped["characterization"]= filter_useful_characterization_tables(grouped.get("characterization",[])) or grouped.get("characterization",[])

    ts_options   = [f"{t.source_file}::{t.table_name}" for t in grouped.get("timeseries",[])]
    char_options = [f"{t.source_file}::{t.table_name}" for t in grouped.get("characterization",[])]

    with st.sidebar:
        st.subheader("Table Selection")
        use_all_ts   = st.checkbox("Use all detected timeseries tables",      value=True)
        use_all_char = st.checkbox("Use all detected characterization tables", value=True)
        selected_ts_names   = ts_options   if use_all_ts   else st.multiselect("Timeseries tables",       ts_options,   ts_options[:min(10,len(ts_options))])
        selected_char_names = char_options if use_all_char else st.multiselect("Characterization tables", char_options, char_options[:min(10,len(char_options))])

    timeseries_tables       = get_loaded_tables_by_names(bundle, selected_ts_names)
    characterization_tables = get_loaded_tables_by_names(bundle, selected_char_names)
    timeseries_df           = concat_tables(timeseries_tables,       add_source_cols=True)
    characterization_df     = concat_tables(characterization_tables, add_source_cols=True)

    try:
        with st.spinner("Đang tiền xử lý và sinh feature..."):
            t0 = time.perf_counter()
            prepared, feature_df, feat_from_cache = _st_prepare_engineer(timeseries_df, characterization_df)
            elapsed_prep = time.perf_counter() - t0
        if feat_from_cache:
            st.toast(f"✅ Feature table loaded from cache ({elapsed_prep:.2f}s)", icon="💾")
        else:
            st.toast(f"✅ Feature engineering done & cached ({elapsed_prep:.2f}s)", icon="⚙️")
    except Exception as exc:
        st.error(f"Lỗi preprocessing / feature engineering: {exc}")
        st.stop()

    targets = get_feature_targets(feature_df)
    schema  = prepared.schema_timeseries

    # ================================================================
    # TABS
    # ================================================================
    tabs = st.tabs([
        "📋 Overview",
        "🔬 Cell Characterization",
        "⚡ Current Imbalance",
        "🌡️ Forecast Thermal",
        "📈 Forecast Imbalance",
        "🩺 SoH / Risk",
        "📡 Graph Network Analysis",
        "🔮 SOH Forecast & RUL",
        "🔍 Explainability",
        "🎯 Scenario Simulator",
        "📤 Export",
    ])

    # ---------------------------------------------------------------- 0
    with tabs[0]:
        st.subheader("Dataset Overview")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Loaded tables",           f"{len(bundle.catalog)}")
        c2.metric("Timeseries rows",         f"{len(prepared.timeseries_df):,}")
        c3.metric("Characterization rows",   f"{len(prepared.characterization_df):,}")
        c4.metric("Engineered feature rows", f"{len(feature_df):,}")
        st.dataframe(bundle.catalog, use_container_width=True)
        if bundle.errors: st.warning("Một số file/tables không parse được."); st.write(bundle.errors)
        if prepared.notes: st.info("\n".join(prepared.notes))
        st.markdown("**Preview timeseries**")
        st.dataframe(prepared.timeseries_df.head(20), use_container_width=True)
        with st.expander("Debug"):
            st.write("Timeseries shape:", prepared.timeseries_df.shape)
            st.write("Feature columns:", feature_df.columns.tolist())
        fig = plot_missing_values(prepared.timeseries_df)
        if fig: st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------- 1
    with tabs[1]:
        st.subheader("Cell Characterization")
        if prepared.characterization_df.empty:
            st.warning("Chưa có characterization table.")
        else:
            num_cols = prepared.characterization_df.select_dtypes(include=[np.number]).columns.tolist()
            cap_cols = [c for c in num_cols if "capacity" in c]
            res_cols = [c for c in num_cols if any(k in c for k in ["resistance","r0","ohmic"])]
            ocv_cols = [c for c in num_cols if "ocv" in c]
            color_col = st.selectbox("Color by", ["<none>"]+[c for c in ["ageing","chemistry"] if c in prepared.characterization_df.columns], key="char_color")
            color_col = None if color_col == "<none>" else color_col
            if cap_cols: st.plotly_chart(plot_numeric_distribution(prepared.characterization_df, cap_cols[0], color=color_col, title="Capacity distribution"), use_container_width=True)
            if res_cols: st.plotly_chart(plot_numeric_distribution(prepared.characterization_df, res_cols[0], color=color_col, title="Resistance distribution"), use_container_width=True)
            if ocv_cols:
                x_cands = [c for c in prepared.characterization_df.columns if c not in ocv_cols][:5]
                if x_cands: st.plotly_chart(plot_ocv_curves(prepared.characterization_df.head(500), x_cands[0], ocv_cols[:6]), use_container_width=True)
            st.dataframe(prepared.characterization_df.head(50), use_container_width=True)

    # ================================================================ 2
    with tabs[2]:
        st.subheader("Current Imbalance Analysis")
        if prepared.timeseries_df.empty or schema is None or not schema.cell_current_cols:
            st.warning("Không có timeseries hoặc không tìm thấy cột dòng điện riêng từng cell.")
        else:
            grp_cands = [c for c in ["test_id","module_id","source_file","source_table","synthetic_test_id"] if c in prepared.timeseries_df.columns]
            case_df = prepared.timeseries_df.copy()
            if grp_cands:
                sc = grp_cands[0]; vals = case_df[sc].astype(str).unique().tolist()
                sv = st.selectbox("Select test condition", vals, key="analysis_case")
                case_df = case_df[case_df[sc].astype(str) == sv].copy()
            st.markdown("#### Delta visualisation controls")
            col_ka, col_kb, col_kc = st.columns(3)
            k_current = col_ka.number_input("Scale k (current)", min_value=0.01, max_value=1000.0, value=1.0, step=0.1, format="%.3f", key="k_current")
            k_thermal = col_kb.number_input("Scale k (thermal)", min_value=0.01, max_value=1000.0, value=1.0, step=0.1, format="%.3f", key="k_thermal")
            roll_win  = col_kc.number_input("Rolling window", min_value=2, max_value=5000, value=50, step=10, key="roll_win")
            show_abs  = st.toggle("Show |delta|", value=True, key="show_abs")
            st.markdown("---"); st.markdown("### Current (dòng điện)")
            for fig in [
                plot_timeseries(case_df, schema.time_col, schema.cell_current_cols, "Raw current per cell [A]"),
                plot_cell_deviation_from_mean(case_df, schema.time_col, schema.cell_current_cols, f"Deviation from mean × {k_current:.3g} [A]", k_current, "A"),
                plot_pairwise_delta(case_df, schema.time_col, schema.cell_current_cols, f"Pairwise delta × {k_current:.3g} [A]", k_current, "A", show_abs),
                plot_rolling_imbalance(case_df, schema.time_col, schema.cell_current_cols, int(roll_win), f"Rolling σ × {k_current:.3g} [A]", k_current, "A"),
                plot_delta_stats_bar(case_df, schema.cell_current_cols, k_current, "A"),
            ]:
                if fig: st.plotly_chart(fig, use_container_width=True)
            with st.expander("📊 4-panel Current Dashboard", expanded=False):
                st.plotly_chart(plot_imbalance_dashboard(case_df, schema.time_col, schema.cell_current_cols, k_current, int(roll_win), "A", "Current Imbalance Dashboard"), use_container_width=True)
            if schema.cell_temp_cols:
                st.markdown("---"); st.markdown("### Temperature (nhiệt độ)")
                for fig in [
                    plot_timeseries(case_df, schema.time_col, schema.cell_temp_cols, "Raw temperature per cell [°C]"),
                    plot_cell_deviation_from_mean(case_df, schema.time_col, schema.cell_temp_cols, f"Deviation from mean × {k_thermal:.3g} [°C]", k_thermal, "°C"),
                    plot_pairwise_delta(case_df, schema.time_col, schema.cell_temp_cols, f"Pairwise delta × {k_thermal:.3g} [°C]", k_thermal, "°C", show_abs),
                    plot_rolling_imbalance(case_df, schema.time_col, schema.cell_temp_cols, int(roll_win), f"Rolling σ × {k_thermal:.3g} [°C]", k_thermal, "°C"),
                    plot_delta_stats_bar(case_df, schema.cell_temp_cols, k_thermal, "°C"),
                ]:
                    if fig: st.plotly_chart(fig, use_container_width=True)
                with st.expander("🌡️ 4-panel Thermal Dashboard", expanded=False):
                    st.plotly_chart(plot_imbalance_dashboard(case_df, schema.time_col, schema.cell_temp_cols, k_thermal, int(roll_win), "°C", "Thermal Imbalance Dashboard"), use_container_width=True)

    # ================================================================
    # Shared training helper with cache
    # ================================================================
    def _train_tab(tab_key: str, target_list: list, tab_label: str):
        if feature_df.empty:
            st.warning("Feature table đang rỗng."); return
        if not target_list:
            st.warning(f"Không đủ target cho {tab_label}."); st.write(feature_df.columns.tolist()); return

        tgt   = st.selectbox("Target",       target_list, key=f"{tab_key}_target")
        mname = st.selectbox("Model",        ["Linear Regression","Ridge","Random Forest","XGBoost"], key=f"{tab_key}_model")
        grp   = st.selectbox("Group column", ["<none>"]+[c for c in ["test_id","module_id","source_file"] if c in feature_df.columns], key=f"{tab_key}_group")

        # -- try auto-load saved model ------------------------------------
        saved = load_model_from_cache(tab_key, tgt, mname)
        if saved is not None and st.session_state.get(f"{tab_key}_model_result") is None:
            st.session_state[f"{tab_key}_model_result"] = saved
            st.info(f"💾 Model loaded from disk cache  ({saved.trained_at})")

        col_btn1, col_btn2 = st.columns([2,1])
        train_clicked = col_btn1.button(f"Train {tab_label} model", key=f"train_{tab_key}")
        force_retrain = col_btn2.checkbox("Force retrain", key=f"force_{tab_key}")

        if train_clicked:
            # honour cache unless force_retrain
            if not force_retrain:
                cached_res = load_model_from_cache(tab_key, tgt, mname)
                if cached_res is not None:
                    st.session_state[f"{tab_key}_model_result"] = cached_res
                    st.success(f"💾 Loaded from disk cache ({cached_res.trained_at}).  Tick 'Force retrain' to override.")
                    return
            with st.spinner(f"Training {mname} on {tgt}..."):
                try:
                    t0  = time.perf_counter()
                    res = train_regression_model(feature_df, tgt, mname, None if grp=="<none>" else grp, exclude_cols=["estimated_cycle_life_band","risk_model_features_used"])
                    st.session_state[f"{tab_key}_model_result"] = res
                    save_model_to_cache(res, tab_key)
                    elapsed = time.perf_counter() - t0
                    gpu_tag = " ⚡ GPU" if res.gpu_used else ""
                    st.success(f"✅ Trained in {elapsed:.2f}s{gpu_tag}  — saved to disk cache.")
                except Exception as exc:
                    st.error(str(exc))

        result: Optional[ModelingResult] = st.session_state.get(f"{tab_key}_model_result")
        if result:
            display_model_metrics(result.metrics, result.cv_scores, result.gpu_used)
            for fig in [
                plot_actual_vs_predicted(result.predictions_df, f"Actual vs Predicted: {tgt}"),
                plot_residuals(result.predictions_df,           f"Residuals: {tgt}"),
                plot_feature_importance(result.feature_importance_df, f"Feature importance: {tgt}"),
            ]:
                if fig: st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------- 3
    with tabs[3]:
        st.subheader("Forecast Temperature / Thermal Factors")
        _train_tab("thermal", targets["thermal"], "thermal")

    # ---------------------------------------------------------------- 4
    with tabs[4]:
        st.subheader("Forecast Current Imbalance")
        _train_tab("current", targets["current"], "current imbalance")

    # ---------------------------------------------------------------- 5
    with tabs[5]:
        st.subheader("SoH / Degradation Risk")
        if feature_df.empty:
            st.warning("Feature table đang rỗng.")
        else:
            c1, c2 = st.columns(2)
            if "degradation_risk_score" in feature_df.columns:
                c1.plotly_chart(plot_risk_gauge(float(feature_df["degradation_risk_score"].mean()), "Average module degradation risk"), use_container_width=True)
            if "relative_lifetime_index" in feature_df.columns:
                c2.plotly_chart(plot_lifetime_index(float(feature_df["relative_lifetime_index"].mean()), "Average relative lifetime index"), use_container_width=True)

    # ---------------------------------------------------------------- 6
    with tabs[6]:
        st.subheader("📡 Graph Network Analysis")
        if prepared.timeseries_df.empty or schema is None or not schema.cell_current_cols:
            st.warning("Cần time-series với cột dòng điện riêng từng cell.")
        else:
            grp_cands = [c for c in ["test_id","module_id","source_file","source_table","synthetic_test_id"] if c in prepared.timeseries_df.columns]
            graph_df = prepared.timeseries_df.copy()
            if grp_cands:
                sc = grp_cands[0]
                sv = st.selectbox("Test condition (graph)", graph_df[sc].astype(str).unique().tolist(), key="graph_case")
                graph_df = graph_df[graph_df[sc].astype(str) == sv].copy()
            ca,cb,cc = st.columns(3)
            ai = ca.slider("α current",0.0,1.0,0.50,0.05,key="g_ai")
            at = cb.slider("α thermal",0.0,1.0,0.30,0.05,key="g_at")
            as_= cc.slider("α SOC",    0.0,1.0,0.20,0.05,key="g_as")
            ni = st.slider("GCN iterations",1,5,2,key="gcn_iter")
            with st.spinner("Building graph..."):
                try:
                    adj,nf,labels = build_battery_graph(graph_df,schema.cell_current_cols,schema.cell_temp_cols or [],schema.time_col,ai,at,as_)
                    nf_gcn = message_passing_aggregate(adj,nf,n_iter=ni)
                    gm = compute_graph_metrics(adj,labels)
                except Exception as exc:
                    st.error(f"Graph error: {exc}"); st.stop()
            st.markdown("#### Node Metrics")
            st.dataframe(gm.set_index("node").style.format("{:.4f}"), use_container_width=True)
            fv = gm["fiedler_value"].iloc[0] if not gm.empty else 0.0
            st.caption(f"Fiedler value: {fv:.4f}")
            for fig,hdr in [(plot_battery_graph(adj,nf,labels,gm),"#### Network Graph"),(plot_adjacency_heatmap(adj,labels),"#### Coupling Matrix"),(plot_gcn_features(nf,nf_gcn,labels),"#### Raw vs GCN Features")]:
                if fig: st.markdown(hdr); st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------- 7
    with tabs[7]:
        st.subheader("🔮 SOH Forecast & RUL")
        if prepared.timeseries_df.empty or schema is None or not schema.cell_current_cols:
            st.warning("Cần time-series với cột dòng điện riêng từng cell.")
        else:
            col1,col2,col3 = st.columns(3)
            soh_thr = col1.slider("EOL threshold (%)",60.0,90.0,80.0,1.0,key="rul_thr")
            horizon = col2.slider("Forecast horizon (cycles)",10,500,100,10,key="rul_horizon")
            method  = col3.selectbox("Forecast method",["ensemble","linear","polynomial"],key="rul_method")
            nom_cap = st.number_input("Nominal capacity (Ah) — 0=auto",0.0,1000.0,0.0,0.1,key="nominal_cap")
            grp_cands=[c for c in ["test_id","module_id","source_file","source_table","synthetic_test_id"] if c in prepared.timeseries_df.columns]
            soh_raw = prepared.timeseries_df.copy()
            if grp_cands:
                sc=grp_cands[0]; sv=st.selectbox("Test condition (SOH)",soh_raw[sc].astype(str).unique().tolist(),key="soh_case")
                soh_raw=soh_raw[soh_raw[sc].astype(str)==sv].copy()

            # auto-load SOH from cache
            soh_fp = fingerprint_dfs(soh_raw)
            if st.session_state.get("soh_hist") is None:
                cached_soh = cache_get("soh_hist", soh_fp)
                if cached_soh is not None:
                    st.session_state["soh_hist"] = cached_soh
                    st.info("💾 SOH history loaded from disk cache.")

            if st.button("Compute SOH & Forecast", key="run_soh"):
                with st.spinner("Estimating SOH per cycle..."):
                    try:
                        cyc_col = next((c for c in ["cycle","cycle_number","cycle_id"] if c in soh_raw.columns),None)
                        soh_h   = estimate_soh_per_cycle(soh_raw,schema.cell_current_cols,schema.time_col,cyc_col,nom_cap if nom_cap>0 else None)
                        st.session_state["soh_hist"]=soh_h
                        cache_set("soh_hist",soh_h,soh_fp,use_parquet=True)
                    except Exception as exc: st.error(f"SOH error: {exc}"); st.stop()
                with st.spinner("Forecasting..."):
                    try:
                        soh_fc=forecast_soh(soh_h,"soh_mean",horizon,method)
                        rul=estimate_rul(soh_h,soh_fc,"soh_mean",soh_thr)
                        st.session_state["soh_forecast"]=soh_fc; st.session_state["rul_info"]=rul
                    except Exception as exc: st.error(f"Forecast error: {exc}"); st.stop()

            soh_h=st.session_state.get("soh_hist"); soh_fc=st.session_state.get("soh_forecast"); rul=st.session_state.get("rul_info")
            if soh_h is not None:
                g1,g2,g3,g4=st.columns(4)
                cs=rul.get("current_soh",np.nan) if rul else np.nan
                g1.metric("Current SOH",f"{cs:.1f}%" if not np.isnan(cs) else "N/A")
                g2.metric("Cycles observed",f"{int(soh_h['cycle'].max())}" if not soh_h.empty else "0")
                g3.metric("Predicted EOL",f"Cycle {rul.get('eol_cycle')}" if rul and rul.get('eol_cycle') else "Beyond horizon")
                g4.metric("RUL (cycles)",f"{int(rul['rul_cycles'])}" if rul and rul.get('rul_cycles') is not None else "—")
                if rul:
                    fg=plot_rul_gauge(rul)
                    if fg: st.plotly_chart(fg,use_container_width=True)
                ff=plot_soh_forecast(soh_h,soh_fc or pd.DataFrame(),schema.cell_current_cols,"soh_mean",rul)
                if ff: st.plotly_chart(ff,use_container_width=True)
                fb=plot_soh_spread(soh_h,schema.cell_current_cols)
                if fb: st.plotly_chart(fb,use_container_width=True)
                with st.expander("Per-cycle SOH table"): st.dataframe(soh_h,use_container_width=True)
                if soh_fc is not None and not soh_fc.empty:
                    comb=pd.concat([soh_h.assign(type="historical"),soh_fc.rename(columns={"soh_forecast":"soh_mean"}).assign(type="forecast")],ignore_index=True)
                    st.download_button("⬇ Download SOH + Forecast CSV",comb.to_csv(index=False).encode(),"soh_forecast.csv","text/csv")
                st.markdown("---"); st.markdown("#### Train ML on SOH")
                if len(soh_h)>=5:
                    sml=st.selectbox("Model",["Linear Regression","Ridge","Random Forest","XGBoost"],key="soh_ml_model")
                    if st.button("Train SOH regression model",key="train_soh"):
                        try:
                            sf=soh_h.copy(); sf["soh_lag1"]=sf["soh_mean"].shift(1); sf["soh_lag2"]=sf["soh_mean"].shift(2); sf["soh_trend"]=sf["soh_mean"].diff(); sf=sf.dropna()
                            if len(sf)>=5:
                                res=train_regression_model(sf,"soh_mean",sml,exclude_cols=["type","soh_min","soh_max","soh_spread"])
                                st.session_state["soh_model_result"]=res
                                save_model_to_cache(res,"soh")
                                display_model_metrics(res.metrics,res.cv_scores,res.gpu_used)
                                for fg in [plot_actual_vs_predicted(res.predictions_df,"SOH: Actual vs Predicted"),plot_feature_importance(res.feature_importance_df,"SOH feature importance")]:
                                    if fg: st.plotly_chart(fg,use_container_width=True)
                            else: st.warning("Không đủ rows.")
                        except Exception as exc: st.error(str(exc))
            else:
                st.info("Bấm **Compute SOH & Forecast** để chạy.")

    # ---------------------------------------------------------------- 8
    with tabs[8]:
        st.subheader("Explainability")
        choice=st.selectbox("Choose trained model",["current_model_result","thermal_model_result","soh_model_result"],key="explain_choice")
        result=st.session_state.get(choice)
        if result is None: st.warning("Hãy train ít nhất một model trước.")
        else:
            fg=plot_feature_importance(result.feature_importance_df,"Top factor ranking")
            if fg: st.plotly_chart(fg,use_container_width=True)
            st.text(auto_explanation_text(result.feature_importance_df,choice))
            st.code(summarize_feature_effects(result.feature_importance_df),language="text")

    # ---------------------------------------------------------------- 9
    with tabs[9]:
        st.subheader("Scenario Simulator")
        if feature_df.empty: st.warning("Feature table đang rỗng.")
        else:
            a,b,c=st.columns(3)
            controls={"operating_temperature":a.slider("Operating temp (°C)",0.0,60.0,25.0,1.0),"interconnection_resistance":b.slider("Interconnection R (mΩ)",0.0,5.0,1.0,0.1),"chemistry":c.selectbox("Chemistry",["NMC","NCA","Mixed"])}
            d,e=st.columns(2); controls["ageing"]=d.selectbox("Ageing status",["unaged","aged"]); controls["ambient_temperature"]=e.slider("Ambient temp (°C)",0.0,60.0,25.0,1.0)
            sc_df=build_scenario_row(feature_df,controls); sc_df=build_risk_scores(sc_df)
            st.dataframe(sc_df,use_container_width=True)
            for rec in rule_based_recommendations(sc_df.iloc[0]): st.write(f"- {rec}")

    # --------------------------------------------------------------- 10
    with tabs[10]:
        st.subheader("Export")
        st.download_button("⬇ Download engineered features CSV", feature_df.to_csv(index=False).encode(),"engineered_features.csv","text/csv")
        report_html=html_report("Parallel Battery Analytics Report",{"Overview":bundle.catalog.to_html(index=False),"Notes":"<br>".join(prepared.notes) if prepared.notes else "No notes."})
        st.download_button("⬇ Download HTML report",report_html.encode(),"parallel_battery_report.html","text/html")


if __name__ == "__main__":
    main()
