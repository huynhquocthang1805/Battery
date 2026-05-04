from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import classify_tables, concat_tables, load_dataset_bundle
from src.explainability import (
    auto_explanation_text,
    summarize_feature_effects,
)
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
from src.modeling import ModelingResult, save_model, train_regression_model
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
    plot_feature_importance,
    plot_lifetime_index,
    plot_missing_values,
    plot_numeric_distribution,
    plot_ocv_curves,
    plot_residuals,
    plot_risk_gauge,
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
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_load_bundle(path_str: str):
    return load_dataset_bundle(path_str)


@st.cache_data(show_spinner=False)
def cached_prepare_and_engineer(timeseries_df: pd.DataFrame, characterization_df: pd.DataFrame):
    prepared = prepare_data(timeseries_df=timeseries_df, characterization_df=characterization_df)
    if prepared.timeseries_df.empty:
        return prepared, pd.DataFrame()
    feature_df = build_feature_table_from_timeseries(prepared.timeseries_df)
    if not prepared.characterization_df.empty and not feature_df.empty:
        feature_df = integrate_characterization_features(feature_df, prepared.characterization_df)
    if not feature_df.empty:
        feature_df = build_risk_scores(feature_df)
    return prepared, feature_df


def ensure_session_key(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


def display_model_metrics(metrics: Dict[str, float], cv_scores: Optional[List[float]]) -> None:
    cols = st.columns(3)
    cols[0].metric("MAE",  f"{metrics.get('MAE',  np.nan):.4f}")
    cols[1].metric("RMSE", f"{metrics.get('RMSE', np.nan):.4f}")
    cols[2].metric("R²",   f"{metrics.get('R2',   np.nan):.4f}")
    if cv_scores:
        st.caption(
            f"Cross-validation RMSE: mean={np.mean(cv_scores):.4f}, std={np.std(cv_scores):.4f}"
        )


def get_loaded_tables_by_names(bundle, names: List[str]):
    out = []
    names_set = set(names)
    for table in bundle.tables:
        full_name = f"{table.source_file}::{table.table_name}"
        if full_name in names_set:
            out.append(table)
    return out


def filter_useful_timeseries_tables(tables):
    useful = []
    for t in tables:
        cols = {str(c).lower() for c in t.df.columns}
        file_name  = t.source_file.lower()
        table_name = t.table_name.lower()
        if table_name == "data" and (file_name.startswith("m1_") or file_name.startswith("m2_")):
            useful.append(t); continue
        if {"test_time_s", "current_a", "voltage_v"} <= cols:
            useful.append(t); continue
        if any(c.startswith("current_a_cell") for c in cols) or any(
            c.startswith("temperature_c_cell") for c in cols
        ):
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
    current_targets = [
        c for c in [
            "sigma_i_start", "sigma_i_mid", "sigma_i_end",
            "delta_soc_max", "delta_soc_end", "delta_t_max", "sigma_t_mean", "ttsb",
        ] if c in feature_df.columns
    ]
    thermal_targets = [
        c for c in [
            "sigma_t_start", "sigma_t_mid", "sigma_t_end", "sigma_t_mean",
            "delta_t_start", "delta_t_mid", "delta_t_end", "delta_t_max",
            "temp_peak", "module_temp_gradient_series_auc", "ttsb",
        ] if c in feature_df.columns
    ]
    soh_targets = [
        c for c in feature_df.columns
        if c.lower() in {"soh", "rul", "remaining_useful_life", "capacity_retention"}
    ]
    return {"current": current_targets, "thermal": thermal_targets, "soh": soh_targets}


def build_scenario_row(feature_df: pd.DataFrame, controls: Dict[str, object]) -> pd.DataFrame:
    skip_cols = {"degradation_risk_score", "relative_lifetime_index",
                 "estimated_cycle_life_band", "risk_model_features_used"}
    base: Dict[str, object] = {}
    for col in feature_df.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            base[col] = float(feature_df[col].median()) if feature_df[col].notna().any() else 0.0
        else:
            mode = feature_df[col].mode(dropna=True)
            base[col] = mode.iloc[0] if not mode.empty else "unknown"
    alias_map = {
        "operating_temperature":    ["operating_temperature", "ambient_temperature", "test_temperature"],
        "interconnection_resistance": ["interconnection_resistance", "branch_resistance"],
        "chemistry":                ["chemistry"],
        "ageing":                   ["ageing", "aging"],
        "ambient_temperature":      ["ambient_temperature", "operating_temperature"],
    }
    for key, value in controls.items():
        for col in alias_map.get(key, [key]):
            if col in base:
                base[col] = value
    return pd.DataFrame([base])


# ============================================================
# MAIN
# ============================================================

def main():
    st.title("Parallel-Connected Multi-Battery Analytics Dashboard")
    st.caption(
        "Phân tích module song song 4 cell — current imbalance · thermal · "
        "Graph Network Analysis (GCN) · SOH Forecast & RUL · Explainability · Scenario Simulation"
    )

    for key in ["current_model_result", "thermal_model_result", "soh_model_result",
                "bundle", "bundle_error", "dataset_path"]:
        ensure_session_key(key, None if key != "dataset_path" else "")

    # ---- Sidebar --------------------------------------------------------
    with st.sidebar:
        st.header("Dataset Configuration")
        dataset_path = st.text_input(
            "Dataset path",
            value=st.session_state.get("dataset_path", ""),
            help="Trỏ tới thư mục hoặc file CSV/XLSX/MAT.",
        )
        if st.button("Load dataset", type="primary"):
            st.session_state["dataset_path"] = dataset_path
            st.session_state["bundle_error"] = None
            try:
                with st.spinner("Đang đọc dữ liệu..."):
                    st.session_state["bundle"] = cached_load_bundle(dataset_path)
            except Exception as exc:
                st.session_state["bundle"] = None
                st.session_state["bundle_error"] = str(exc)

        if st.button("Clear Cache"):
            st.cache_data.clear()
            for k in ["bundle", "bundle_error", "current_model_result",
                      "thermal_model_result", "soh_model_result"]:
                st.session_state.pop(k, None)
            st.rerun()

    if st.session_state.get("bundle_error"):
        st.error(f"Không thể load dataset: {st.session_state['bundle_error']}")
        return

    bundle = st.session_state.get("bundle")
    if bundle is None:
        st.info("Nhập dataset path ở sidebar và bấm **Load dataset** để bắt đầu.")
        return

    if bundle.catalog.empty:
        st.error("Không tìm thấy bảng dữ liệu hợp lệ trong path đã cung cấp.")
        if bundle.errors:
            st.write(bundle.errors)
        return

    grouped = classify_tables(bundle)
    grouped["timeseries"] = (
        filter_useful_timeseries_tables(grouped.get("timeseries", []))
        or grouped.get("timeseries", [])
    )
    grouped["characterization"] = (
        filter_useful_characterization_tables(grouped.get("characterization", []))
        or grouped.get("characterization", [])
    )

    ts_options   = [f"{t.source_file}::{t.table_name}" for t in grouped.get("timeseries", [])]
    char_options = [f"{t.source_file}::{t.table_name}" for t in grouped.get("characterization", [])]

    with st.sidebar:
        st.subheader("Table Selection")
        use_all_ts   = st.checkbox("Use all detected timeseries tables",       value=True)
        use_all_char = st.checkbox("Use all detected characterization tables",  value=True)
        selected_ts_names   = ts_options   if use_all_ts   else st.multiselect("Timeseries tables",       ts_options,   ts_options[:min(10, len(ts_options))])
        selected_char_names = char_options if use_all_char else st.multiselect("Characterization tables", char_options, char_options[:min(10, len(char_options))])

    timeseries_tables      = get_loaded_tables_by_names(bundle, selected_ts_names)
    characterization_tables = get_loaded_tables_by_names(bundle, selected_char_names)
    timeseries_df          = concat_tables(timeseries_tables,      add_source_cols=True)
    characterization_df    = concat_tables(characterization_tables, add_source_cols=True)

    try:
        with st.spinner("Đang tiền xử lý và sinh feature..."):
            prepared, feature_df = cached_prepare_and_engineer(timeseries_df, characterization_df)
    except Exception as exc:
        st.error(f"Lỗi preprocessing / feature engineering: {exc}")
        st.stop()

    targets = get_feature_targets(feature_df)
    schema  = prepared.schema_timeseries

    # ====================================================================
    # TABS
    # ====================================================================
    tabs = st.tabs([
        "📋 Overview",
        "🔬 Cell Characterization",
        "⚡ Current Imbalance",
        "🌡️ Forecast Thermal",
        "📈 Forecast Imbalance",
        "🩺 SoH / Risk",
        "📡 Graph Network Analysis",       # NEW
        "🔮 SOH Forecast & RUL",            # NEW
        "🔍 Explainability",
        "🎯 Scenario Simulator",
        "📤 Export",
    ])

    # ------------------------------------------------------------------ 0
    with tabs[0]:
        st.subheader("Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loaded tables",           f"{len(bundle.catalog)}")
        c2.metric("Timeseries rows",         f"{len(prepared.timeseries_df):,}")
        c3.metric("Characterization rows",   f"{len(prepared.characterization_df):,}")
        c4.metric("Engineered feature rows", f"{len(feature_df):,}")
        st.dataframe(bundle.catalog, use_container_width=True)
        if bundle.errors:
            st.warning("Một số file/tables không parse được hoàn chỉnh.")
            st.write(bundle.errors)
        if prepared.notes:
            st.info("\n".join(prepared.notes))
        st.markdown("**Preview timeseries**")
        st.dataframe(prepared.timeseries_df.head(20), use_container_width=True)
        with st.expander("Debug summary"):
            st.write("Timeseries shape:",      prepared.timeseries_df.shape)
            st.write("Characterization shape:", prepared.characterization_df.shape)
            st.write("Feature shape:",          feature_df.shape)
            st.write("Feature columns:",        feature_df.columns.tolist())
        fig = plot_missing_values(prepared.timeseries_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------ 1
    with tabs[1]:
        st.subheader("Cell Characterization")
        if prepared.characterization_df.empty:
            st.warning("Chưa có characterization table.")
        else:
            numeric_cols     = prepared.characterization_df.select_dtypes(include=[np.number]).columns.tolist()
            capacity_cols    = [c for c in numeric_cols if "capacity" in c]
            resistance_cols  = [c for c in numeric_cols if any(k in c for k in ["resistance", "r0", "ohmic"])]
            ocv_cols         = [c for c in numeric_cols if "ocv" in c]
            color_candidates = [c for c in ["ageing", "chemistry"] if c in prepared.characterization_df.columns]
            color_col = st.selectbox("Color by", ["<none>"] + color_candidates, key="char_color")
            color_col = None if color_col == "<none>" else color_col
            if capacity_cols:
                st.plotly_chart(plot_numeric_distribution(prepared.characterization_df, capacity_cols[0], color=color_col, title="Capacity distribution"), use_container_width=True)
            if resistance_cols:
                st.plotly_chart(plot_numeric_distribution(prepared.characterization_df, resistance_cols[0], color=color_col, title="Resistance distribution"), use_container_width=True)
            if ocv_cols:
                x_cands = [c for c in prepared.characterization_df.columns if c not in ocv_cols][:5]
                if x_cands:
                    st.plotly_chart(plot_ocv_curves(prepared.characterization_df.head(500), x_cands[0], ocv_cols[:6]), use_container_width=True)
            st.dataframe(prepared.characterization_df.head(50), use_container_width=True)

    # ------------------------------------------------------------------ 2
    with tabs[2]:
        st.subheader("Current Imbalance Analysis")
        if prepared.timeseries_df.empty:
            st.warning("Không có timeseries để phân tích.")
        else:
            grp_candidates = [c for c in ["test_id", "module_id", "source_file", "source_table", "synthetic_test_id"] if c in prepared.timeseries_df.columns]
            case_df = prepared.timeseries_df.copy()
            if grp_candidates:
                sel_col = grp_candidates[0]
                vals    = case_df[sel_col].astype(str).unique().tolist()
                sel_val = st.selectbox("Select test condition", vals, key="analysis_case")
                case_df = case_df[case_df[sel_col].astype(str) == sel_val].copy()
            if schema and schema.time_col and schema.cell_current_cols:
                st.plotly_chart(plot_timeseries(case_df, schema.time_col, schema.cell_current_cols, "Current time-series per cell"), use_container_width=True)
                if schema.cell_temp_cols:
                    st.plotly_chart(plot_timeseries(case_df, schema.time_col, schema.cell_temp_cols, "Temperature time-series per cell"), use_container_width=True)

    # ------------------------------------------------------------------ 3
    with tabs[3]:
        st.subheader("Forecast Temperature / Thermal Factors")
        if feature_df.empty:
            st.warning("Feature table đang rỗng.")
        elif not targets["thermal"]:
            st.warning("Không đủ thermal target.")
            st.write(feature_df.columns.tolist())
        else:
            th_tgt   = st.selectbox("Thermal target", targets["thermal"], key="thermal_target")
            th_model = st.selectbox("Model", ["Linear Regression", "Ridge", "Random Forest", "XGBoost"], key="thermal_model")
            th_grp   = st.selectbox("Group column", ["<none>"] + [c for c in ["test_id", "module_id", "source_file"] if c in feature_df.columns], key="thermal_group")
            if st.button("Train thermal model", key="train_thermal"):
                try:
                    res = train_regression_model(feature_df, th_tgt, th_model, None if th_grp == "<none>" else th_grp, exclude_cols=["estimated_cycle_life_band", "risk_model_features_used"])
                    st.session_state["thermal_model_result"] = res
                    st.success("Thermal model trained.")
                except Exception as exc:
                    st.error(str(exc))
            result: ModelingResult | None = st.session_state.get("thermal_model_result")
            if result:
                display_model_metrics(result.metrics, result.cv_scores)
                for fig in [plot_actual_vs_predicted(result.predictions_df, f"Actual vs Predicted: {th_tgt}"),
                            plot_residuals(result.predictions_df, f"Residuals: {th_tgt}"),
                            plot_feature_importance(result.feature_importance_df, f"Feature importance: {th_tgt}")]:
                    if fig: st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------ 4
    with tabs[4]:
        st.subheader("Forecast Current Imbalance")
        if feature_df.empty:
            st.warning("Feature table đang rỗng.")
        elif not targets["current"]:
            st.warning("Không đủ feature/target để train model imbalance.")
            st.write(feature_df.columns.tolist())
        else:
            cur_tgt   = st.selectbox("Target",  targets["current"], key="current_target")
            cur_model = st.selectbox("Model",   ["Linear Regression", "Ridge", "Random Forest", "XGBoost"], key="current_model")
            cur_grp   = st.selectbox("Group column", ["<none>"] + [c for c in ["test_id", "module_id", "source_file"] if c in feature_df.columns], key="current_group")
            if st.button("Train imbalance model", key="train_current"):
                try:
                    res = train_regression_model(feature_df, cur_tgt, cur_model, None if cur_grp == "<none>" else cur_grp, exclude_cols=["estimated_cycle_life_band", "risk_model_features_used"])
                    st.session_state["current_model_result"] = res
                    st.success("Imbalance model trained.")
                except Exception as exc:
                    st.error(str(exc))
            result = st.session_state.get("current_model_result")
            if result:
                display_model_metrics(result.metrics, result.cv_scores)
                for fig in [plot_actual_vs_predicted(result.predictions_df, f"Actual vs Predicted: {cur_tgt}"),
                            plot_residuals(result.predictions_df, f"Residuals: {cur_tgt}"),
                            plot_feature_importance(result.feature_importance_df, f"Feature importance: {cur_tgt}")]:
                    if fig: st.plotly_chart(fig, use_container_width=True)
                if st.button("Save current model", key="save_current_model"):
                    mp = Path("saved_model_current.joblib")
                    save_model(result.pipeline, mp)
                    st.success(f"Saved to {mp.resolve()}")

    # ------------------------------------------------------------------ 5
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

    # ================================================================== 6  NEW
    with tabs[6]:
        st.subheader("📡 Graph Network Analysis — 4 Cells as Graph Nodes")
        st.markdown(
            "Mỗi **cell** là một **node**. Các **cạnh** (edges) được trọng số hóa "
            "bằng tổ hợp **ΔI** (dòng điện), **ΔT** (nhiệt độ) và **ΔSOC** giữa các cặp pin. "
            "GCN message-passing tổng hợp đặc trưng hàng xóm vào mỗi node."
        )

        if prepared.timeseries_df.empty or schema is None or not schema.cell_current_cols:
            st.warning("Cần có dữ liệu time-series với ít nhất các cột dòng điện riêng từng cell.")
        else:
            # --- Select test case ----------------------------------------
            grp_candidates = [c for c in ["test_id", "module_id", "source_file", "source_table", "synthetic_test_id"]
                              if c in prepared.timeseries_df.columns]
            graph_df = prepared.timeseries_df.copy()
            if grp_candidates:
                sel_col = grp_candidates[0]
                vals    = graph_df[sel_col].astype(str).unique().tolist()
                sel_val = st.selectbox("Select test condition for graph", vals, key="graph_case")
                graph_df = graph_df[graph_df[sel_col].astype(str) == sel_val].copy()

            # --- Hyperparameters -----------------------------------------
            col_a, col_b, col_c = st.columns(3)
            alpha_i   = col_a.slider("α current (ΔI weight)",   0.0, 1.0, 0.50, 0.05, key="g_ai")
            alpha_t   = col_b.slider("α thermal (ΔT weight)",   0.0, 1.0, 0.30, 0.05, key="g_at")
            alpha_soc = col_c.slider("α SOC (ΔSOC weight)",     0.0, 1.0, 0.20, 0.05, key="g_as")
            n_iter    = st.slider("GCN message-passing iterations", 1, 5, 2, key="gcn_iter")

            with st.spinner("Đang xây dựng đồ thị pin..."):
                try:
                    adj, node_feat, labels = build_battery_graph(
                        graph_df,
                        cell_current_cols=schema.cell_current_cols,
                        cell_temp_cols=schema.cell_temp_cols or [],
                        time_col=schema.time_col,
                        alpha_current=alpha_i,
                        alpha_thermal=alpha_t,
                        alpha_soc=alpha_soc,
                    )
                    node_feat_gcn = message_passing_aggregate(adj, node_feat, n_iter=n_iter)
                    graph_metrics = compute_graph_metrics(adj, labels)
                except Exception as exc:
                    st.error(f"Graph build error: {exc}")
                    st.stop()

            # --- Graph metrics table -------------------------------------
            st.markdown("#### Graph Node Metrics")
            st.dataframe(graph_metrics.set_index("node").style.format("{:.4f}"), use_container_width=True)

            fiedler = graph_metrics["fiedler_value"].iloc[0] if not graph_metrics.empty else 0.0
            st.caption(
                f"**Fiedler value (algebraic connectivity):** {fiedler:.4f} — "
                "giá trị cao hơn → mạng pin kết nối chặt chẽ hơn, "
                "suy hao lan truyền nhanh hơn."
            )

            # --- Network graph -------------------------------------------
            st.markdown("#### Battery Module Network Graph")
            fig_graph = plot_battery_graph(adj, node_feat, labels, graph_metrics)
            if fig_graph:
                st.plotly_chart(fig_graph, use_container_width=True)

            # --- Adjacency heatmap ---------------------------------------
            st.markdown("#### Cell-to-Cell Coupling Similarity")
            fig_heat = plot_adjacency_heatmap(adj, labels)
            if fig_heat:
                st.plotly_chart(fig_heat, use_container_width=True)

            # --- GCN features comparison ---------------------------------
            st.markdown("#### Raw vs GCN-Aggregated Node Features")
            fig_gcn = plot_gcn_features(node_feat, node_feat_gcn, labels)
            if fig_gcn:
                st.plotly_chart(fig_gcn, use_container_width=True)

            st.info(
                "💡 **Diễn giải:** Node có **Degradation Propagation Risk** cao "
                "là cell có xu hướng bị suy hao và kéo theo các cell lân cận. "
                "Sau GCN aggregation, đặc trưng node đã tổng hợp thêm thông tin "
                "từ các hàng xóm — đây là đầu vào phong phú hơn cho mô hình SOH."
            )

    # ================================================================== 7  NEW
    with tabs[7]:
        st.subheader("🔮 SOH Forecast & Remaining Useful Life (RUL)")
        st.markdown(
            "SOH được ước lượng theo từng **cycle** từ dung lượng phóng Coulomb-counted. "
            "Forecast sử dụng **ensemble** (Linear + Polynomial + Exponential Decay). "
            "RUL = số cycle đến khi SOH < ngưỡng EOL."
        )

        if prepared.timeseries_df.empty or schema is None or not schema.cell_current_cols:
            st.warning("Cần time-series với cột dòng điện riêng từng cell.")
        else:
            # --- Controls ------------------------------------------------
            col1, col2, col3 = st.columns(3)
            soh_threshold = col1.slider("EOL threshold (%)", 60.0, 90.0, 80.0, 1.0, key="rul_thr")
            horizon       = col2.slider("Forecast horizon (cycles)", 10, 500, 100, 10, key="rul_horizon")
            method        = col3.selectbox("Forecast method", ["ensemble", "linear", "polynomial"], key="rul_method")
            nominal_cap   = st.number_input("Nominal capacity (Ah) — 0 = auto-detect", 0.0, 1000.0, 0.0, 0.1, key="nominal_cap")
            nominal_cap_val = nominal_cap if nominal_cap > 0 else None

            # --- Select test case ----------------------------------------
            grp_candidates = [c for c in ["test_id", "module_id", "source_file", "source_table", "synthetic_test_id"]
                              if c in prepared.timeseries_df.columns]
            soh_df_raw = prepared.timeseries_df.copy()
            if grp_candidates:
                sel_col = grp_candidates[0]
                vals    = soh_df_raw[sel_col].astype(str).unique().tolist()
                sel_val = st.selectbox("Select test condition for SOH", vals, key="soh_case")
                soh_df_raw = soh_df_raw[soh_df_raw[sel_col].astype(str) == sel_val].copy()

            if st.button("Compute SOH & Forecast", key="run_soh"):
                with st.spinner("Đang ước lượng SOH từng cycle..."):
                    try:
                        cycle_col_cand = next((c for c in ["cycle", "cycle_number", "cycle_id"] if c in soh_df_raw.columns), None)
                        soh_hist = estimate_soh_per_cycle(
                            soh_df_raw,
                            cell_current_cols=schema.cell_current_cols,
                            time_col=schema.time_col,
                            cycle_col=cycle_col_cand,
                            nominal_capacity_ah=nominal_cap_val,
                        )
                        st.session_state["soh_hist"] = soh_hist
                    except Exception as exc:
                        st.error(f"SOH estimation error: {exc}")
                        st.stop()

                with st.spinner("Đang forecast SOH..."):
                    try:
                        soh_fc = forecast_soh(soh_hist, target_col="soh_mean", horizon=horizon, method=method)
                        rul    = estimate_rul(soh_hist, soh_fc, target_col="soh_mean", threshold=soh_threshold)
                        st.session_state["soh_forecast"] = soh_fc
                        st.session_state["rul_info"]     = rul
                    except Exception as exc:
                        st.error(f"Forecast error: {exc}")
                        st.stop()

            soh_hist = st.session_state.get("soh_hist")
            soh_fc   = st.session_state.get("soh_forecast")
            rul_info = st.session_state.get("rul_info")

            if soh_hist is not None:
                # RUL gauge + metrics
                g1, g2, g3, g4 = st.columns(4)
                cur_soh = rul_info.get("current_soh", np.nan) if rul_info else np.nan
                eol_cyc = rul_info.get("eol_cycle")          if rul_info else None
                rul_cyc = rul_info.get("rul_cycles")          if rul_info else None
                g1.metric("Current SOH", f"{cur_soh:.1f} %" if not np.isnan(cur_soh) else "N/A")
                g2.metric("Cycles observed", f"{int(soh_hist['cycle'].max())}" if not soh_hist.empty else "0")
                g3.metric("Predicted EOL cycle", f"Cycle {eol_cyc}" if eol_cyc else "Beyond horizon")
                g4.metric("RUL (cycles)", f"{int(rul_cyc)}" if rul_cyc is not None else "—")

                if rul_info:
                    fig_gauge = plot_rul_gauge(rul_info)
                    if fig_gauge:
                        st.plotly_chart(fig_gauge, use_container_width=True)

                # Main forecast chart
                fig_fc = plot_soh_forecast(
                    soh_hist, soh_fc or pd.DataFrame(),
                    cell_cols=schema.cell_current_cols,
                    target_col="soh_mean",
                    rul_info=rul_info,
                )
                if fig_fc:
                    st.plotly_chart(fig_fc, use_container_width=True)

                # SOH spread boxplot
                fig_box = plot_soh_spread(soh_hist, schema.cell_current_cols)
                if fig_box:
                    st.plotly_chart(fig_box, use_container_width=True)

                # Per-cycle table
                with st.expander("Per-cycle SOH table"):
                    st.dataframe(soh_hist, use_container_width=True)

                # Download
                if soh_fc is not None and not soh_fc.empty:
                    combined = pd.concat([soh_hist.assign(type="historical"),
                                          soh_fc.rename(columns={"soh_forecast": "soh_mean"}).assign(type="forecast")],
                                         ignore_index=True)
                    st.download_button(
                        "⬇ Download SOH + Forecast CSV",
                        data=combined.to_csv(index=False).encode("utf-8"),
                        file_name="soh_forecast.csv", mime="text/csv",
                    )

                # Train ML model on SOH (optional)
                st.markdown("---")
                st.markdown("#### Train ML Model on SOH History")
                if len(soh_hist) >= 5:
                    soh_ml_model = st.selectbox("Model", ["Linear Regression", "Ridge", "Random Forest", "XGBoost"], key="soh_ml_model")
                    if st.button("Train SOH regression model", key="train_soh"):
                        try:
                            soh_feat = soh_hist.copy()
                            # Add lag features
                            soh_feat["soh_lag1"]  = soh_feat["soh_mean"].shift(1)
                            soh_feat["soh_lag2"]  = soh_feat["soh_mean"].shift(2)
                            soh_feat["soh_trend"] = soh_feat["soh_mean"].diff()
                            soh_feat = soh_feat.dropna()
                            if len(soh_feat) >= 5:
                                res = train_regression_model(soh_feat, "soh_mean", soh_ml_model, exclude_cols=["type", "soh_min", "soh_max", "soh_spread"])
                                st.session_state["soh_model_result"] = res
                                display_model_metrics(res.metrics, res.cv_scores)
                                fig_ap = plot_actual_vs_predicted(res.predictions_df, "SOH: Actual vs Predicted")
                                if fig_ap: st.plotly_chart(fig_ap, use_container_width=True)
                                fig_fi = plot_feature_importance(res.feature_importance_df, "SOH feature importance")
                                if fig_fi: st.plotly_chart(fig_fi, use_container_width=True)
                            else:
                                st.warning("Không đủ rows sau khi tạo lag features.")
                        except Exception as exc:
                            st.error(str(exc))
            else:
                st.info("Bấm **Compute SOH & Forecast** để chạy phân tích.")

    # ------------------------------------------------------------------ 8
    with tabs[8]:
        st.subheader("Explainability")
        choice = st.selectbox("Choose trained model",
                              ["current_model_result", "thermal_model_result", "soh_model_result"],
                              key="explain_choice")
        result = st.session_state.get(choice)
        if result is None:
            st.warning("Hãy train ít nhất một model trước.")
        else:
            fig = plot_feature_importance(result.feature_importance_df, "Top factor ranking")
            if fig: st.plotly_chart(fig, use_container_width=True)
            st.text(auto_explanation_text(result.feature_importance_df, choice))
            st.code(summarize_feature_effects(result.feature_importance_df), language="text")

    # ------------------------------------------------------------------ 9
    with tabs[9]:
        st.subheader("Scenario Simulator")
        if feature_df.empty:
            st.warning("Feature table đang rỗng.")
        else:
            a, b, c = st.columns(3)
            controls = {
                "operating_temperature":    a.slider("Operating temperature (°C)", 0.0, 60.0, 25.0, 1.0),
                "interconnection_resistance": b.slider("Interconnection resistance (mΩ)", 0.0, 5.0, 1.0, 0.1),
                "chemistry":                c.selectbox("Chemistry", ["NMC", "NCA", "Mixed"]),
            }
            d, e = st.columns(2)
            controls["ageing"]             = d.selectbox("Ageing status", ["unaged", "aged"])
            controls["ambient_temperature"] = e.slider("Ambient temperature (°C)", 0.0, 60.0, 25.0, 1.0)
            scenario_df = build_scenario_row(feature_df, controls)
            scenario_df = build_risk_scores(scenario_df)
            st.dataframe(scenario_df, use_container_width=True)
            for rec in rule_based_recommendations(scenario_df.iloc[0]):
                st.write(f"- {rec}")

    # ----------------------------------------------------------------- 10
    with tabs[10]:
        st.subheader("Export")
        st.download_button(
            "⬇ Download engineered features as CSV",
            data=feature_df.to_csv(index=False).encode("utf-8"),
            file_name="engineered_features.csv", mime="text/csv",
        )
        report_html = html_report(
            "Parallel Battery Analytics Report",
            {
                "Overview": bundle.catalog.to_html(index=False),
                "Notes":    "<br>".join(prepared.notes) if prepared.notes else "No notes.",
            },
        )
        st.download_button(
            "⬇ Download HTML report",
            data=report_html.encode("utf-8"),
            file_name="parallel_battery_report.html", mime="text/html",
        )


if __name__ == "__main__":
    main()
