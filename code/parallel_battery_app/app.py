from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

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
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_load_bundle(path_str: str):
    return load_dataset_bundle(path_str)


@st.cache_data(show_spinner=False)
def cached_prepare_and_engineer(
    timeseries_df: pd.DataFrame, characterization_df: pd.DataFrame
):
    prepared = prepare_data(
        timeseries_df=timeseries_df, characterization_df=characterization_df
    )
    if prepared.timeseries_df.empty:
        return prepared, pd.DataFrame()
    feature_df = build_feature_table_from_timeseries(prepared.timeseries_df)
    if not prepared.characterization_df.empty and not feature_df.empty:
        feature_df = integrate_characterization_features(
            feature_df, prepared.characterization_df
        )
    if not feature_df.empty:
        feature_df = build_risk_scores(feature_df)
    return prepared, feature_df


def ensure_session_key(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


def display_model_metrics(
    metrics: Dict[str, float], cv_scores: Optional[List[float]]
) -> None:
    cols = st.columns(3)
    cols[0].metric("MAE",  f"{metrics.get('MAE',  np.nan):.4f}")
    cols[1].metric("RMSE", f"{metrics.get('RMSE', np.nan):.4f}")
    cols[2].metric("R²",   f"{metrics.get('R2',   np.nan):.4f}")
    if cv_scores:
        st.caption(
            f"Cross-validation RMSE: "
            f"mean={np.mean(cv_scores):.4f}, std={np.std(cv_scores):.4f}"
        )


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


def build_scenario_row(
    feature_df: pd.DataFrame, controls: Dict[str, object]
) -> pd.DataFrame:
    skip_cols = {
        "degradation_risk_score", "relative_lifetime_index",
        "estimated_cycle_life_band", "risk_model_features_used",
    }
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
        "operating_temperature":      ["operating_temperature", "ambient_temperature", "test_temperature"],
        "interconnection_resistance":  ["interconnection_resistance", "branch_resistance"],
        "chemistry":                   ["chemistry"],
        "ageing":                      ["ageing", "aging"],
        "ambient_temperature":         ["ambient_temperature", "operating_temperature"],
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

    for key in [
        "current_model_result", "thermal_model_result", "soh_model_result",
        "bundle", "bundle_error", "dataset_path",
    ]:
        ensure_session_key(key, None if key != "dataset_path" else "")

    # ------------------------------------------------------------------ sidebar
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
            for k in [
                "bundle", "bundle_error", "current_model_result",
                "thermal_model_result", "soh_model_result",
            ]:
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
        st.error("Không tìm thấy bảng dữ liệu hợp lệ.")
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
        use_all_ts   = st.checkbox("Use all detected timeseries tables",      value=True)
        use_all_char = st.checkbox("Use all detected characterization tables", value=True)
        selected_ts_names = (
            ts_options if use_all_ts
            else st.multiselect("Timeseries tables", ts_options, ts_options[: min(10, len(ts_options))])
        )
        selected_char_names = (
            char_options if use_all_char
            else st.multiselect("Characterization tables", char_options, char_options[: min(10, len(char_options))])
        )

    timeseries_tables       = get_loaded_tables_by_names(bundle, selected_ts_names)
    characterization_tables = get_loaded_tables_by_names(bundle, selected_char_names)
    timeseries_df           = concat_tables(timeseries_tables,       add_source_cols=True)
    characterization_df     = concat_tables(characterization_tables, add_source_cols=True)

    try:
        with st.spinner("Đang tiền xử lý và sinh feature..."):
            prepared, feature_df = cached_prepare_and_engineer(
                timeseries_df, characterization_df
            )
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
            st.write("Timeseries shape:",       prepared.timeseries_df.shape)
            st.write("Characterization shape:", prepared.characterization_df.shape)
            st.write("Feature shape:",          feature_df.shape)
            st.write("Feature columns:",        feature_df.columns.tolist())
        fig = plot_missing_values(prepared.timeseries_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------- 1
    with tabs[1]:
        st.subheader("Cell Characterization")
        if prepared.characterization_df.empty:
            st.warning("Chưa có characterization table.")
        else:
            num_cols     = prepared.characterization_df.select_dtypes(include=[np.number]).columns.tolist()
            cap_cols     = [c for c in num_cols if "capacity" in c]
            res_cols     = [c for c in num_cols if any(k in c for k in ["resistance", "r0", "ohmic"])]
            ocv_cols     = [c for c in num_cols if "ocv" in c]
            color_cands  = [c for c in ["ageing", "chemistry"] if c in prepared.characterization_df.columns]
            color_col    = st.selectbox("Color by", ["<none>"] + color_cands, key="char_color")
            color_col    = None if color_col == "<none>" else color_col
            if cap_cols:
                st.plotly_chart(plot_numeric_distribution(prepared.characterization_df, cap_cols[0], color=color_col, title="Capacity distribution"), use_container_width=True)
            if res_cols:
                st.plotly_chart(plot_numeric_distribution(prepared.characterization_df, res_cols[0], color=color_col, title="Resistance distribution"), use_container_width=True)
            if ocv_cols:
                x_cands = [c for c in prepared.characterization_df.columns if c not in ocv_cols][:5]
                if x_cands:
                    st.plotly_chart(plot_ocv_curves(prepared.characterization_df.head(500), x_cands[0], ocv_cols[:6]), use_container_width=True)
            st.dataframe(prepared.characterization_df.head(50), use_container_width=True)

    # ================================================================ 2  UPDATED
    with tabs[2]:
        st.subheader("Current Imbalance Analysis")

        if prepared.timeseries_df.empty or schema is None or not schema.cell_current_cols:
            st.warning("Không có timeseries hoặc không tìm thấy cột dòng điện riêng từng cell.")
        else:
            # --- select test case ---
            grp_cands = [
                c for c in ["test_id", "module_id", "source_file", "source_table", "synthetic_test_id"]
                if c in prepared.timeseries_df.columns
            ]
            case_df = prepared.timeseries_df.copy()
            if grp_cands:
                sel_col = grp_cands[0]
                vals    = case_df[sel_col].astype(str).unique().tolist()
                sel_val = st.selectbox("Select test condition", vals, key="analysis_case")
                case_df = case_df[case_df[sel_col].astype(str) == sel_val].copy()

            # --- controls ---
            st.markdown("#### Delta visualisation controls")
            col_ka, col_kb, col_kc = st.columns(3)
            k_current = col_ka.number_input(
                "Scale factor k (current)", min_value=0.01, max_value=1000.0,
                value=1.0, step=0.1, format="%.3f", key="k_current",
                help="Delta_ij × k  — increase to amplify small differences",
            )
            k_thermal = col_kb.number_input(
                "Scale factor k (thermal)", min_value=0.01, max_value=1000.0,
                value=1.0, step=0.1, format="%.3f", key="k_thermal",
            )
            roll_win = col_kc.number_input(
                "Rolling window (samples)", min_value=2, max_value=5000,
                value=50, step=10, key="roll_win",
            )
            show_abs = st.toggle("Show |delta| (absolute values) for pairwise plot", value=True, key="show_abs")

            # ── Current plots ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("### Current (dòng điện)")

            # 1. Raw signals
            st.plotly_chart(
                plot_timeseries(case_df, schema.time_col, schema.cell_current_cols,
                                "Raw current per cell  [A]"),
                use_container_width=True,
            )

            # 2. Deviation from mean
            st.plotly_chart(
                plot_cell_deviation_from_mean(
                    case_df, schema.time_col, schema.cell_current_cols,
                    title=f"Current deviation from module mean  × {k_current:.3g}  [A]",
                    scale_factor=k_current, unit_label="A",
                ),
                use_container_width=True,
            )

            # 3. Pairwise delta
            st.plotly_chart(
                plot_pairwise_delta(
                    case_df, schema.time_col, schema.cell_current_cols,
                    title=f"Pairwise current delta  × {k_current:.3g}  [A]",
                    scale_factor=k_current, unit_label="A", show_abs=show_abs,
                ),
                use_container_width=True,
            )

            # 4. Rolling imbalance
            st.plotly_chart(
                plot_rolling_imbalance(
                    case_df, schema.time_col, schema.cell_current_cols,
                    window=int(roll_win),
                    title=f"Rolling current imbalance index  × {k_current:.3g}  [A]",
                    scale_factor=k_current, unit_label="A",
                ),
                use_container_width=True,
            )

            # 5. Summary bar
            st.plotly_chart(
                plot_delta_stats_bar(
                    case_df, schema.cell_current_cols,
                    scale_factor=k_current, unit_label="A",
                ),
                use_container_width=True,
            )

            # 6. Full dashboard (4-panel)
            with st.expander("📊 Full imbalance dashboard (4-panel view)", expanded=False):
                st.plotly_chart(
                    plot_imbalance_dashboard(
                        case_df, schema.time_col, schema.cell_current_cols,
                        scale_factor=k_current, window=int(roll_win),
                        unit_label="A", main_title="Current Imbalance Dashboard",
                    ),
                    use_container_width=True,
                )

            # ── Thermal plots ──────────────────────────────────────────
            if schema.cell_temp_cols:
                st.markdown("---")
                st.markdown("### Temperature (nhiệt độ)")

                st.plotly_chart(
                    plot_timeseries(case_df, schema.time_col, schema.cell_temp_cols,
                                    "Raw temperature per cell  [°C]"),
                    use_container_width=True,
                )

                st.plotly_chart(
                    plot_cell_deviation_from_mean(
                        case_df, schema.time_col, schema.cell_temp_cols,
                        title=f"Temperature deviation from mean  × {k_thermal:.3g}  [°C]",
                        scale_factor=k_thermal, unit_label="°C",
                    ),
                    use_container_width=True,
                )

                st.plotly_chart(
                    plot_pairwise_delta(
                        case_df, schema.time_col, schema.cell_temp_cols,
                        title=f"Pairwise temperature delta  × {k_thermal:.3g}  [°C]",
                        scale_factor=k_thermal, unit_label="°C", show_abs=show_abs,
                    ),
                    use_container_width=True,
                )

                st.plotly_chart(
                    plot_rolling_imbalance(
                        case_df, schema.time_col, schema.cell_temp_cols,
                        window=int(roll_win),
                        title=f"Rolling thermal imbalance index  × {k_thermal:.3g}  [°C]",
                        scale_factor=k_thermal, unit_label="°C",
                    ),
                    use_container_width=True,
                )

                st.plotly_chart(
                    plot_delta_stats_bar(
                        case_df, schema.cell_temp_cols,
                        scale_factor=k_thermal, unit_label="°C",
                    ),
                    use_container_width=True,
                )

                with st.expander("🌡️ Full thermal dashboard (4-panel view)", expanded=False):
                    st.plotly_chart(
                        plot_imbalance_dashboard(
                            case_df, schema.time_col, schema.cell_temp_cols,
                            scale_factor=k_thermal, window=int(roll_win),
                            unit_label="°C", main_title="Thermal Imbalance Dashboard",
                        ),
                        use_container_width=True,
                    )

    # ---------------------------------------------------------------- 3
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
                    res = train_regression_model(
                        feature_df, th_tgt, th_model,
                        None if th_grp == "<none>" else th_grp,
                        exclude_cols=["estimated_cycle_life_band", "risk_model_features_used"],
                    )
                    st.session_state["thermal_model_result"] = res
                    st.success("Thermal model trained.")
                except Exception as exc:
                    st.error(str(exc))
            result: ModelingResult | None = st.session_state.get("thermal_model_result")
            if result:
                display_model_metrics(result.metrics, result.cv_scores)
                for fig in [
                    plot_actual_vs_predicted(result.predictions_df, f"Actual vs Predicted: {th_tgt}"),
                    plot_residuals(result.predictions_df,           f"Residuals: {th_tgt}"),
                    plot_feature_importance(result.feature_importance_df, f"Feature importance: {th_tgt}"),
                ]:
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------- 4
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
                    res = train_regression_model(
                        feature_df, cur_tgt, cur_model,
                        None if cur_grp == "<none>" else cur_grp,
                        exclude_cols=["estimated_cycle_life_band", "risk_model_features_used"],
                    )
                    st.session_state["current_model_result"] = res
                    st.success("Imbalance model trained.")
                except Exception as exc:
                    st.error(str(exc))
            result = st.session_state.get("current_model_result")
            if result:
                display_model_metrics(result.metrics, result.cv_scores)
                for fig in [
                    plot_actual_vs_predicted(result.predictions_df, f"Actual vs Predicted: {cur_tgt}"),
                    plot_residuals(result.predictions_df,           f"Residuals: {cur_tgt}"),
                    plot_feature_importance(result.feature_importance_df, f"Feature importance: {cur_tgt}"),
                ]:
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                if st.button("Save current model", key="save_current_model"):
                    mp = Path("saved_model_current.joblib")
                    save_model(result.pipeline, mp)
                    st.success(f"Saved to {mp.resolve()}")

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
        st.subheader("📡 Graph Network Analysis — 4 Cells as Graph Nodes")
        st.markdown(
            "Mỗi **cell** là một **node**. Cạnh được trọng số hoá bằng "
            "**ΔI** · **ΔT** · **ΔSOC**. GCN message-passing tổng hợp đặc trưng hàng xóm."
        )
        if prepared.timeseries_df.empty or schema is None or not schema.cell_current_cols:
            st.warning("Cần time-series với cột dòng điện riêng từng cell.")
        else:
            grp_cands = [c for c in ["test_id", "module_id", "source_file", "source_table", "synthetic_test_id"] if c in prepared.timeseries_df.columns]
            graph_df  = prepared.timeseries_df.copy()
            if grp_cands:
                sc = grp_cands[0]
                sv = st.selectbox("Test condition (graph)", graph_df[sc].astype(str).unique().tolist(), key="graph_case")
                graph_df = graph_df[graph_df[sc].astype(str) == sv].copy()
            ca, cb, cc = st.columns(3)
            ai  = ca.slider("α current", 0.0, 1.0, 0.50, 0.05, key="g_ai")
            at  = cb.slider("α thermal", 0.0, 1.0, 0.30, 0.05, key="g_at")
            as_ = cc.slider("α SOC",     0.0, 1.0, 0.20, 0.05, key="g_as")
            ni  = st.slider("GCN iterations", 1, 5, 2, key="gcn_iter")
            with st.spinner("Building graph..."):
                try:
                    adj, nf, labels = build_battery_graph(
                        graph_df,
                        cell_current_cols=schema.cell_current_cols,
                        cell_temp_cols=schema.cell_temp_cols or [],
                        time_col=schema.time_col,
                        alpha_current=ai, alpha_thermal=at, alpha_soc=as_,
                    )
                    nf_gcn = message_passing_aggregate(adj, nf, n_iter=ni)
                    gm     = compute_graph_metrics(adj, labels)
                except Exception as exc:
                    st.error(f"Graph build error: {exc}"); st.stop()
            st.markdown("#### Node Metrics")
            st.dataframe(gm.set_index("node").style.format("{:.4f}"), use_container_width=True)
            fv = gm["fiedler_value"].iloc[0] if not gm.empty else 0.0
            st.caption(f"Fiedler value: {fv:.4f} — higher = tighter coupling")
            for fig, hdr in [
                (plot_battery_graph(adj, nf, labels, gm), "#### Network Graph"),
                (plot_adjacency_heatmap(adj, labels),     "#### Coupling Similarity Matrix"),
                (plot_gcn_features(nf, nf_gcn, labels),  "#### Raw vs GCN-Aggregated Features"),
            ]:
                if fig:
                    st.markdown(hdr)
                    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------- 7
    with tabs[7]:
        st.subheader("🔮 SOH Forecast & Remaining Useful Life (RUL)")
        if prepared.timeseries_df.empty or schema is None or not schema.cell_current_cols:
            st.warning("Cần time-series với cột dòng điện riêng từng cell.")
        else:
            col1, col2, col3 = st.columns(3)
            soh_thr = col1.slider("EOL threshold (%)", 60.0, 90.0, 80.0, 1.0, key="rul_thr")
            horizon = col2.slider("Forecast horizon (cycles)", 10, 500, 100, 10, key="rul_horizon")
            method  = col3.selectbox("Forecast method", ["ensemble", "linear", "polynomial"], key="rul_method")
            nom_cap = st.number_input("Nominal capacity (Ah) — 0 = auto", 0.0, 1000.0, 0.0, 0.1, key="nominal_cap")
            nom_cap_val = nom_cap if nom_cap > 0 else None
            grp_cands = [c for c in ["test_id", "module_id", "source_file", "source_table", "synthetic_test_id"] if c in prepared.timeseries_df.columns]
            soh_raw = prepared.timeseries_df.copy()
            if grp_cands:
                sc = grp_cands[0]
                sv = st.selectbox("Test condition (SOH)", soh_raw[sc].astype(str).unique().tolist(), key="soh_case")
                soh_raw = soh_raw[soh_raw[sc].astype(str) == sv].copy()
            if st.button("Compute SOH & Forecast", key="run_soh"):
                with st.spinner("Estimating SOH per cycle..."):
                    try:
                        cyc_col = next((c for c in ["cycle", "cycle_number", "cycle_id"] if c in soh_raw.columns), None)
                        soh_h   = estimate_soh_per_cycle(soh_raw, schema.cell_current_cols, schema.time_col, cyc_col, nom_cap_val)
                        st.session_state["soh_hist"] = soh_h
                    except Exception as exc:
                        st.error(f"SOH error: {exc}"); st.stop()
                with st.spinner("Forecasting..."):
                    try:
                        soh_fc = forecast_soh(soh_h, "soh_mean", horizon, method)
                        rul    = estimate_rul(soh_h, soh_fc, "soh_mean", soh_thr)
                        st.session_state["soh_forecast"] = soh_fc
                        st.session_state["rul_info"]     = rul
                    except Exception as exc:
                        st.error(f"Forecast error: {exc}"); st.stop()
            soh_h  = st.session_state.get("soh_hist")
            soh_fc = st.session_state.get("soh_forecast")
            rul    = st.session_state.get("rul_info")
            if soh_h is not None:
                g1, g2, g3, g4 = st.columns(4)
                cs   = rul.get("current_soh", np.nan) if rul else np.nan
                eolc = rul.get("eol_cycle")            if rul else None
                rulc = rul.get("rul_cycles")           if rul else None
                g1.metric("Current SOH",        f"{cs:.1f} %" if not np.isnan(cs) else "N/A")
                g2.metric("Cycles observed",    f"{int(soh_h['cycle'].max())}" if not soh_h.empty else "0")
                g3.metric("Predicted EOL",      f"Cycle {eolc}" if eolc else "Beyond horizon")
                g4.metric("RUL (cycles)",       f"{int(rulc)}" if rulc is not None else "—")
                if rul:
                    fg = plot_rul_gauge(rul)
                    if fg: st.plotly_chart(fg, use_container_width=True)
                ff = plot_soh_forecast(soh_h, soh_fc or pd.DataFrame(), schema.cell_current_cols, "soh_mean", rul)
                if ff: st.plotly_chart(ff, use_container_width=True)
                fb = plot_soh_spread(soh_h, schema.cell_current_cols)
                if fb: st.plotly_chart(fb, use_container_width=True)
                with st.expander("Per-cycle SOH table"):
                    st.dataframe(soh_h, use_container_width=True)
                if soh_fc is not None and not soh_fc.empty:
                    comb = pd.concat([
                        soh_h.assign(type="historical"),
                        soh_fc.rename(columns={"soh_forecast": "soh_mean"}).assign(type="forecast"),
                    ], ignore_index=True)
                    st.download_button("⬇ Download SOH + Forecast CSV", comb.to_csv(index=False).encode(), "soh_forecast.csv", "text/csv")
                st.markdown("---")
                st.markdown("#### Train ML Model on SOH History")
                if len(soh_h) >= 5:
                    sml = st.selectbox("Model", ["Linear Regression", "Ridge", "Random Forest", "XGBoost"], key="soh_ml_model")
                    if st.button("Train SOH regression model", key="train_soh"):
                        try:
                            sf = soh_h.copy()
                            sf["soh_lag1"]  = sf["soh_mean"].shift(1)
                            sf["soh_lag2"]  = sf["soh_mean"].shift(2)
                            sf["soh_trend"] = sf["soh_mean"].diff()
                            sf = sf.dropna()
                            if len(sf) >= 5:
                                res = train_regression_model(sf, "soh_mean", sml, exclude_cols=["type", "soh_min", "soh_max", "soh_spread"])
                                st.session_state["soh_model_result"] = res
                                display_model_metrics(res.metrics, res.cv_scores)
                                for fg in [plot_actual_vs_predicted(res.predictions_df, "SOH: Actual vs Predicted"),
                                           plot_feature_importance(res.feature_importance_df, "SOH feature importance")]:
                                    if fg: st.plotly_chart(fg, use_container_width=True)
                            else:
                                st.warning("Không đủ rows sau khi tạo lag features.")
                        except Exception as exc:
                            st.error(str(exc))
            else:
                st.info("Bấm **Compute SOH & Forecast** để chạy phân tích.")

    # ---------------------------------------------------------------- 8
    with tabs[8]:
        st.subheader("Explainability")
        choice = st.selectbox("Choose trained model", ["current_model_result", "thermal_model_result", "soh_model_result"], key="explain_choice")
        result = st.session_state.get(choice)
        if result is None:
            st.warning("Hãy train ít nhất một model trước.")
        else:
            fg = plot_feature_importance(result.feature_importance_df, "Top factor ranking")
            if fg: st.plotly_chart(fg, use_container_width=True)
            st.text(auto_explanation_text(result.feature_importance_df, choice))
            st.code(summarize_feature_effects(result.feature_importance_df), language="text")

    # ---------------------------------------------------------------- 9
    with tabs[9]:
        st.subheader("Scenario Simulator")
        if feature_df.empty:
            st.warning("Feature table đang rỗng.")
        else:
            a, b, c = st.columns(3)
            controls = {
                "operating_temperature":      a.slider("Operating temperature (°C)", 0.0, 60.0, 25.0, 1.0),
                "interconnection_resistance":  b.slider("Interconnection resistance (mΩ)", 0.0, 5.0, 1.0, 0.1),
                "chemistry":                   c.selectbox("Chemistry", ["NMC", "NCA", "Mixed"]),
            }
            d, e = st.columns(2)
            controls["ageing"]             = d.selectbox("Ageing status", ["unaged", "aged"])
            controls["ambient_temperature"] = e.slider("Ambient temperature (°C)", 0.0, 60.0, 25.0, 1.0)
            sc_df = build_scenario_row(feature_df, controls)
            sc_df = build_risk_scores(sc_df)
            st.dataframe(sc_df, use_container_width=True)
            for rec in rule_based_recommendations(sc_df.iloc[0]):
                st.write(f"- {rec}")

    # --------------------------------------------------------------- 10
    with tabs[10]:
        st.subheader("Export")
        st.download_button(
            "⬇ Download engineered features as CSV",
            feature_df.to_csv(index=False).encode("utf-8"),
            "engineered_features.csv", "text/csv",
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
            report_html.encode("utf-8"),
            "parallel_battery_report.html", "text/html",
        )


if __name__ == "__main__":
    main()
