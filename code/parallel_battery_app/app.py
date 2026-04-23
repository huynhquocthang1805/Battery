from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import classify_tables, concat_tables, load_dataset_bundle
from src.explainability import (
    auto_explanation_text,
    compute_shap_artifacts,
    make_pdp_figure,
    make_shap_dependence_figure,
    make_shap_summary_figure,
    summarize_feature_effects,
)
from src.feature_engineering import (
    build_feature_table_from_timeseries,
    build_risk_scores,
    integrate_characterization_features,
)
from src.inference import rule_based_recommendations
from src.modeling import ModelingResult, save_model, train_regression_model
from src.preprocessing import prepare_data
from src.utils import html_report
from src.visualization import (
    plot_actual_vs_predicted,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_lifetime_index,
    plot_missing_values,
    plot_numeric_distribution,
    plot_ocv_curves,
    plot_residuals,
    plot_risk_gauge,
    plot_timeseries,
    scenario_comparison_bar,
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
    cols[0].metric("MAE", f"{metrics.get('MAE', np.nan):.4f}")
    cols[1].metric("RMSE", f"{metrics.get('RMSE', np.nan):.4f}")
    cols[2].metric("R²", f"{metrics.get('R2', np.nan):.4f}")
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
        file_name = t.source_file.lower()
        table_name = t.table_name.lower()

        if table_name == "data" and (file_name.startswith("m1_") or file_name.startswith("m2_")):
            useful.append(t)
            continue

        if {"test_time_s", "current_a", "voltage_v"} <= cols:
            useful.append(t)
            continue

        if any(c.startswith("current_a_cell") for c in cols) or any(
            c.startswith("temperature_c_cell") for c in cols
        ):
            useful.append(t)
            continue

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
        c
        for c in [
            "sigma_i_start",
            "sigma_i_mid",
            "sigma_i_end",
            "delta_soc_max",
            "delta_soc_end",
            "delta_t_max",
            "sigma_t_mean",
            "ttsb",
        ]
        if c in feature_df.columns
    ]
    thermal_targets = [
        c
        for c in [
            "sigma_t_start",
            "sigma_t_mid",
            "sigma_t_end",
            "sigma_t_mean",
            "delta_t_start",
            "delta_t_mid",
            "delta_t_end",
            "delta_t_max",
            "temp_peak",
            "module_temp_gradient_series_auc",
            "ttsb",
        ]
        if c in feature_df.columns
    ]
    soh_targets = [
        c
        for c in feature_df.columns
        if c.lower() in {"soh", "rul", "remaining_useful_life", "capacity_retention"}
    ]
    return {"current": current_targets, "thermal": thermal_targets, "soh": soh_targets}


def build_scenario_row(feature_df: pd.DataFrame, controls: Dict[str, object]) -> pd.DataFrame:
    base: Dict[str, object] = {}
    skip_cols = {
        "degradation_risk_score",
        "relative_lifetime_index",
        "estimated_cycle_life_band",
        "risk_model_features_used",
    }
    for col in feature_df.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            base[col] = float(feature_df[col].median()) if feature_df[col].notna().any() else 0.0
        else:
            mode = feature_df[col].mode(dropna=True)
            base[col] = mode.iloc[0] if not mode.empty else "unknown"

    alias_map = {
        "operating_temperature": ["operating_temperature", "ambient_temperature", "test_temperature"],
        "interconnection_resistance": ["interconnection_resistance", "branch_resistance"],
        "chemistry": ["chemistry"],
        "ageing": ["ageing", "aging"],
        "ambient_temperature": ["ambient_temperature", "operating_temperature"],
    }

    for key, value in controls.items():
        target_cols = alias_map.get(key, [key])
        for col in target_cols:
            if col in base:
                base[col] = value

    return pd.DataFrame([base])


def main():
    st.title("Parallel-Connected Multi-Battery Analytics Dashboard")
    st.caption(
        "Phân tích module song song 4 cell: current imbalance, thermal behavior, SoH proxy/degradation risk, explainability và scenario simulation."
    )

    ensure_session_key("current_model_result", None)
    ensure_session_key("thermal_model_result", None)
    ensure_session_key("soh_model_result", None)
    ensure_session_key("bundle", None)
    ensure_session_key("bundle_error", None)
    ensure_session_key("dataset_path", "")

    with st.sidebar:
        st.header("Dataset Configuration")
        dataset_path = st.text_input(
            "Dataset path",
            value=st.session_state.get("dataset_path", ""),
            help="Trỏ tới thư mục hoặc file CSV/XLSX/MAT thật.",
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
            for key in [
                "bundle",
                "bundle_error",
                "current_model_result",
                "thermal_model_result",
                "soh_model_result",
            ]:
                st.session_state.pop(key, None)
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
    grouped["timeseries"] = filter_useful_timeseries_tables(grouped.get("timeseries", [])) or grouped.get("timeseries", [])
    grouped["characterization"] = filter_useful_characterization_tables(grouped.get("characterization", [])) or grouped.get("characterization", [])

    ts_options = [f"{t.source_file}::{t.table_name}" for t in grouped.get("timeseries", [])]
    char_options = [f"{t.source_file}::{t.table_name}" for t in grouped.get("characterization", [])]

    with st.sidebar:
        st.subheader("Table Selection")
        use_all_ts = st.checkbox("Use all detected timeseries tables", value=True)
        use_all_char = st.checkbox("Use all detected characterization tables", value=True)

        if use_all_ts:
            selected_ts_names = ts_options
        else:
            selected_ts_names = st.multiselect(
                "Timeseries tables",
                options=ts_options,
                default=ts_options[: min(10, len(ts_options))],
            )

        if use_all_char:
            selected_char_names = char_options
        else:
            selected_char_names = st.multiselect(
                "Characterization tables",
                options=char_options,
                default=char_options[: min(10, len(char_options))],
            )

    timeseries_tables = get_loaded_tables_by_names(bundle, selected_ts_names)
    characterization_tables = get_loaded_tables_by_names(bundle, selected_char_names)

    timeseries_df = concat_tables(timeseries_tables, add_source_cols=True)
    characterization_df = concat_tables(characterization_tables, add_source_cols=True)

    try:
        with st.spinner("Đang tiền xử lý và sinh feature..."):
            prepared, feature_df = cached_prepare_and_engineer(timeseries_df, characterization_df)
    except Exception as exc:
        st.error(f"Lỗi preprocessing / feature engineering: {exc}")
        st.stop()

    targets = get_feature_targets(feature_df)

    tabs = st.tabs(
        [
            "Overview",
            "Cell Characterization",
            "Current Imbalance Analysis",
            "Forecast Temperature / Thermal",
            "Forecast Current Imbalance",
            "SoH / Degradation Risk",
            "Explainability",
            "Scenario Simulator",
            "Export",
        ]
    )

    with tabs[0]:
        st.subheader("Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loaded tables", f"{len(bundle.catalog)}")
        c2.metric("Timeseries rows", f"{len(prepared.timeseries_df):,}")
        c3.metric("Characterization rows", f"{len(prepared.characterization_df):,}")
        c4.metric("Engineered feature rows", f"{len(feature_df):,}")

        st.dataframe(bundle.catalog, width="stretch")

        if bundle.errors:
            st.warning("Một số file/tables không parse được hoàn chỉnh.")
            st.write(bundle.errors)

        if prepared.notes:
            st.info("\n".join(prepared.notes))

        st.markdown("**Preview timeseries**")
        st.dataframe(prepared.timeseries_df.head(20), width="stretch")

        with st.expander("Debug summary"):
            st.write("Timeseries shape:", prepared.timeseries_df.shape)
            st.write("Characterization shape:", prepared.characterization_df.shape)
            st.write("Feature shape:", feature_df.shape)
            st.write("Feature columns:", feature_df.columns.tolist())

        fig = plot_missing_values(prepared.timeseries_df)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

    with tabs[1]:
        st.subheader("Cell Characterization")
        if prepared.characterization_df.empty:
            st.warning("Chưa có characterization table.")
        else:
            numeric_cols = prepared.characterization_df.select_dtypes(include=[np.number]).columns.tolist()
            capacity_cols = [c for c in numeric_cols if "capacity" in c]
            resistance_cols = [c for c in numeric_cols if any(k in c for k in ["resistance", "r0", "ohmic"])]
            ocv_cols = [c for c in numeric_cols if "ocv" in c]
            color_candidates = [c for c in ["ageing", "chemistry"] if c in prepared.characterization_df.columns]
            color_col = st.selectbox("Color by", options=["<none>"] + color_candidates, key="char_color")
            color_col = None if color_col == "<none>" else color_col
            if capacity_cols:
                st.plotly_chart(plot_numeric_distribution(prepared.characterization_df, capacity_cols[0], color=color_col, title="Capacity distribution"), width="stretch")
            if resistance_cols:
                st.plotly_chart(plot_numeric_distribution(prepared.characterization_df, resistance_cols[0], color=color_col, title="Resistance distribution"), width="stretch")
            if ocv_cols:
                x_candidates = [c for c in prepared.characterization_df.columns if c not in ocv_cols][:5]
                if x_candidates:
                    st.plotly_chart(plot_ocv_curves(prepared.characterization_df.head(500), x_candidates[0], ocv_cols[:6]), width="stretch")
            st.dataframe(prepared.characterization_df.head(50), width="stretch")

    with tabs[2]:
        st.subheader("Current Imbalance Analysis")
        if prepared.timeseries_df.empty:
            st.warning("Không có timeseries để phân tích.")
        else:
            candidate_group_cols = [c for c in ["test_id", "module_id", "source_file", "source_table", "synthetic_test_id"] if c in prepared.timeseries_df.columns]
            case_df = prepared.timeseries_df.copy()
            if candidate_group_cols:
                selector_col = candidate_group_cols[0]
                vals = case_df[selector_col].astype(str).unique().tolist()
                selected = st.selectbox("Select test condition", options=vals, key="analysis_case")
                case_df = case_df[case_df[selector_col].astype(str) == str(selected)].copy()

            schema = prepared.schema_timeseries
            if schema and schema.time_col and schema.cell_current_cols:
                st.plotly_chart(plot_timeseries(case_df, schema.time_col, schema.cell_current_cols, "Current time-series of each cell"), width="stretch")
                if schema.cell_temp_cols:
                    st.plotly_chart(plot_timeseries(case_df, schema.time_col, schema.cell_temp_cols, "Temperature time-series of each cell"), width="stretch")

    with tabs[3]:
        st.subheader("Forecast Temperature / Thermal Factors")
        if feature_df.empty:
            st.warning("Feature table đang rỗng.")
        elif not targets["thermal"]:
            st.warning("Không đủ thermal target để train model nhiệt độ.")
            st.write("Các cột hiện có:", feature_df.columns.tolist())
        else:
            thermal_target = st.selectbox("Thermal target", options=targets["thermal"], key="thermal_target")
            thermal_model_name = st.selectbox("Model", options=["Linear Regression", "Ridge", "Random Forest", "XGBoost"], key="thermal_model")
            group_col = st.selectbox("Group column for split", options=["<none>"] + [c for c in ["test_id", "module_id", "source_file"] if c in feature_df.columns], key="thermal_group")
            if st.button("Train thermal model", key="train_thermal"):
                try:
                    result = train_regression_model(feature_df, thermal_target, thermal_model_name, None if group_col == "<none>" else group_col, exclude_cols=["estimated_cycle_life_band", "risk_model_features_used"])
                    st.session_state["thermal_model_result"] = result
                    st.success("Thermal model trained successfully.")
                except Exception as exc:
                    st.error(str(exc))

            result: ModelingResult | None = st.session_state.get("thermal_model_result")
            if result is not None:
                display_model_metrics(result.metrics, result.cv_scores)
                for fig in [
                    plot_actual_vs_predicted(result.predictions_df, f"Actual vs Predicted: {thermal_target}"),
                    plot_residuals(result.predictions_df, f"Residuals: {thermal_target}"),
                    plot_feature_importance(result.feature_importance_df, f"Thermal feature importance: {thermal_target}"),
                ]:
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")

    with tabs[4]:
        st.subheader("Forecast Current Imbalance")
        if feature_df.empty:
            st.warning("Feature table đang rỗng.")
        elif not targets["current"]:
            st.warning("Không đủ feature/target để train model imbalance.")
            st.write("Các cột hiện có:", feature_df.columns.tolist())
        else:
            current_target = st.selectbox("Target", options=targets["current"], key="current_target")
            model_name = st.selectbox("Model", options=["Linear Regression", "Ridge", "Random Forest", "XGBoost"], key="current_model")
            group_col = st.selectbox("Group column for split", options=["<none>"] + [c for c in ["test_id", "module_id", "source_file"] if c in feature_df.columns], key="current_group")
            if st.button("Train imbalance model", key="train_current"):
                try:
                    result = train_regression_model(feature_df, current_target, model_name, None if group_col == "<none>" else group_col, exclude_cols=["estimated_cycle_life_band", "risk_model_features_used"])
                    st.session_state["current_model_result"] = result
                    st.success("Imbalance model trained successfully.")
                except Exception as exc:
                    st.error(str(exc))

            result: ModelingResult | None = st.session_state.get("current_model_result")
            if result is not None:
                display_model_metrics(result.metrics, result.cv_scores)
                for fig in [
                    plot_actual_vs_predicted(result.predictions_df, f"Actual vs Predicted: {current_target}"),
                    plot_residuals(result.predictions_df, f"Residuals: {current_target}"),
                    plot_feature_importance(result.feature_importance_df, f"Feature importance: {current_target}"),
                ]:
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
                model_path = Path("saved_model_current.joblib")
                if st.button("Save current model", key="save_current_model"):
                    save_model(result.pipeline, model_path)
                    st.success(f"Saved to {model_path.resolve()}")

    with tabs[5]:
        st.subheader("SoH / Degradation Risk")
        if feature_df.empty:
            st.warning("Feature table đang rỗng.")
        else:
            c1, c2 = st.columns(2)
            if "degradation_risk_score" in feature_df.columns:
                c1.plotly_chart(plot_risk_gauge(float(feature_df["degradation_risk_score"].mean()), "Average module degradation risk"), width="stretch")
            if "relative_lifetime_index" in feature_df.columns:
                c2.plotly_chart(plot_lifetime_index(float(feature_df["relative_lifetime_index"].mean()), "Average relative lifetime index"), width="stretch")

    with tabs[6]:
        st.subheader("Explainability")
        choice = st.selectbox("Choose trained model", options=["current_model_result", "thermal_model_result", "soh_model_result"], key="explain_choice")
        result = st.session_state.get(choice)
        if result is None:
            st.warning("Hãy train ít nhất một model trước.")
        else:
            fig = plot_feature_importance(result.feature_importance_df, "Top factor ranking")
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
            st.text(auto_explanation_text(result.feature_importance_df, choice))
            st.code(summarize_feature_effects(result.feature_importance_df), language="text")

    with tabs[7]:
        st.subheader("Scenario Simulator")
        if feature_df.empty:
            st.warning("Feature table đang rỗng.")
        else:
            a, b, c = st.columns(3)
            controls = {
                "operating_temperature": a.slider("Operating temperature (°C)", 0.0, 60.0, 25.0, 1.0),
                "interconnection_resistance": b.slider("Interconnection resistance (mΩ)", 0.0, 5.0, 1.0, 0.1),
                "chemistry": c.selectbox("Chemistry", ["NMC", "NCA", "Mixed"]),
            }
            d, e = st.columns(2)
            controls["ageing"] = d.selectbox("Ageing status", ["unaged", "aged"])
            controls["ambient_temperature"] = e.slider("Ambient temperature (°C)", 0.0, 60.0, 25.0, 1.0)

            scenario_df = build_scenario_row(feature_df, controls)
            scenario_df = build_risk_scores(scenario_df)
            st.dataframe(scenario_df, width="stretch")

            for rec in rule_based_recommendations(scenario_df.iloc[0]):
                st.write(f"- {rec}")

    with tabs[8]:
        st.subheader("Export")
        st.download_button("Download engineered features as CSV", data=feature_df.to_csv(index=False).encode("utf-8"), file_name="engineered_features.csv", mime="text/csv")
        report_html = html_report(
            "Parallel Battery Analytics Report",
            {
                "Overview": bundle.catalog.to_html(index=False),
                "Notes": "<br>".join(prepared.notes) if prepared.notes else "No notes.",
            },
        )
        st.download_button("Download HTML report", data=report_html.encode("utf-8"), file_name="parallel_battery_report.html", mime="text/html")


if __name__ == "__main__":
    main()
