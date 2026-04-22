from __future__ import annotations

import io
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
from src.feature_engineering import build_feature_table_from_timeseries, build_risk_scores, integrate_characterization_features
from src.inference import rule_based_recommendations, scenario_dataframe_from_controls
from src.modeling import ModelingResult, save_model, train_regression_model
from src.preprocessing import prepare_data
from src.utils import html_report, missing_summary
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
        .block-container { padding-top: 1.0rem; }
        h1, h2, h3 { color: #e5e7eb; }
        .metric-card { background: rgba(17, 24, 39, 0.8); padding: 12px; border-radius: 10px; }
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
    feature_df = build_feature_table_from_timeseries(prepared.timeseries_df)
    feature_df = integrate_characterization_features(feature_df, prepared.characterization_df)
    feature_df = build_risk_scores(feature_df)
    return prepared, feature_df


@st.cache_data(show_spinner=False)
def cached_png_bytes(fig):
    buffer = io.BytesIO()
    fig.write_image(buffer, format="png")
    return buffer.getvalue()



def display_model_metrics(metrics: Dict[str, float], cv_scores: Optional[List[float]]) -> None:
    cols = st.columns(3)
    cols[0].metric("MAE", f"{metrics.get('MAE', np.nan):.4f}")
    cols[1].metric("RMSE", f"{metrics.get('RMSE', np.nan):.4f}")
    cols[2].metric("R²", f"{metrics.get('R2', np.nan):.4f}")
    if cv_scores:
        st.caption(f"Cross-validation RMSE: mean={np.mean(cv_scores):.4f}, std={np.std(cv_scores):.4f}")



def candidate_table_names(bundle) -> List[str]:
    return [f"{t.source_file}::{t.table_name}" for t in bundle.tables]



def get_table_by_name(bundle, name: str) -> pd.DataFrame:
    for table in bundle.tables:
        full_name = f"{table.source_file}::{table.table_name}"
        if full_name == name:
            return table.df.copy()
    return pd.DataFrame()



def infer_default_table_selection(bundle):
    grouped = classify_tables(bundle)
    ts_default = f"{grouped['timeseries'][0].source_file}::{grouped['timeseries'][0].table_name}" if grouped.get("timeseries") else None
    char_default = f"{grouped['characterization'][0].source_file}::{grouped['characterization'][0].table_name}" if grouped.get("characterization") else None
    return ts_default, char_default



def ensure_session_key(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default



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
    soh_targets = [c for c in feature_df.columns if c.lower() in {"soh", "rul", "remaining_useful_life", "capacity_retention"}]
    return {"current": current_targets, "thermal": thermal_targets, "soh": soh_targets}



def build_scenario_row(feature_df: pd.DataFrame, controls: Dict[str, object]) -> pd.DataFrame:
    base: Dict[str, object] = {}
    for col in feature_df.columns:
        if col in {"degradation_risk_score", "relative_lifetime_index", "estimated_cycle_life_band", "risk_model_features_used"}:
            continue
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            base[col] = float(feature_df[col].median()) if feature_df[col].notna().any() else 0.0
        else:
            mode = feature_df[col].mode(dropna=True)
            base[col] = mode.iloc[0] if not mode.empty else "unknown"

    name_map = {
        "operating_temperature": ["operating_temperature", "ambient_temperature", "test_temperature"],
        "interconnection_resistance": ["interconnection_resistance", "branch_resistance"],
        "chemistry": ["chemistry"],
        "ageing": ["ageing", "aging"],
        "capacity_dispersion": ["capacity_std", "char_capacity_mean_std", "global_char_capacity_mean"],
        "resistance_dispersion": ["resistance_std", "temp_cells_std", "current_cells_std"],
    }
    for control_name, control_value in controls.items():
        if control_name in name_map:
            for col in name_map[control_name]:
                if col in base:
                    base[col] = control_value
    return pd.DataFrame([base])



def main():
    st.title("Parallel-Connected Multi-Battery Analytics Dashboard")
    st.caption(
        "Phân tích module song song 4 cell: current imbalance, thermal behavior, SoH proxy / degradation risk, explainability và scenario simulation."
    )

    ensure_session_key("current_model_result", None)
    ensure_session_key("thermal_model_result", None)
    ensure_session_key("soh_model_result", None)

    with st.sidebar:
        st.header("Dataset Configuration")
        dataset_path = st.text_input(
            "Dataset path",
            value="/path/to/parallel_battery_dataset",
            help="Trỏ tới thư mục hoặc file CSV/XLSX/MAT thật của anh.",
        )
        load_clicked = st.button("Load dataset", type="primary")
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.caption("App ưu tiên dữ liệu thật. Nếu path sai hoặc thiếu cột, UI sẽ báo rõ lỗi.")

    if not load_clicked:
        st.info("Nhập dataset path ở sidebar và bấm **Load dataset** để bắt đầu.")
        return

    try:
        with st.spinner("Đang đọc dữ liệu..."):
            bundle = cached_load_bundle(dataset_path)
    except Exception as exc:
        st.error(f"Không thể load dataset: {exc}")
        return

    if bundle.catalog.empty:
        st.error("Không tìm thấy bảng dữ liệu hợp lệ trong path đã cung cấp.")
        if bundle.errors:
            st.write(bundle.errors)
        return

    ts_default, char_default = infer_default_table_selection(bundle)
    table_names = candidate_table_names(bundle)

    with st.sidebar:
        st.subheader("Table Selection")
        selected_ts_name = st.selectbox(
            "Timeseries table",
            options=table_names,
            index=table_names.index(ts_default) if ts_default in table_names else 0,
        )
        selected_char_name = st.selectbox(
            "Characterization table (optional)",
            options=["<none>"] + table_names,
            index=(table_names.index(char_default) + 1) if char_default in table_names else 0,
        )

    timeseries_df = get_table_by_name(bundle, selected_ts_name)
    characterization_df = pd.DataFrame() if selected_char_name == "<none>" else get_table_by_name(bundle, selected_char_name)

    with st.spinner("Đang tiền xử lý và sinh feature..."):
        prepared, feature_df = cached_prepare_and_engineer(timeseries_df, characterization_df)

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

    # Tab 1
    with tabs[0]:
        st.subheader("Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loaded files/tables", f"{len(bundle.catalog)}")
        c2.metric("Timeseries rows", f"{len(prepared.timeseries_df):,}")
        c3.metric("Characterization rows", f"{len(prepared.characterization_df):,}")
        c4.metric("Engineered feature rows", f"{len(feature_df):,}")

        st.markdown("**Catalog**")
        st.dataframe(bundle.catalog, width="stretch")

        if bundle.errors:
            st.warning("Một số file không đọc được:")
            st.write(bundle.errors)
        if prepared.notes:
            st.info("\n".join(prepared.notes))

        st.markdown("**Preview timeseries**")
        st.dataframe(prepared.timeseries_df.head(20), width="stretch")

        missing_fig = plot_missing_values(prepared.timeseries_df)
        if missing_fig is not None:
            st.plotly_chart(missing_fig, width="stretch")

        dist_cols = st.columns(4)
        for idx, col_name in enumerate(["chemistry", "ageing", "operating_temperature", "interconnection_resistance"]):
            if col_name in feature_df.columns:
                fig = plot_categorical_distribution(feature_df, col_name) if feature_df[col_name].dtype == object else plot_numeric_distribution(feature_df, col_name)
                if fig is not None:
                    dist_cols[idx].plotly_chart(fig, width="stretch")

    # Tab 2
    with tabs[1]:
        st.subheader("Cell Characterization")
        if prepared.characterization_df.empty:
            st.warning("Chưa có bảng characterization. Vẫn có thể phân tích từ timeseries nhưng phần này sẽ bị hạn chế.")
        else:
            numeric_cols = prepared.characterization_df.select_dtypes(include=[np.number]).columns.tolist()
            capacity_cols = [c for c in numeric_cols if "capacity" in c]
            resistance_cols = [c for c in numeric_cols if ("resistance" in c or "r0" in c or "ohmic" in c)]
            ocv_cols = [c for c in numeric_cols if "ocv" in c]
            color_candidates = [c for c in ["ageing", "chemistry"] if c in prepared.characterization_df.columns]
            color_col = st.selectbox("Color by", options=["<none>"] + color_candidates, index=0)
            color_col = None if color_col == "<none>" else color_col

            if capacity_cols:
                chosen_capacity = st.selectbox("Capacity column", options=capacity_cols)
                st.plotly_chart(
                    plot_numeric_distribution(prepared.characterization_df, chosen_capacity, color=color_col, title="Capacity distribution"),
                    width="stretch",
                )
            if resistance_cols:
                chosen_res = st.selectbox("Resistance column", options=resistance_cols)
                st.plotly_chart(
                    plot_numeric_distribution(prepared.characterization_df, chosen_res, color=color_col, title="Resistance distribution"),
                    width="stretch",
                )
            if ocv_cols:
                x_candidates = [c for c in prepared.characterization_df.columns if c not in ocv_cols][:10]
                if x_candidates:
                    x_col = st.selectbox("OCV x-axis", options=x_candidates)
                    st.plotly_chart(plot_ocv_curves(prepared.characterization_df.head(500), x_col, ocv_cols[:6]), width="stretch")
            st.dataframe(prepared.characterization_df.head(50), width="stretch")

    # Tab 3
    with tabs[2]:
        st.subheader("Current Imbalance Analysis")
        if prepared.timeseries_df.empty:
            st.warning("Không có timeseries để phân tích.")
        else:
            candidate_group_cols = [c for c in ["test_id", "module_id", "source_file", "source_table", "synthetic_test_id"] if c in prepared.timeseries_df.columns]
            if candidate_group_cols:
                selector_col = candidate_group_cols[0]
                group_values = prepared.timeseries_df[selector_col].astype(str).unique().tolist()
                selected_group = st.selectbox("Select test condition", options=group_values)
                case_df = prepared.timeseries_df[prepared.timeseries_df[selector_col].astype(str) == str(selected_group)].copy()
            else:
                case_df = prepared.timeseries_df.copy()

            schema = prepared.schema_timeseries
            if schema and schema.time_col and schema.cell_current_cols:
                st.plotly_chart(
                    plot_timeseries(case_df, schema.time_col, schema.cell_current_cols, "Current time-series of each cell"),
                    width="stretch",
                )
                if schema.cell_temp_cols:
                    st.plotly_chart(
                        plot_timeseries(case_df, schema.time_col, schema.cell_temp_cols, "Temperature time-series of each cell"),
                        width="stretch",
                    )

            metrics_to_show = [c for c in ["sigma_i_start", "sigma_i_mid", "sigma_i_end", "current_spread_mean", "delta_soc_max", "delta_t_max", "sigma_t_mean", "ttsb"] if c in feature_df.columns]
            st.dataframe(feature_df[metrics_to_show].head(20), width="stretch")
            heatmap_cols = metrics_to_show + [c for c in ["operating_temperature", "interconnection_resistance"] if c in feature_df.columns]
            corr_fig = plot_correlation_heatmap(feature_df, columns=heatmap_cols, title="Feature/imbalance correlation")
            if corr_fig is not None:
                st.plotly_chart(corr_fig, width="stretch")

    # Tab 4
    with tabs[3]:
        st.subheader("Forecast Temperature / Thermal Factors")
        st.caption("Theo paper thực nghiệm, operating temperature và interconnection resistance có ảnh hưởng phi tuyến rõ hơn đối với thermal responses và TTSB.")
        if feature_df.empty or not targets["thermal"]:
            st.warning("Không đủ thermal target để train model nhiệt độ.")
        else:
            thermal_target = st.selectbox("Thermal target", options=targets["thermal"])
            thermal_model_name = st.selectbox("Model", options=["Linear Regression", "Ridge", "Random Forest", "XGBoost"], key="thermal_model_select")
            thermal_group_col = st.selectbox(
                "Group column for split",
                options=["<none>"] + [c for c in ["test_id", "module_id", "source_file"] if c in feature_df.columns],
                key="thermal_group_select",
            )
            if st.button("Train thermal model", key="train_thermal"):
                try:
                    with st.spinner("Training thermal model..."):
                        result = train_regression_model(
                            feature_df,
                            target_col=thermal_target,
                            model_name=thermal_model_name,
                            group_col=None if thermal_group_col == "<none>" else thermal_group_col,
                            exclude_cols=["estimated_cycle_life_band", "risk_model_features_used"],
                        )
                    st.session_state["thermal_model_result"] = result
                    st.success("Thermal model trained successfully.")
                except Exception as exc:
                    st.error(f"Thermal training failed: {exc}")

            result: ModelingResult | None = st.session_state.get("thermal_model_result")
            if result is not None:
                display_model_metrics(result.metrics, result.cv_scores)
                fig1 = plot_actual_vs_predicted(result.predictions_df, title=f"Actual vs Predicted: {thermal_target}")
                fig2 = plot_residuals(result.predictions_df, title=f"Residuals: {thermal_target}")
                fig3 = plot_feature_importance(result.feature_importance_df, title=f"Thermal feature importance: {thermal_target}")
                col_a, col_b = st.columns(2)
                if fig1 is not None:
                    col_a.plotly_chart(fig1, width="stretch")
                if fig2 is not None:
                    col_b.plotly_chart(fig2, width="stretch")
                if fig3 is not None:
                    st.plotly_chart(fig3, width="stretch")
                st.markdown(auto_explanation_text(result.feature_importance_df, thermal_target))

    # Tab 5
    with tabs[4]:
        st.subheader("Forecast Current Imbalance")
        if feature_df.empty or not targets["current"]:
            st.warning("Không đủ feature/target để train model imbalance.")
        else:
            current_target = st.selectbox("Target", options=targets["current"])
            model_name = st.selectbox("Model", options=["Linear Regression", "Ridge", "Random Forest", "XGBoost"], key="current_model_select")
            group_col = st.selectbox(
                "Group column for split",
                options=["<none>"] + [c for c in ["test_id", "module_id", "source_file"] if c in feature_df.columns],
                key="current_group_select",
            )
            if st.button("Train imbalance model", key="train_current"):
                try:
                    with st.spinner("Training imbalance model..."):
                        result = train_regression_model(
                            feature_df,
                            target_col=current_target,
                            model_name=model_name,
                            group_col=None if group_col == "<none>" else group_col,
                            exclude_cols=["estimated_cycle_life_band", "risk_model_features_used"],
                        )
                    st.session_state["current_model_result"] = result
                    st.success("Imbalance model trained successfully.")
                except Exception as exc:
                    st.error(f"Imbalance training failed: {exc}")

            result: ModelingResult | None = st.session_state.get("current_model_result")
            if result is not None:
                display_model_metrics(result.metrics, result.cv_scores)
                fig1 = plot_actual_vs_predicted(result.predictions_df, title=f"Actual vs Predicted: {current_target}")
                fig2 = plot_residuals(result.predictions_df, title=f"Residuals: {current_target}")
                fig3 = plot_feature_importance(result.feature_importance_df, title=f"Feature importance: {current_target}")
                col_a, col_b = st.columns(2)
                if fig1 is not None:
                    col_a.plotly_chart(fig1, width="stretch")
                if fig2 is not None:
                    col_b.plotly_chart(fig2, width="stretch")
                if fig3 is not None:
                    st.plotly_chart(fig3, width="stretch")
                model_path = Path("saved_model_current.joblib")
                if st.button("Save current model", key="save_current_model"):
                    save_model(result.pipeline, model_path)
                    st.success(f"Saved to {model_path.resolve()}")

    # Tab 6
    with tabs[5]:
        st.subheader("SoH / Degradation Risk")
        st.info(
            "Nếu dataset không có long-term SoH/RUL label đầy đủ, app chạy ở chế độ proxy-based: xây degradation risk score minh bạch và lifetime index tương đối."
        )
        if feature_df.empty:
            st.warning("Feature table đang rỗng.")
        else:
            c1, c2 = st.columns(2)
            module_risk = float(feature_df["degradation_risk_score"].mean()) if "degradation_risk_score" in feature_df.columns else np.nan
            lifetime_index = float(feature_df["relative_lifetime_index"].mean()) if "relative_lifetime_index" in feature_df.columns else np.nan
            if np.isfinite(module_risk):
                c1.plotly_chart(plot_risk_gauge(module_risk, title="Average module degradation risk"), width="stretch")
            if np.isfinite(lifetime_index):
                c2.plotly_chart(plot_lifetime_index(lifetime_index, title="Average relative lifetime index"), width="stretch")
            st.dataframe(
                feature_df[[c for c in ["degradation_risk_score", "relative_lifetime_index", "estimated_cycle_life_band", "risk_model_features_used"] if c in feature_df.columns]].head(50),
                width="stretch",
            )

            if targets["soh"]:
                st.markdown("### Supervised SoH/RUL regression")
                soh_target = st.selectbox("SoH/RUL label", options=targets["soh"])
                soh_model_name = st.selectbox("Model", options=["Linear Regression", "Ridge", "Random Forest", "XGBoost"], key="soh_model_select")
                if st.button("Train SoH/RUL model", key="train_soh"):
                    try:
                        result = train_regression_model(
                            feature_df,
                            target_col=soh_target,
                            model_name=soh_model_name,
                            group_col="test_id" if "test_id" in feature_df.columns else None,
                            exclude_cols=["estimated_cycle_life_band", "risk_model_features_used"],
                        )
                        st.session_state["soh_model_result"] = result
                        st.success("SoH/RUL model trained successfully.")
                    except Exception as exc:
                        st.error(f"SoH/RUL training failed: {exc}")
                result = st.session_state.get("soh_model_result")
                if result is not None:
                    display_model_metrics(result.metrics, result.cv_scores)
                    fig = plot_actual_vs_predicted(result.predictions_df, title=f"Actual vs Predicted: {soh_target}")
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
            else:
                st.caption("Không phát hiện nhãn SoH/RUL tuyệt đối. UI đang dùng risk score và relative lifetime index.")

    # Tab 7
    with tabs[6]:
        st.subheader("Explainability")
        result_for_explain = st.session_state.get("current_model_result") or st.session_state.get("thermal_model_result") or st.session_state.get("soh_model_result")
        if result_for_explain is None:
            st.warning("Hãy train ít nhất một model trước khi vào explainability.")
        else:
            chosen_target = st.selectbox("Explain model for target", options=["current_model_result", "thermal_model_result", "soh_model_result"])
            result_map = {
                "current_model_result": st.session_state.get("current_model_result"),
                "thermal_model_result": st.session_state.get("thermal_model_result"),
                "soh_model_result": st.session_state.get("soh_model_result"),
            }
            result: ModelingResult | None = result_map.get(chosen_target)
            if result is not None:
                st.plotly_chart(plot_feature_importance(result.feature_importance_df, title="Top factor ranking"), width="stretch")
                st.text(auto_explanation_text(result.feature_importance_df, chosen_target))
                st.markdown("**Feature importance summary**")
                st.code(summarize_feature_effects(result.feature_importance_df), language="text")

                explain_features = feature_df.drop(columns=[c for c in ["estimated_cycle_life_band", "risk_model_features_used"] if c in feature_df.columns], errors="ignore")
                current_target_col = None
                if not result.predictions_df.empty:
                    # Approximate the target based on session context.
                    pass
                # SHAP on feature table without explicit target exclusion may fail if labels are present; keep numeric/cat features only.
                if st.checkbox("Compute SHAP", value=False):
                    try:
                        target_guess = None
                        for target_list in targets.values():
                            if st.session_state.get("current_model_result") is result and target_list:
                                target_guess = None
                        # Use all columns except obvious outputs.
                        shap_input = feature_df.drop(
                            columns=[
                                c
                                for c in [
                                    "degradation_risk_score",
                                    "relative_lifetime_index",
                                    "estimated_cycle_life_band",
                                    "risk_model_features_used",
                                ]
                                if c in feature_df.columns
                            ],
                            errors="ignore",
                        )
                        shap_artifacts = compute_shap_artifacts(result.pipeline, shap_input)
                        shap_fig = make_shap_summary_figure(shap_artifacts)
                        if shap_fig is not None:
                            st.pyplot(shap_fig, clear_figure=True)
                        if shap_artifacts.feature_names:
                            selected_feature = st.selectbox("SHAP/PDP feature", options=shap_artifacts.feature_names)
                            dep_fig = make_shap_dependence_figure(shap_artifacts, selected_feature)
                            if dep_fig is not None:
                                st.pyplot(dep_fig, clear_figure=True)
                            pdp_fig = make_pdp_figure(result.pipeline, shap_input, selected_feature)
                            if pdp_fig is not None:
                                st.pyplot(pdp_fig, clear_figure=True)
                    except Exception as exc:
                        st.error(f"Explainability computation failed: {exc}")

    # Tab 8
    with tabs[7]:
        st.subheader("Scenario Simulator")
        if feature_df.empty:
            st.warning("Feature table đang rỗng.")
        else:
            col1, col2, col3 = st.columns(3)
            temperature = col1.slider("Operating temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=1.0)
            interconnection_res = col2.slider("Interconnection resistance (mΩ)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            chemistry = col3.selectbox("Chemistry", options=["NMC", "NCA", "Mix"])
            col4, col5, col6 = st.columns(3)
            ageing = col4.selectbox("Ageing status", options=["unaged", "aged"])
            capacity_dispersion = col5.slider("Capacity dispersion proxy", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
            resistance_dispersion = col6.slider("Resistance dispersion proxy", min_value=0.0, max_value=1.0, value=0.15, step=0.01)

            controls = {
                "operating_temperature": temperature,
                "interconnection_resistance": interconnection_res,
                "chemistry": chemistry,
                "ageing": ageing,
                "capacity_dispersion": capacity_dispersion,
                "resistance_dispersion": resistance_dispersion,
            }
            scenario_df = build_scenario_row(feature_df, controls)
            scenario_df = build_risk_scores(scenario_df)
            st.dataframe(scenario_df, width="stretch")

            current_result = st.session_state.get("current_model_result")
            thermal_result = st.session_state.get("thermal_model_result")
            prediction_rows = []
            if current_result is not None:
                try:
                    pred = current_result.pipeline.predict(scenario_df[current_result.pipeline.feature_names_in_]) if hasattr(current_result.pipeline, "feature_names_in_") else current_result.pipeline.predict(scenario_df)
                    prediction_rows.append({"scenario": "current_target", "prediction": float(np.ravel(pred)[0])})
                except Exception:
                    pass
            if thermal_result is not None:
                try:
                    pred = thermal_result.pipeline.predict(scenario_df[thermal_result.pipeline.feature_names_in_]) if hasattr(thermal_result.pipeline, "feature_names_in_") else thermal_result.pipeline.predict(scenario_df)
                    prediction_rows.append({"scenario": "thermal_target", "prediction": float(np.ravel(pred)[0])})
                except Exception:
                    pass
            if prediction_rows:
                pred_df = pd.DataFrame(prediction_rows)
                st.plotly_chart(scenario_comparison_bar(pred_df, x_col="scenario", y_cols=["prediction"], title="Scenario predictions"), width="stretch")

            risk_score = float(scenario_df["degradation_risk_score"].iloc[0]) if "degradation_risk_score" in scenario_df.columns else np.nan
            lifetime_idx = float(scenario_df["relative_lifetime_index"].iloc[0]) if "relative_lifetime_index" in scenario_df.columns else np.nan
            c1, c2 = st.columns(2)
            if np.isfinite(risk_score):
                c1.plotly_chart(plot_risk_gauge(risk_score), width="stretch")
            if np.isfinite(lifetime_idx):
                c2.plotly_chart(plot_lifetime_index(lifetime_idx), width="stretch")

            st.markdown("### Recommendation")
            for rec in rule_based_recommendations(scenario_df.iloc[0]):
                st.write(f"- {rec}")
            if np.isfinite(risk_score) and risk_score >= 70:
                st.error("Scenario này có degradation risk cao.")
            elif np.isfinite(risk_score) and risk_score >= 40:
                st.warning("Scenario này có degradation risk trung bình.")
            else:
                st.success("Scenario này tương đối an toàn theo proxy hiện tại.")

    # Tab 9
    with tabs[8]:
        st.subheader("Export")
        st.download_button(
            "Download engineered features as CSV",
            data=feature_df.to_csv(index=False).encode("utf-8"),
            file_name="engineered_features.csv",
            mime="text/csv",
        )
        report_html = html_report(
            "Parallel Battery Analytics Report",
            {
                "Overview": bundle.catalog.to_html(index=False),
                "Notes": "<br>".join(prepared.notes) if prepared.notes else "No notes.",
                "Risk": feature_df[[c for c in ["degradation_risk_score", "relative_lifetime_index", "estimated_cycle_life_band"] if c in feature_df.columns]].head(20).to_html(index=False),
            },
        )
        st.download_button(
            "Download HTML report",
            data=report_html.encode("utf-8"),
            file_name="parallel_battery_report.html",
            mime="text/html",
        )
        st.caption("PNG export cần kaleido. Nếu môi trường thiếu kaleido, hãy dùng export HTML hoặc CSV.")


if __name__ == "__main__":
    main()
