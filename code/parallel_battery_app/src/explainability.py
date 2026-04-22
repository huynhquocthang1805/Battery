from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
from sklearn.pipeline import Pipeline

from .utils import logger

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None


@dataclass
class ShapArtifacts:
    values: Optional[np.ndarray]
    expected_value: Optional[float]
    feature_names: List[str]
    transformed_X: Optional[np.ndarray]



def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return transformed, feature_names



def compute_shap_artifacts(pipeline: Pipeline, X: pd.DataFrame, max_samples: int = 500) -> ShapArtifacts:
    if shap is None:
        return ShapArtifacts(values=None, expected_value=None, feature_names=[], transformed_X=None)

    X_sample = X.head(max_samples).copy()
    transformed_X, feature_names = transform_features(pipeline, X_sample)
    model = pipeline.named_steps["model"]

    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed_X)
            expected_value = float(np.ravel(explainer.expected_value)[0]) if np.size(explainer.expected_value) else None
        else:
            explainer = shap.Explainer(model.predict, transformed_X)
            explanation = explainer(transformed_X)
            shap_values = explanation.values
            expected_value = float(np.ravel(explanation.base_values)[0]) if np.size(explanation.base_values) else None
        return ShapArtifacts(
            values=np.array(shap_values),
            expected_value=expected_value,
            feature_names=feature_names,
            transformed_X=np.array(transformed_X),
        )
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return ShapArtifacts(values=None, expected_value=None, feature_names=feature_names, transformed_X=None)



def make_shap_summary_figure(artifacts: ShapArtifacts):
    if shap is None or artifacts.values is None or artifacts.transformed_X is None:
        return None
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        artifacts.values,
        artifacts.transformed_X,
        feature_names=artifacts.feature_names,
        show=False,
        plot_size=(10, 6),
    )
    return fig



def make_shap_dependence_figure(artifacts: ShapArtifacts, feature_name: str):
    if shap is None or artifacts.values is None or artifacts.transformed_X is None:
        return None
    if feature_name not in artifacts.feature_names:
        return None
    fig = plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        feature_name,
        artifacts.values,
        artifacts.transformed_X,
        feature_names=artifacts.feature_names,
        show=False,
    )
    return fig



def make_pdp_figure(pipeline: Pipeline, X: pd.DataFrame, feature_name: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        PartialDependenceDisplay.from_estimator(
            pipeline,
            X,
            [feature_name],
            ax=ax,
            kind="average",
            grid_resolution=20,
        )
        return fig
    except Exception as exc:
        logger.warning("PDP failed for %s: %s", feature_name, exc)
        plt.close(fig)
        return None



def summarize_feature_effects(feature_importance_df: pd.DataFrame, top_k: int = 5) -> str:
    if feature_importance_df.empty:
        return "Chưa có feature importance để giải thích."
    top = feature_importance_df.head(top_k)
    bullets = [f"- {row.feature}: importance={row.importance:.4f}" for row in top.itertuples(index=False)]
    return "\n".join(bullets)



def auto_explanation_text(feature_importance_df: pd.DataFrame, target_name: str) -> str:
    if feature_importance_df.empty:
        return f"Chưa đủ thông tin để giải thích yếu tố nào làm tăng {target_name}."
    top_features = feature_importance_df.head(5)["feature"].tolist()
    text = [f"Các yếu tố chi phối mạnh nhất đối với {target_name} hiện tại là: {', '.join(top_features)}."]
    joined = " ".join(top_features).lower()
    if "interconnection" in joined or "resistance" in joined:
        text.append("Điện trở liên kết hoặc nội trở đang là driver quan trọng, phù hợp với kết quả từ các paper tham chiếu.")
    if "temperature" in joined or "temp" in joined:
        text.append("Nhiệt độ/gradient nhiệt đang tác động rõ tới response; cần kiểm soát thermal management tốt hơn.")
    if "chemistry" in joined or "ageing" in joined or "aging" in joined:
        text.append("Sự khác biệt chemistry hoặc aged/unaged mismatch đang đóng góp vào bất cân bằng hiệu năng.")
    return " ".join(text)
