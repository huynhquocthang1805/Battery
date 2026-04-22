from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
except Exception:
    shap = None

from sklearn.inspection import PartialDependenceDisplay


@dataclass
class ShapArtifacts:
    values: Optional[np.ndarray]
    data: pd.DataFrame
    feature_names: List[str]


def auto_explanation_text(feature_importance_df: pd.DataFrame, target: str) -> str:
    if feature_importance_df is None or feature_importance_df.empty:
        return f"No explainability summary available for {target}."
    top = feature_importance_df.head(5)["feature"].tolist()
    return f"Top drivers for {target}: " + ", ".join(top)


def summarize_feature_effects(feature_importance_df: pd.DataFrame) -> str:
    if feature_importance_df.empty:
        return "No feature importance available."
    lines = [f"- {r.feature}: {r.importance:.4f}" for r in feature_importance_df.head(10).itertuples()]
    return "\n".join(lines)


def compute_shap_artifacts(pipeline, X: pd.DataFrame) -> ShapArtifacts:
    if shap is None:
        raise RuntimeError("shap is not installed")
    sample = X.head(min(200, len(X))).copy()
    prep = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_proc = prep.transform(sample)
    feature_names = list(prep.get_feature_names_out())
    explainer = shap.Explainer(model, X_proc)
    values = explainer(X_proc).values
    return ShapArtifacts(values=values, data=pd.DataFrame(X_proc, columns=feature_names), feature_names=feature_names)


def make_shap_summary_figure(artifacts: ShapArtifacts):
    if artifacts.values is None:
        return None
    fig = plt.figure(figsize=(8, 5))
    shap.summary_plot(artifacts.values, artifacts.data, feature_names=artifacts.feature_names, show=False)
    return fig


def make_shap_dependence_figure(artifacts: ShapArtifacts, feature: str):
    if artifacts.values is None or feature not in artifacts.feature_names:
        return None
    fig = plt.figure(figsize=(7, 5))
    shap.dependence_plot(feature, artifacts.values, artifacts.data, feature_names=artifacts.feature_names, show=False)
    return fig


def make_pdp_figure(pipeline, X: pd.DataFrame, feature: str):
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        PartialDependenceDisplay.from_estimator(pipeline, X.head(min(500, len(X))), [feature], ax=ax)
        return fig
    except Exception:
        return None
