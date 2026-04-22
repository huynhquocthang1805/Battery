from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_numeric_distribution(df: pd.DataFrame, col: str, color: str | None = None, title: str | None = None):
    if col not in df.columns:
        return None
    return px.histogram(df, x=col, color=color, marginal="box", nbins=30, title=title or col)


def plot_categorical_distribution(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    counts = df[col].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = [col, "count"]
    return px.bar(counts, x=col, y="count", title=f"Distribution: {col}")


def plot_ocv_curves(df: pd.DataFrame, x_col: str, y_cols: list[str]):
    if x_col not in df.columns:
        return None
    fig = go.Figure()
    for y in y_cols:
        if y in df.columns:
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y], mode="lines", name=y))
    fig.update_layout(title="OCV curves")
    return fig


def plot_timeseries(df: pd.DataFrame, time_col: str, y_cols: list[str], title: str):
    fig = go.Figure()
    for y in y_cols:
        if y in df.columns:
            fig.add_trace(go.Scatter(x=df[time_col], y=df[y], mode="lines", name=y))
    fig.update_layout(title=title, xaxis_title=time_col)
    return fig


def plot_missing_values(df: pd.DataFrame):
    if df.empty:
        return None
    miss = df.isna().mean().sort_values(ascending=False).head(30)
    return px.bar(x=miss.index, y=miss.values, title="Missing value ratio (top 30)")


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str], title: str):
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        return None
    corr = df[cols].corr(numeric_only=True)
    return px.imshow(corr, text_auto=True, title=title)


def plot_actual_vs_predicted(pred_df: pd.DataFrame, title: str):
    if pred_df.empty:
        return None
    return px.scatter(pred_df, x="actual", y="predicted", trendline="ols", title=title)


def plot_residuals(pred_df: pd.DataFrame, title: str):
    if pred_df.empty:
        return None
    tmp = pred_df.copy()
    tmp["residual"] = tmp["actual"] - tmp["predicted"]
    return px.scatter(tmp, x="predicted", y="residual", title=title)


def plot_feature_importance(feat_df: pd.DataFrame, title: str):
    if feat_df is None or feat_df.empty:
        return None
    return px.bar(feat_df.head(20), x="importance", y="feature", orientation="h", title=title)


def plot_risk_gauge(value: float, title: str = "Risk"):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={"text": title}, gauge={"axis": {"range": [0, 100]}}))
    return fig


def plot_lifetime_index(value: float, title: str = "Relative lifetime index"):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={"text": title}, gauge={"axis": {"range": [0, 100]}}))
    return fig


def scenario_comparison_bar(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str):
    if df.empty:
        return None
    y = y_cols[0]
    return px.bar(df, x=x_col, y=y, title=title)
