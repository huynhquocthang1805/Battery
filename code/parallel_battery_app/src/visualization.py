from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .utils import infer_schema, logger, missing_summary


PLOTLY_TEMPLATE = "plotly_dark"



def preview_table(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return df.head(n)



def plot_missing_values(df: pd.DataFrame):
    summary = missing_summary(df)
    if summary.empty:
        return None
    summary = summary.head(30)
    fig = px.bar(
        summary,
        x="column",
        y="missing_pct",
        title="Missing values (%)",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig



def plot_categorical_distribution(df: pd.DataFrame, column: str, title: Optional[str] = None):
    if column not in df.columns:
        return None
    counts = df[column].astype(str).fillna("NA").value_counts(dropna=False).reset_index()
    counts.columns = [column, "count"]
    return px.bar(counts, x=column, y="count", title=title or f"Distribution: {column}", template=PLOTLY_TEMPLATE)



def plot_numeric_distribution(df: pd.DataFrame, column: str, color: Optional[str] = None, title: Optional[str] = None):
    if column not in df.columns:
        return None
    return px.histogram(
        df,
        x=column,
        color=color if color in df.columns else None,
        marginal="box",
        nbins=40,
        title=title or f"Distribution: {column}",
        template=PLOTLY_TEMPLATE,
    )



def plot_ocv_curves(df: pd.DataFrame, x_col: str, y_cols: List[str], color_col: Optional[str] = None):
    if df.empty or x_col not in df.columns or not y_cols:
        return None
    fig = go.Figure()
    color_values = df[color_col].astype(str) if color_col and color_col in df.columns else None
    for col in y_cols:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(width=2),
            )
        )
    fig.update_layout(title="OCV-related curves", template=PLOTLY_TEMPLATE, xaxis_title=x_col, yaxis_title="Voltage / OCV")
    return fig



def plot_timeseries(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str):
    if df.empty or x_col not in df.columns or not y_cols:
        return None
    fig = go.Figure()
    for col in y_cols:
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(x=df[x_col], y=df[col], mode="lines", name=col))
    fig.update_layout(template=PLOTLY_TEMPLATE, title=title, xaxis_title=x_col)
    return fig



def plot_correlation_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None, title: str = "Correlation heatmap"):
    num_df = df.select_dtypes(include=[np.number])
    if columns:
        columns = [c for c in columns if c in num_df.columns]
        num_df = num_df[columns]
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=False, aspect="auto", title=title, template=PLOTLY_TEMPLATE)
    return fig



def plot_actual_vs_predicted(predictions_df: pd.DataFrame, title: str = "Actual vs Predicted"):
    if predictions_df.empty:
        return None
    fig = px.scatter(
        predictions_df,
        x="y_true",
        y="y_pred",
        trendline="ols",
        title=title,
        template=PLOTLY_TEMPLATE,
    )
    min_v = np.nanmin([predictions_df["y_true"].min(), predictions_df["y_pred"].min()])
    max_v = np.nanmax([predictions_df["y_true"].max(), predictions_df["y_pred"].max()])
    fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(dash="dash"))
    return fig



def plot_residuals(predictions_df: pd.DataFrame, title: str = "Residual plot"):
    if predictions_df.empty:
        return None
    return px.scatter(
        predictions_df,
        x="y_pred",
        y="residual",
        title=title,
        template=PLOTLY_TEMPLATE,
    )



def plot_feature_importance(feature_importance_df: pd.DataFrame, top_k: int = 20, title: str = "Feature importance"):
    if feature_importance_df.empty:
        return None
    work = feature_importance_df.head(top_k).sort_values("importance", ascending=True)
    return px.bar(
        work,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        template=PLOTLY_TEMPLATE,
    )



def plot_risk_gauge(score: float, title: str = "Degradation risk score"):
    score = float(np.clip(score, 0, 100))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "orange"},
                "steps": [
                    {"range": [0, 33], "color": "#14532d"},
                    {"range": [33, 66], "color": "#854d0e"},
                    {"range": [66, 100], "color": "#7f1d1d"},
                ],
            },
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig



def plot_lifetime_index(score: float, title: str = "Relative lifetime index"):
    score = float(np.clip(score, 0, 100))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#60a5fa"},
                "steps": [
                    {"range": [0, 25], "color": "#7f1d1d"},
                    {"range": [25, 50], "color": "#854d0e"},
                    {"range": [50, 75], "color": "#14532d"},
                    {"range": [75, 100], "color": "#166534"},
                ],
            },
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig



def scenario_comparison_bar(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str):
    if df.empty or x_col not in df.columns:
        return None
    fig = go.Figure()
    for y_col in y_cols:
        if y_col not in df.columns:
            continue
        fig.add_trace(go.Bar(name=y_col, x=df[x_col], y=df[y_col]))
    fig.update_layout(barmode="group", template=PLOTLY_TEMPLATE, title=title)
    return fig
