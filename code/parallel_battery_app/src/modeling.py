from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None


@dataclass
class ModelingResult:
    pipeline: Pipeline
    metrics: dict
    predictions_df: pd.DataFrame
    feature_importance_df: pd.DataFrame
    cv_scores: Optional[List[float]]


def _make_model(name: str):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=1.0, random_state=None)
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=1)
    if name == "XGBoost" and XGBRegressor is not None:
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42)
    if name == "XGBoost":
        return RandomForestRegressor(n_estimators=300, random_state=42)
    raise ValueError(f"Unsupported model: {name}")


def train_regression_model(feature_df: pd.DataFrame, target_col: str, model_name: str, group_col: str | None = None, exclude_cols: Optional[List[str]] = None) -> ModelingResult:
    exclude_cols = exclude_cols or []
    df = feature_df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target not found: {target_col}")
    drop_cols = set(exclude_cols + [target_col])
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()
    if len(X) < 5:
        raise ValueError("Not enough rows to train a model. Need at least 5 feature rows.")
    groups = X[group_col].astype(str) if group_col and group_col in X.columns else None
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ],
        remainder="drop",
    )
    model = _make_model(model_name)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    if groups is not None and groups.nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_train = groups.iloc[train_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        g_train = None
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else np.nan,
    }
    pred_df = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": np.ravel(y_pred)})
    feat_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    model_step = pipe.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        imp = np.asarray(model_step.feature_importances_)
    elif hasattr(model_step, "coef_"):
        imp = np.abs(np.ravel(model_step.coef_))
    else:
        imp = np.zeros(len(feat_names))
    feat_imp = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
    cv_scores = None
    try:
        if g_train is not None and len(X_train) >= 6 and g_train.nunique() >= 2:
            cv = GroupKFold(n_splits=min(3, g_train.nunique()))
            scores = cross_val_score(pipe, X_train, y_train, groups=g_train, cv=cv, scoring="neg_root_mean_squared_error")
            cv_scores = (-scores).tolist()
        elif len(X_train) >= 6:
            cv = KFold(n_splits=min(3, len(X_train)), shuffle=True, random_state=42)
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
            cv_scores = (-scores).tolist()
    except Exception:
        cv_scores = None
    return ModelingResult(pipe, metrics, pred_df, feat_imp, cv_scores)


def save_model(pipeline: Pipeline, path: Path) -> None:
    joblib.dump(pipeline, path)
