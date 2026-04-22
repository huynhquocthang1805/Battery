from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import logger

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBRegressor = None


@dataclass
class ModelingResult:
    pipeline: Pipeline
    metrics: Dict[str, float]
    predictions_df: pd.DataFrame
    feature_names: List[str]
    feature_importance_df: pd.DataFrame
    cv_scores: Optional[List[float]]


MODEL_REGISTRY = {
    "Linear Regression": "linear",
    "Ridge": "ridge",
    "Random Forest": "rf",
    "XGBoost": "xgb",
}



def get_numeric_and_categorical_features(df: pd.DataFrame, target_col: str, exclude_cols: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    exclude = set(exclude_cols or [])
    exclude.add(target_col)
    X = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return numeric_features, categorical_features



def build_preprocessor(numeric_features: List[str], categorical_features: List[str], scale_numeric: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )



def build_regressor(model_name: str):
    key = MODEL_REGISTRY.get(model_name, "ridge")
    if key == "linear":
        return LinearRegression()
    if key == "ridge":
        return Ridge(alpha=1.0, random_state=None)
    if key == "rf":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    if key == "xgb":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed. Install xgboost or choose another model.")
        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")



def train_regression_model(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    group_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    do_cv: bool = True,
    exclude_cols: Optional[List[str]] = None,
) -> ModelingResult:
    work_df = df.copy()
    work_df = work_df.replace([np.inf, -np.inf], np.nan)
    work_df = work_df.dropna(subset=[target_col])
    if work_df.empty:
        raise ValueError(f"No rows with non-null target '{target_col}' are available.")

    numeric_features, categorical_features = get_numeric_and_categorical_features(
        work_df, target_col, exclude_cols=exclude_cols
    )
    if not numeric_features and not categorical_features:
        raise ValueError("No usable features remain after excluding target and non-feature columns.")

    scale_numeric = model_name in {"Linear Regression", "Ridge"}
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=scale_numeric)
    regressor = build_regressor(model_name)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", regressor)])

    groups = work_df[group_col] if group_col and group_col in work_df.columns else None
    feature_cols = numeric_features + categorical_features
    X = work_df[feature_cols]
    y = pd.to_numeric(work_df[target_col], errors="coerce")

    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    else:
        n_test = max(1, int(len(work_df) * test_size))
        indices = np.arange(len(work_df))
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(mean_squared_error(y_test, y_pred, squared=False)),
        "R2": float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else np.nan,
    }

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    logger.info("Feature names after preprocessing: %s", feature_names)

    feature_importance_df = compute_feature_importance(pipeline, feature_names)
    predictions_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred,
            "residual": y_test.values - y_pred,
        },
        index=X_test.index,
    )

    cv_scores = None
    if do_cv and len(work_df) >= 10:
        try:
            if groups is not None:
                n_splits = min(5, max(2, work_df[group_col].nunique()))
                cv = GroupKFold(n_splits=n_splits)
                scores = cross_val_score(
                    pipeline, X, y, groups=groups, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=None
                )
            else:
                from sklearn.model_selection import KFold

                n_splits = min(5, max(2, len(work_df) // 5))
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                scores = cross_val_score(
                    pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=None
                )
            cv_scores = [-float(s) for s in scores]
        except Exception as exc:
            logger.warning("Cross-validation failed: %s", exc)
            cv_scores = None

    return ModelingResult(
        pipeline=pipeline,
        metrics=metrics,
        predictions_df=predictions_df,
        feature_names=feature_names,
        feature_importance_df=feature_importance_df,
        cv_scores=cv_scores,
    )



def compute_feature_importance(pipeline: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_")
    elif hasattr(model, "coef_"):
        coefs = getattr(model, "coef_")
        importances = np.abs(np.ravel(coefs))

    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.DataFrame({"feature": feature_names, "importance": np.ravel(importances)})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)



def save_model(pipeline: Pipeline, output_path: str | Path) -> None:
    joblib.dump(pipeline, output_path)



def load_model(model_path: str | Path) -> Pipeline:
    return joblib.load(model_path)
