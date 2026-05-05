"""modeling.py — single-target + multi-output regression with GPU support."""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, KFold, cross_val_score, train_test_split,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def _detect_gpu() -> dict:
    result = {"has_cuda": False, "has_mps": False, "xgb_device": "cpu",
              "n_jobs": -1, "info": "CPU only (all cores)"}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            timeout=4, stderr=subprocess.DEVNULL,
        ).decode().strip()
        if out:
            result.update(has_cuda=True, xgb_device="cuda",
                          info=f"NVIDIA GPU: {out.splitlines()[0]}")
            return result
    except Exception:
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
            result.update(has_mps=True, info="Apple Silicon MPS (XGBoost on CPU)")
            return result
    except Exception:
        pass
    import os
    result["info"] = f"CPU only ({os.cpu_count() or 1} logical cores, n_jobs=-1)"
    return result

_GPU_INFO: Optional[dict] = None
def get_gpu_info() -> dict:
    global _GPU_INFO
    if _GPU_INFO is None:
        _GPU_INFO = _detect_gpu()
    return _GPU_INFO


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ModelingResult:
    """Single-target regression result."""
    pipeline: Pipeline
    metrics: dict
    predictions_df: pd.DataFrame
    feature_importance_df: pd.DataFrame
    cv_scores: Optional[List[float]]
    model_name: str = ""
    target_col: str = ""
    trained_at: str = ""
    gpu_used: bool = False


@dataclass
class MultiOutputModelingResult:
    """
    Multi-target regression result — one model trained on ALL thermal
    targets simultaneously.

    Attributes
    ----------
    pipeline              sklearn Pipeline (preprocessor + model)
    metrics_per_target    {target: {MAE, RMSE, R2, train_time_s}}
    predictions_df        actual_<tgt> + predicted_<tgt> columns for every target
    feature_importance_df mean importance across all targets
    feat_imp_per_target   {target: DataFrame(feature, importance)}
    model_name / target_cols / trained_at / gpu_used
    """
    pipeline: Pipeline
    metrics_per_target: Dict[str, Dict[str, float]]
    predictions_df: pd.DataFrame
    feature_importance_df: pd.DataFrame
    feat_imp_per_target: Dict[str, pd.DataFrame] = field(default_factory=dict)
    model_name: str = ""
    target_cols: List[str] = field(default_factory=list)
    trained_at: str = ""
    gpu_used: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols     = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp",   SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="drop",
        n_jobs=-1,
    )


def _make_base_model(name: str, gpu: dict, multi_output: bool = False):
    nj = gpu["n_jobs"]
    if name == "Linear Regression":
        m = LinearRegression(n_jobs=nj)
        return MultiOutputRegressor(m, n_jobs=nj) if multi_output else m
    if name == "Ridge":
        m = Ridge(alpha=1.0)
        return MultiOutputRegressor(m, n_jobs=nj) if multi_output else m
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=300, random_state=42, min_samples_leaf=1, n_jobs=nj)
    if name == "XGBoost":
        if XGBRegressor is not None:
            kw = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=nj)
            if gpu["has_cuda"]:
                kw["device"] = "cuda"; kw["tree_method"] = "hist"
            m = XGBRegressor(**kw)
            return MultiOutputRegressor(m, n_jobs=nj) if multi_output else m
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=nj)
    raise ValueError(f"Unsupported model: {name}")


def _split(X, y, groups, test_size=0.25):
    if groups is not None and groups.nunique() >= 2:
        sp = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        ti, vi = next(sp.split(X, y, groups))
        return X.iloc[ti], X.iloc[vi], y.iloc[ti], y.iloc[vi], groups.iloc[ti]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
    return Xtr, Xte, ytr, yte, None


def _cross_validate(pipe, X_tr, y_tr, g_tr) -> Optional[List[float]]:
    try:
        if g_tr is not None and len(X_tr) >= 6 and g_tr.nunique() >= 2:
            cv = GroupKFold(n_splits=min(3, g_tr.nunique()))
            sc = cross_val_score(pipe, X_tr, y_tr, groups=g_tr,
                                 cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        elif len(X_tr) >= 6:
            cv = KFold(n_splits=min(3, len(X_tr)), shuffle=True, random_state=42)
            sc = cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
        else:
            return None
        return (-sc).tolist()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Single-target  (unchanged public API)
# ---------------------------------------------------------------------------
def train_regression_model(
    feature_df:   pd.DataFrame,
    target_col:   str,
    model_name:   str,
    group_col:    Optional[str] = None,
    exclude_cols: Optional[List[str]] = None,
    use_gpu:      bool = True,
) -> ModelingResult:
    gpu = get_gpu_info() if use_gpu else {"has_cuda": False, "has_mps": False,
                                           "xgb_device": "cpu", "n_jobs": -1}
    exclude_cols = exclude_cols or []
    df = feature_df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target not found: {target_col}")
    drop = set(exclude_cols + [target_col])
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = y.notna(); X, y = X.loc[mask].copy(), y.loc[mask].copy()
    if len(X) < 5: raise ValueError("Need >= 5 rows.")
    groups = X[group_col].astype(str) if group_col and group_col in X.columns else None
    pipe   = Pipeline([("preprocessor", _build_preprocessor(X)),
                       ("model", _make_base_model(model_name, gpu))])
    X_tr, X_te, y_tr, y_te, g_tr = _split(X, y, groups)
    t0 = time.perf_counter(); pipe.fit(X_tr, y_tr); elapsed = time.perf_counter() - t0
    y_pred = pipe.predict(X_te)
    metrics = {
        "MAE":  float(mean_absolute_error(y_te, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "R2":   float(r2_score(y_te, y_pred)) if len(y_te) >= 2 else np.nan,
        "train_time_s": round(elapsed, 3),
    }
    pred_df = pd.DataFrame({"actual": y_te.to_numpy(), "predicted": np.ravel(y_pred)})
    feat_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    m_step = pipe.named_steps["model"]
    imp = (np.asarray(m_step.feature_importances_) if hasattr(m_step, "feature_importances_")
           else np.abs(np.ravel(m_step.coef_)) if hasattr(m_step, "coef_")
           else np.zeros(len(feat_names)))
    feat_imp = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
    return ModelingResult(
        pipeline=pipe, metrics=metrics, predictions_df=pred_df,
        feature_importance_df=feat_imp, cv_scores=_cross_validate(pipe, X_tr, y_tr, g_tr),
        model_name=model_name, target_col=target_col,
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"), gpu_used=gpu["has_cuda"],
    )


# ---------------------------------------------------------------------------
# Multi-output  (NEW — all thermal targets at once)
# ---------------------------------------------------------------------------
def train_multi_output_model(
    feature_df:   pd.DataFrame,
    target_cols:  List[str],
    model_name:   str,
    group_col:    Optional[str] = None,
    exclude_cols: Optional[List[str]] = None,
    use_gpu:      bool = True,
) -> MultiOutputModelingResult:
    """
    Train ONE model on ALL target_cols simultaneously.

    Pipeline
    --------
    1. Validate: keep only target_cols that exist in feature_df.
    2. Build X by dropping all targets + exclude_cols.
    3. Build Y (n_rows × n_targets), drop rows where ALL targets are NaN,
       then median-impute remaining NaN cells.
    4. Fit shared ColumnTransformer preprocessor on X.
    5. Wrap linear models in MultiOutputRegressor; tree models handle 2D y natively.
    6. Evaluate MAE/RMSE/R2 independently per target on the hold-out set.
    7. Extract per-target feature importances where the estimator exposes them.
    """
    gpu = get_gpu_info() if use_gpu else {"has_cuda": False, "has_mps": False,
                                           "xgb_device": "cpu", "n_jobs": -1}
    exclude_cols = list(exclude_cols or [])
    valid_tgts = [c for c in target_cols if c in feature_df.columns]
    if not valid_tgts:
        raise ValueError(f"No target columns found: {target_cols}")

    df   = feature_df.copy()
    drop = set(exclude_cols + valid_tgts)
    X    = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    Y    = df[valid_tgts].apply(pd.to_numeric, errors="coerce")

    keep = Y.notna().any(axis=1)
    X, Y = X.loc[keep].copy(), Y.loc[keep].copy()
    for col in Y.columns:
        med = Y[col].median()
        Y[col] = Y[col].fillna(med if not np.isnan(med) else 0.0)

    if len(X) < 5:
        raise ValueError(f"Need >= 5 rows — got {len(X)}.")

    groups       = X[group_col].astype(str) if group_col and group_col in X.columns else None
    needs_wrap   = model_name in ("Linear Regression", "Ridge")
    pipe         = Pipeline([
        ("preprocessor", _build_preprocessor(X)),
        ("model",        _make_base_model(model_name, gpu, multi_output=needs_wrap)),
    ])

    X_tr, X_te, Y_tr, Y_te, g_tr = _split(X, Y, groups)
    t0 = time.perf_counter()
    pipe.fit(X_tr, Y_tr)
    elapsed = time.perf_counter() - t0
    log.info("MultiOutput %s  %.2fs  targets=%s  GPU=%s", model_name, elapsed, valid_tgts, gpu["has_cuda"])

    Y_pred = np.atleast_2d(pipe.predict(X_te))
    if Y_pred.shape[0] == len(valid_tgts) and Y_pred.shape[1] != len(valid_tgts):
        Y_pred = Y_pred.T

    # per-target metrics
    metrics_per_target: Dict[str, Dict[str, float]] = {}
    for i, tgt in enumerate(valid_tgts):
        y_t = Y_te.iloc[:, i].to_numpy(); y_p = Y_pred[:, i]
        metrics_per_target[tgt] = {
            "MAE":  float(mean_absolute_error(y_t, y_p)),
            "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
            "R2":   float(r2_score(y_t, y_p)) if len(y_t) >= 2 else np.nan,
            "train_time_s": round(elapsed, 3),
        }

    # predictions dataframe
    pred_rows = {}
    for i, tgt in enumerate(valid_tgts):
        pred_rows[f"actual_{tgt}"]    = Y_te.iloc[:, i].to_numpy()
        pred_rows[f"predicted_{tgt}"] = Y_pred[:, i]
    predictions_df = pd.DataFrame(pred_rows)

    # feature importances
    feat_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    m_step     = pipe.named_steps["model"]
    feat_imp_per_target: Dict[str, pd.DataFrame] = {}

    if isinstance(m_step, MultiOutputRegressor):
        all_imp = []
        for est, tgt in zip(m_step.estimators_, valid_tgts):
            imp = (np.asarray(est.feature_importances_) if hasattr(est, "feature_importances_")
                   else np.abs(np.ravel(est.coef_)) if hasattr(est, "coef_")
                   else np.zeros(len(feat_names)))
            feat_imp_per_target[tgt] = (pd.DataFrame({"feature": feat_names, "importance": imp})
                                        .sort_values("importance", ascending=False))
            all_imp.append(imp)
        mean_imp = np.mean(all_imp, axis=0)
    elif hasattr(m_step, "feature_importances_"):
        raw = m_step.feature_importances_
        if raw.ndim == 2:
            for i, tgt in enumerate(valid_tgts):
                feat_imp_per_target[tgt] = (pd.DataFrame({"feature": feat_names, "importance": raw[i]})
                                             .sort_values("importance", ascending=False))
            mean_imp = raw.mean(axis=0)
        else:
            mean_imp = raw
            fi = pd.DataFrame({"feature": feat_names, "importance": mean_imp}).sort_values("importance", ascending=False)
            for tgt in valid_tgts:
                feat_imp_per_target[tgt] = fi
    else:
        mean_imp = np.zeros(len(feat_names))

    feat_imp_mean = (pd.DataFrame({"feature": feat_names, "importance": mean_imp})
                     .sort_values("importance", ascending=False))

    return MultiOutputModelingResult(
        pipeline=pipe,
        metrics_per_target=metrics_per_target,
        predictions_df=predictions_df,
        feature_importance_df=feat_imp_mean,
        feat_imp_per_target=feat_imp_per_target,
        model_name=model_name,
        target_cols=valid_tgts,
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        gpu_used=gpu["has_cuda"],
    )


def save_model(pipeline, path: Path) -> None:
    joblib.dump(pipeline, path)

def load_model(path: Path):
    return joblib.load(path)