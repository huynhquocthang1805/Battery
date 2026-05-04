"""modeling.py  — with GPU acceleration + auto model persistence."""
from __future__ import annotations

import logging
import subprocess
import sys
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
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU / accelerator detection
# ---------------------------------------------------------------------------

def _detect_gpu() -> dict:
    """
    Probe the runtime for GPU / accelerator availability.
    Returns a dict with keys:
        has_cuda  : NVIDIA GPU visible to CUDA
        has_mps   : Apple Silicon MPS
        xgb_device: 'cuda' | 'mps' | 'cpu'
        n_jobs    : -1 (all cores)
        info      : human-readable string
    """
    result = {"has_cuda": False, "has_mps": False, "xgb_device": "cpu",
              "n_jobs": -1, "info": "CPU only (all cores)"}
    # --- NVIDIA CUDA ---
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            timeout=4, stderr=subprocess.DEVNULL,
        ).decode().strip()
        if out:
            result["has_cuda"]   = True
            result["xgb_device"] = "cuda"
            result["info"]       = f"NVIDIA GPU detected: {out.splitlines()[0]}"
            return result
    except Exception:
        pass
    # --- Apple MPS ---
    try:
        import torch  # type: ignore
        if torch.backends.mps.is_available():
            result["has_mps"]    = True
            result["xgb_device"] = "cpu"   # XGBoost MPS not stable; use CPU
            result["info"]       = "Apple Silicon MPS detected (XGBoost uses CPU)"
            return result
    except Exception:
        pass
    # --- CPU cores ---
    import os
    n = os.cpu_count() or 1
    result["info"] = f"CPU only  ({n} logical cores, n_jobs=-1)"
    return result


_GPU_INFO: Optional[dict] = None


def get_gpu_info() -> dict:
    global _GPU_INFO
    if _GPU_INFO is None:
        _GPU_INFO = _detect_gpu()
    return _GPU_INFO


# ---------------------------------------------------------------------------
# XGBoost import (optional)
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelingResult:
    pipeline: Pipeline
    metrics: dict
    predictions_df: pd.DataFrame
    feature_importance_df: pd.DataFrame
    cv_scores: Optional[List[float]]
    model_name: str = ""
    target_col: str = ""
    trained_at: str = ""
    gpu_used: bool = False


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _make_model(name: str, gpu: dict):
    nj = gpu["n_jobs"]
    if name == "Linear Regression":
        return LinearRegression(n_jobs=nj)
    if name == "Ridge":
        return Ridge(alpha=1.0)
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=300, random_state=42,
            min_samples_leaf=1, n_jobs=nj,
        )
    if name == "XGBoost":
        if XGBRegressor is not None:
            kwargs = dict(
                n_estimators=300, max_depth=4,
                learning_rate=0.05, subsample=0.9,
                colsample_bytree=0.9, random_state=42,
                n_jobs=nj,
            )
            if gpu["has_cuda"]:
                kwargs["device"]      = "cuda"
                kwargs["tree_method"] = "hist"
            return XGBRegressor(**kwargs)
        # fallback
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=nj)
    raise ValueError(f"Unsupported model: {name}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_regression_model(
    feature_df:   pd.DataFrame,
    target_col:   str,
    model_name:   str,
    group_col:    Optional[str] = None,
    exclude_cols: Optional[List[str]] = None,
    use_gpu:      bool = True,
) -> ModelingResult:
    import time
    gpu = get_gpu_info() if use_gpu else {"has_cuda": False, "has_mps": False,
                                           "xgb_device": "cpu", "n_jobs": -1,
                                           "info": "GPU disabled by caller"}
    exclude_cols = exclude_cols or []
    df = feature_df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target not found: {target_col}")

    drop_cols = set(exclude_cols + [target_col])
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid].copy(), y.loc[valid].copy()
    if len(X) < 5:
        raise ValueError("Need at least 5 feature rows to train.")

    groups = X[group_col].astype(str) if group_col and group_col in X.columns else None
    numeric_cols     = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp",   SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imp",  SimpleImputer(strategy="most_frequent")),
                ("ohe",  OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="drop",
        n_jobs=-1,          # parallel column transforms
    )

    model = _make_model(model_name, gpu)
    pipe  = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # --- train/test split ------------------------------------------------
    if groups is not None and groups.nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_train = groups.iloc[train_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        g_train = None

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    log.info("Trained %s in %.2f s  (GPU=%s)", model_name, elapsed, gpu["has_cuda"])

    y_pred = pipe.predict(X_test)
    metrics = {
        "MAE":  float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2":   float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else np.nan,
        "train_time_s": round(elapsed, 3),
    }

    pred_df   = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": np.ravel(y_pred)})
    feat_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    model_step = pipe.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        imp = np.asarray(model_step.feature_importances_)
    elif hasattr(model_step, "coef_"):
        imp = np.abs(np.ravel(model_step.coef_))
    else:
        imp = np.zeros(len(feat_names))
    feat_imp = (
        pd.DataFrame({"feature": feat_names, "importance": imp})
        .sort_values("importance", ascending=False)
    )

    cv_scores = None
    try:
        if g_train is not None and len(X_train) >= 6 and g_train.nunique() >= 2:
            cv = GroupKFold(n_splits=min(3, g_train.nunique()))
            sc = cross_val_score(pipe, X_train, y_train, groups=g_train,
                                 cv=cv, scoring="neg_root_mean_squared_error",
                                 n_jobs=-1)
            cv_scores = (-sc).tolist()
        elif len(X_train) >= 6:
            cv = KFold(n_splits=min(3, len(X_train)), shuffle=True, random_state=42)
            sc = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="neg_root_mean_squared_error",
                                 n_jobs=-1)
            cv_scores = (-sc).tolist()
    except Exception:
        cv_scores = None

    import time as _t
    return ModelingResult(
        pipeline=pipe,
        metrics=metrics,
        predictions_df=pred_df,
        feature_importance_df=feat_imp,
        cv_scores=cv_scores,
        model_name=model_name,
        target_col=target_col,
        trained_at=_t.strftime("%Y-%m-%d %H:%M:%S"),
        gpu_used=gpu["has_cuda"],
    )


def save_model(pipeline: Pipeline, path: Path) -> None:
    joblib.dump(pipeline, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)
