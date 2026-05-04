"""modeling.py  — GPU acceleration + auto model persistence + multi-target batch training."""
from __future__ import annotations

import logging
import subprocess
import time as _time
from dataclasses import dataclass, field
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
    result = {"has_cuda": False, "has_mps": False, "xgb_device": "cpu",
              "n_jobs": -1, "info": "CPU only (all cores)"}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            timeout=4, stderr=subprocess.DEVNULL,
        ).decode().strip()
        if out:
            result["has_cuda"]   = True
            result["xgb_device"] = "cuda"
            result["info"]       = f"NVIDIA GPU: {out.splitlines()[0]}"
            return result
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if torch.backends.mps.is_available():
            result["has_mps"]    = True
            result["xgb_device"] = "cpu"
            result["info"]       = "Apple Silicon MPS (XGBoost → CPU)"
            return result
    except Exception:
        pass
    import os
    n = os.cpu_count() or 1
    result["info"] = f"CPU only ({n} logical cores, n_jobs=-1)"
    return result


_GPU_INFO: Optional[dict] = None


def get_gpu_info() -> dict:
    global _GPU_INFO
    if _GPU_INFO is None:
        _GPU_INFO = _detect_gpu()
    return _GPU_INFO


# ---------------------------------------------------------------------------
# XGBoost (optional)
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
    pipeline:             Pipeline
    metrics:              dict
    predictions_df:       pd.DataFrame
    feature_importance_df: pd.DataFrame
    cv_scores:            Optional[List[float]]
    model_name:           str = ""
    target_col:           str = ""
    trained_at:           str = ""
    gpu_used:             bool = False


@dataclass
class MultiTargetResult:
    """Container for a batch of ModelingResult objects (one per thermal target)."""
    results:    Dict[str, ModelingResult]       # target_col -> result
    errors:     Dict[str, str]                  # target_col -> error message
    model_name: str = ""
    trained_at: str = ""
    total_time_s: float = 0.0

    # ---------- convenience properties ----------
    def metrics_df(self) -> pd.DataFrame:
        """One row per target: MAE, RMSE, R², train_time_s."""
        rows = []
        for tgt, res in self.results.items():
            row = {"target": tgt}
            row.update(res.metrics)
            rows.append(row)
        return pd.DataFrame(rows).set_index("target") if rows else pd.DataFrame()

    def best_target(self, metric: str = "R2") -> Optional[str]:
        df = self.metrics_df()
        if df.empty or metric not in df.columns:
            return None
        return str(df[metric].idxmax())

    def worst_target(self, metric: str = "R2") -> Optional[str]:
        df = self.metrics_df()
        if df.empty or metric not in df.columns:
            return None
        return str(df[metric].idxmin())


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
        return RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=1, n_jobs=nj)
    if name == "XGBoost":
        if XGBRegressor is not None:
            kwargs = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=nj)
            if gpu["has_cuda"]:
                kwargs["device"] = "cuda"
                kwargs["tree_method"] = "hist"
            return XGBRegressor(**kwargs)
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=nj)
    raise ValueError(f"Unsupported model: {name}")


# ---------------------------------------------------------------------------
# Single-target training
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
                                          "xgb_device": "cpu", "n_jobs": -1, "info": "GPU disabled"}
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

    groups       = X[group_col].astype(str) if group_col and group_col in X.columns else None
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols     = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc",  StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop",
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("model", _make_model(model_name, gpu))])

    if groups is not None and groups.nunique() >= 2:
        spl = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        tr, te = next(spl.split(X, y, groups))
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]
        g_train = groups.iloc[tr]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        g_train = None

    t0 = _time.perf_counter()
    pipe.fit(X_train, y_train)
    elapsed = _time.perf_counter() - t0

    y_pred  = pipe.predict(X_test)
    metrics = {
        "MAE":  float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2":   float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else np.nan,
        "train_time_s": round(elapsed, 3),
    }

    pred_df   = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": np.ravel(y_pred)})
    fn        = pipe.named_steps["preprocessor"].get_feature_names_out()
    ms        = pipe.named_steps["model"]
    imp       = (np.asarray(ms.feature_importances_) if hasattr(ms, "feature_importances_")
                 else np.abs(np.ravel(ms.coef_)) if hasattr(ms, "coef_")
                 else np.zeros(len(fn)))
    feat_imp  = pd.DataFrame({"feature": fn, "importance": imp}).sort_values("importance", ascending=False)

    cv_scores = None
    try:
        if g_train is not None and len(X_train) >= 6 and g_train.nunique() >= 2:
            cv = GroupKFold(n_splits=min(3, g_train.nunique()))
            sc = cross_val_score(pipe, X_train, y_train, groups=g_train,
                                 cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        elif len(X_train) >= 6:
            cv = KFold(n_splits=min(3, len(X_train)), shuffle=True, random_state=42)
            sc = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
        else:
            sc = None
        if sc is not None:
            cv_scores = (-sc).tolist()
    except Exception:
        pass

    return ModelingResult(
        pipeline=pipe, metrics=metrics, predictions_df=pred_df,
        feature_importance_df=feat_imp, cv_scores=cv_scores,
        model_name=model_name, target_col=target_col,
        trained_at=_time.strftime("%Y-%m-%d %H:%M:%S"),
        gpu_used=gpu["has_cuda"],
    )


# ---------------------------------------------------------------------------
# Multi-target batch training  (all thermal targets in one call)
# ---------------------------------------------------------------------------

def train_multi_target(
    feature_df:   pd.DataFrame,
    target_cols:  List[str],
    model_name:   str,
    group_col:    Optional[str] = None,
    exclude_cols: Optional[List[str]] = None,
    use_gpu:      bool = True,
    n_parallel:   int = -1,
) -> MultiTargetResult:
    """
    Train one regression model per target column, executed in parallel
    using joblib threads (safe with sklearn/numpy).

    Parameters
    ----------
    feature_df   : engineered feature table (one row per test condition)
    target_cols  : list of thermal target column names to train
    model_name   : e.g. 'Random Forest'
    group_col    : optional column used for group-aware train/test split
    exclude_cols : columns to never use as input features
    use_gpu      : enable XGBoost CUDA acceleration if available
    n_parallel   : number of parallel threads (-1 = all cores)

    Returns
    -------
    MultiTargetResult with .results dict and .errors dict
    """
    gpu = get_gpu_info() if use_gpu else {"has_cuda": False, "has_mps": False,
                                          "xgb_device": "cpu", "n_jobs": -1, "info": "GPU disabled"}

    def _train_one(tgt: str) -> Tuple[str, object]:
        try:
            res = train_regression_model(
                feature_df, tgt, model_name,
                group_col=group_col,
                exclude_cols=exclude_cols,
                use_gpu=use_gpu,
            )
            return tgt, res
        except Exception as exc:
            return tgt, exc

    wall_t0  = _time.perf_counter()
    outcomes = joblib.Parallel(n_jobs=n_parallel, prefer="threads")(
        joblib.delayed(_train_one)(tgt) for tgt in target_cols
    )
    total_s  = round(_time.perf_counter() - wall_t0, 3)

    results: Dict[str, ModelingResult] = {}
    errors:  Dict[str, str] = {}
    for tgt, obj in outcomes:
        if isinstance(obj, ModelingResult):
            results[tgt] = obj
        else:
            errors[tgt] = str(obj)

    return MultiTargetResult(
        results=results,
        errors=errors,
        model_name=model_name,
        trained_at=_time.strftime("%Y-%m-%d %H:%M:%S"),
        total_time_s=total_s,
    )


def save_model(pipeline: Pipeline, path: Path) -> None:
    joblib.dump(pipeline, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)
