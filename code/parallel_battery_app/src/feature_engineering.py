from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import infer_schema


def _safe_stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    return {
        "mean": float(np.nanmean(arr)) if arr.size else np.nan,
        "std": float(np.nanstd(arr)) if arr.size else np.nan,
        "min": float(np.nanmin(arr)) if arr.size else np.nan,
        "max": float(np.nanmax(arr)) if arr.size else np.nan,
        "range": float(np.nanmax(arr) - np.nanmin(arr)) if arr.size else np.nan,
    }


def _window(arr: np.ndarray, part: str) -> np.ndarray:
    n = len(arr)
    if n == 0:
        return arr
    w = max(1, int(n * 0.2))
    if part == "start":
        return arr[:w]
    if part == "mid":
        s = max(0, n // 2 - w // 2)
        return arr[s:s + w]
    return arr[-w:]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return np.nan
    return float(np.dot(x, y - y.mean()) / denom)


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    return float(np.trapz(np.nan_to_num(y, nan=0.0), np.nan_to_num(x, nan=0.0)))


def _coulomb_soc(time_s: np.ndarray, current_a: np.ndarray, capacity_ah: float | None = None) -> np.ndarray:
    if len(time_s) == 0:
        return np.array([])
    t = np.asarray(time_s, dtype=float)
    i = np.asarray(current_a, dtype=float)
    dt = np.diff(t, prepend=t[0])
    if capacity_ah is None or not np.isfinite(capacity_ah) or capacity_ah <= 0:
        capacity_ah = max(np.nansum(np.abs(i) * np.maximum(dt, 0)) / 3600.0, 1e-6)
    delta = np.cumsum(i * np.maximum(dt, 0) / 3600.0) / capacity_ah
    soc = 1.0 - delta
    return soc


def build_feature_table_from_timeseries(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    if timeseries_df.empty:
        return pd.DataFrame()
    schema = infer_schema(timeseries_df)
    group_cols = [c for c in [schema.test_id_col, schema.module_id_col, "source_file", "source_table"] if c and c in timeseries_df.columns]
    if not group_cols:
        timeseries_df = timeseries_df.copy()
        timeseries_df["synthetic_test_id"] = "test_001"
        group_cols = ["synthetic_test_id"]
    records: List[dict] = []
    for keys, grp in timeseries_df.groupby(group_cols, dropna=False):
        g = grp.sort_values(schema.time_col) if schema.time_col and schema.time_col in grp.columns else grp.copy()
        rec = {}
        if not isinstance(keys, tuple):
            keys = (keys,)
        for c, v in zip(group_cols, keys):
            rec[c] = v
        for c in [schema.chemistry_col, schema.ageing_col, schema.operating_temp_col, schema.interconnection_res_col]:
            if c and c in g.columns:
                val = g[c].dropna()
                rec[c] = val.iloc[0] if not val.empty else np.nan
        t = g[schema.time_col].to_numpy(dtype=float) if schema.time_col and schema.time_col in g.columns else np.arange(len(g), dtype=float)
        rec["n_samples"] = len(g)
        if schema.module_current_col and schema.module_current_col in g.columns:
            mod_i = g[schema.module_current_col].to_numpy(dtype=float)
            rec.update({f"module_current_{k}": v for k, v in _safe_stats(mod_i).items()})
            rec["module_current_slope"] = _slope(t, mod_i)
        if schema.module_voltage_col and schema.module_voltage_col in g.columns:
            mod_v = g[schema.module_voltage_col].to_numpy(dtype=float)
            rec.update({f"module_voltage_{k}": v for k, v in _safe_stats(mod_v).items()})
            rec["module_voltage_slope"] = _slope(t, mod_v)
        if schema.cell_current_cols:
            current_mat = g[schema.cell_current_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            sigma_i = np.nanstd(current_mat, axis=1)
            pairwise = []
            for i in range(current_mat.shape[1]):
                for j in range(i + 1, current_mat.shape[1]):
                    pairwise.append(np.nanmean(np.abs(current_mat[:, i] - current_mat[:, j])))
            rec["current_spread_mean"] = float(np.nanmean(sigma_i))
            rec["current_spread_auc"] = _auc(t, sigma_i)
            rec["pairwise_current_diff_mean"] = float(np.nanmean(pairwise)) if pairwise else np.nan
            for part in ["start", "mid", "end"]:
                rec[f"sigma_i_{part}"] = float(np.nanmean(_window(sigma_i, part)))
            for idx, col in enumerate(schema.cell_current_cols, 1):
                vec = g[col].to_numpy(dtype=float)
                rec.update({f"{col}_{k}": v for k, v in _safe_stats(vec).items()})
                rec[f"{col}_slope"] = _slope(t, vec)
            soc_mat = np.column_stack([_coulomb_soc(t, g[col].to_numpy(dtype=float)) for col in schema.cell_current_cols])
            delta_soc = np.nanmax(soc_mat, axis=1) - np.nanmin(soc_mat, axis=1)
            rec["delta_soc_max"] = float(np.nanmax(delta_soc))
            rec["delta_soc_end"] = float(delta_soc[-1]) if len(delta_soc) else np.nan
        if schema.cell_temp_cols:
            temp_mat = g[schema.cell_temp_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            sigma_t = np.nanstd(temp_mat, axis=1)
            delta_t = np.nanmax(temp_mat, axis=1) - np.nanmin(temp_mat, axis=1)
            rec["sigma_t_mean"] = float(np.nanmean(sigma_t))
            rec["delta_t_max"] = float(np.nanmax(delta_t))
            rec["temp_peak"] = float(np.nanmax(temp_mat))
            rec["module_temp_gradient_series_auc"] = _auc(t, delta_t)
            for part in ["start", "mid", "end"]:
                rec[f"sigma_t_{part}"] = float(np.nanmean(_window(sigma_t, part)))
                rec[f"delta_t_{part}"] = float(np.nanmean(_window(delta_t, part)))
            rec["delta_t_start"] = rec.get("delta_t_start", np.nan)
            rec["delta_t_mid"] = rec.get("delta_t_mid", np.nan)
            rec["delta_t_end"] = rec.get("delta_t_end", np.nan)
            if "ambient_temperature" in g.columns:
                amb = pd.to_numeric(g["ambient_temperature"], errors="coerce").to_numpy(dtype=float)
                rec["ambient_temperature"] = float(np.nanmean(amb))
                rec["temp_rise_over_ambient"] = float(np.nanmax(np.nanmax(temp_mat, axis=1) - amb))
            for idx, col in enumerate(schema.cell_temp_cols, 1):
                vec = g[col].to_numpy(dtype=float)
                rec.update({f"{col}_{k}": v for k, v in _safe_stats(vec).items()})
                rec[f"{col}_slope"] = _slope(t, vec)
        if schema.cell_current_cols:
            sigma_i = g[schema.cell_current_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            sigma_series = np.nanstd(sigma_i, axis=1)
            threshold = 0.05 * np.nanmax(np.abs(g[schema.module_current_col].to_numpy(dtype=float))) if schema.module_current_col and schema.module_current_col in g.columns else 0.05
            mask = sigma_series <= threshold
            rec["ttsb"] = float(t[np.argmax(mask)] if mask.any() else np.nanmax(t)) if len(t) else np.nan
        records.append(rec)
    return pd.DataFrame(records)


def integrate_characterization_features(feature_df: pd.DataFrame, characterization_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty or characterization_df.empty:
        return feature_df
    out = feature_df.copy()
    char = characterization_df.copy()
    numeric_cols = char.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return out
    if "chemistry" in out.columns and "chemistry" in char.columns:
        grp_cols = [c for c in ["chemistry", "ageing"] if c in char.columns and c in out.columns]
        if grp_cols:
            agg = char.groupby(grp_cols, dropna=False)[numeric_cols].mean().reset_index().add_prefix("char_")
            rename_back = {f"char_{c}": c for c in grp_cols}
            agg = agg.rename(columns=rename_back)
            out = out.merge(agg, on=grp_cols, how="left")
            return out
    summary = char[numeric_cols].mean(numeric_only=True).add_prefix("global_char_")
    for k, v in summary.items():
        out[k] = v
    return out


def build_risk_scores(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return feature_df
    out = feature_df.copy()
    drivers = {
        "sigma_i_mean": out[[c for c in ["sigma_i_start", "sigma_i_mid", "sigma_i_end"] if c in out.columns]].mean(axis=1),
        "delta_soc": out["delta_soc_max"] if "delta_soc_max" in out.columns else pd.Series(0.0, index=out.index),
        "sigma_t": out["sigma_t_mean"] if "sigma_t_mean" in out.columns else pd.Series(0.0, index=out.index),
        "delta_t": out["delta_t_max"] if "delta_t_max" in out.columns else pd.Series(0.0, index=out.index),
        "ttsb": out["ttsb"] if "ttsb" in out.columns else pd.Series(0.0, index=out.index),
    }
    for name, series in list(drivers.items()):
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() == 0:
            drivers[name] = pd.Series(0.0, index=out.index)
        else:
            lo, hi = float(s.min()), float(s.max())
            drivers[name] = (s - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=out.index)
    age_penalty = pd.Series(0.0, index=out.index)
    if "ageing" in out.columns:
        age_penalty = out["ageing"].astype(str).str.lower().isin(["aged", "yes", "1"]).astype(float)
    mix_penalty = pd.Series(0.0, index=out.index)
    if "chemistry" in out.columns:
        mix_penalty = out["chemistry"].astype(str).str.lower().isin(["mix", "mixed"]).astype(float)
    score = 100 * (
        0.24 * drivers["sigma_i_mean"] +
        0.16 * drivers["delta_soc"] +
        0.20 * drivers["sigma_t"] +
        0.16 * drivers["delta_t"] +
        0.12 * drivers["ttsb"] +
        0.07 * age_penalty +
        0.05 * mix_penalty
    )
    out["degradation_risk_score"] = score.clip(0, 100)
    out["relative_lifetime_index"] = (100 - out["degradation_risk_score"]).clip(0, 100)
    out["estimated_cycle_life_band"] = pd.cut(out["relative_lifetime_index"], bins=[-1, 33, 66, 100], labels=["low", "medium", "high"])
    out["risk_model_features_used"] = "sigma_I, delta_SoC, sigma_T, delta_T, TTSB, ageing, chemistry_mix"
    return out
