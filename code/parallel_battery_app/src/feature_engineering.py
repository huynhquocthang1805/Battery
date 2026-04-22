from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import (
    ColumnSchema,
    as_ordered_unique,
    compute_slope,
    infer_schema,
    logger,
    rolling_stats,
    safe_divide,
)


DEFAULT_NOMINAL_CAPACITY_AH = 5.0



def _window_slices(n: int) -> Dict[str, slice]:
    third = max(1, n // 3)
    return {
        "start": slice(0, third),
        "mid": slice(third, min(2 * third, n)),
        "end": slice(min(2 * third, n - 1), n),
    }



def _series_stats(values: pd.Series, prefix: str) -> Dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna()
    if arr.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_slope": np.nan,
            f"{prefix}_auc": np.nan,
        }
    x = np.arange(len(arr), dtype=float)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std(ddof=0)),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_range": float(arr.max() - arr.min()),
        f"{prefix}_slope": compute_slope(x, arr.to_numpy()),
        f"{prefix}_auc": float(np.trapz(arr.to_numpy(), x=x)),
    }



def estimate_soc_by_coulomb_counting(
    current_series: pd.Series,
    dt_seconds: float = 1.0,
    nominal_capacity_ah: float = DEFAULT_NOMINAL_CAPACITY_AH,
    initial_soc: float = 1.0,
) -> pd.Series:
    current = pd.to_numeric(current_series, errors="coerce").fillna(0.0)
    delta_ah = current * dt_seconds / 3600.0
    soc = initial_soc - delta_ah.cumsum() / max(nominal_capacity_ah, 1e-9)
    return soc.clip(lower=0.0, upper=1.05)



def compute_pairwise_differences(df: pd.DataFrame, cols: List[str], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if len(cols) < 2:
        return out
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            diff = (pd.to_numeric(df[cols[i]], errors="coerce") - pd.to_numeric(df[cols[j]], errors="coerce")).abs()
            name = f"{prefix}_diff_{i+1}_{j+1}"
            out[f"{name}_mean"] = float(diff.mean()) if not diff.empty else np.nan
            out[f"{name}_max"] = float(diff.max()) if not diff.empty else np.nan
    return out



def compute_imbalance_metrics(group: pd.DataFrame, schema: ColumnSchema) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    current_matrix = group[schema.cell_current_cols].apply(pd.to_numeric, errors="coerce")
    n = len(group)
    windows = _window_slices(n)

    sigma_i = current_matrix.std(axis=1, ddof=0)
    spread_i = current_matrix.max(axis=1) - current_matrix.min(axis=1)

    metrics["current_spread_mean"] = float(spread_i.mean()) if not spread_i.empty else np.nan
    metrics["current_spread_max"] = float(spread_i.max()) if not spread_i.empty else np.nan
    for key, window in windows.items():
        sigma_window = sigma_i.iloc[window]
        metrics[f"sigma_i_{key}"] = float(sigma_window.mean()) if not sigma_window.empty else np.nan

    metrics.update(compute_pairwise_differences(group, schema.cell_current_cols, "current"))

    if schema.cell_temp_cols:
        temp_matrix = group[schema.cell_temp_cols].apply(pd.to_numeric, errors="coerce")
        sigma_t = temp_matrix.std(axis=1, ddof=0)
        delta_t = temp_matrix.max(axis=1) - temp_matrix.min(axis=1)
        metrics["sigma_t_mean"] = float(sigma_t.mean()) if not sigma_t.empty else np.nan
        metrics["delta_t_max"] = float(delta_t.max()) if not delta_t.empty else np.nan
        metrics["temperature_gradient_auc"] = float(np.trapz(delta_t.fillna(0.0).to_numpy())) if not delta_t.empty else np.nan
        metrics.update(compute_pairwise_differences(group, schema.cell_temp_cols, "temperature"))

    return metrics



def compute_soc_features(group: pd.DataFrame, schema: ColumnSchema) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if not schema.cell_current_cols:
        return features
    dt_seconds = 1.0
    if schema.time_col and schema.time_col in group.columns:
        time_values = pd.to_numeric(group[schema.time_col], errors="coerce")
        if time_values.notna().sum() >= 2:
            diffs = time_values.diff().dropna()
            dt_seconds = float(diffs.median()) if not diffs.empty else 1.0
    soc_df = pd.DataFrame(index=group.index)
    for current_col in schema.cell_current_cols:
        soc_df[f"soc_{current_col}"] = estimate_soc_by_coulomb_counting(
            group[current_col], dt_seconds=dt_seconds, nominal_capacity_ah=DEFAULT_NOMINAL_CAPACITY_AH
        )
    delta_soc = soc_df.max(axis=1) - soc_df.min(axis=1)
    features["delta_soc_max"] = float(delta_soc.max()) if not delta_soc.empty else np.nan
    features["delta_soc_end"] = float(delta_soc.iloc[-1]) if not delta_soc.empty else np.nan
    features["sigma_soc_mean"] = float(soc_df.std(axis=1, ddof=0).mean()) if not soc_df.empty else np.nan
    return features



def compute_ttsb(group: pd.DataFrame, schema: ColumnSchema, threshold_a: float = 0.2) -> float:
    if len(schema.cell_current_cols) < 2:
        return np.nan
    currents = group[schema.cell_current_cols].apply(pd.to_numeric, errors="coerce")
    branch_deviation = currents.sub(currents.mean(axis=1), axis=0).abs().sum(axis=1)
    balanced_idx = np.where(branch_deviation.fillna(np.inf).to_numpy() <= threshold_a)[0]
    if balanced_idx.size == 0:
        return np.nan
    idx = int(balanced_idx[0])
    if schema.time_col and schema.time_col in group.columns:
        return float(pd.to_numeric(group[schema.time_col], errors="coerce").iloc[idx])
    return float(idx)



def compute_dispersion_features(df: pd.DataFrame, columns: List[str], prefix: str) -> Dict[str, float]:
    if not columns:
        return {}
    values = df[columns].apply(pd.to_numeric, errors="coerce")
    flattened = values.to_numpy().astype(float).ravel()
    flattened = flattened[np.isfinite(flattened)]
    if flattened.size == 0:
        return {}
    mean_value = float(np.mean(flattened))
    std_value = float(np.std(flattened))
    return {
        f"{prefix}_mean": mean_value,
        f"{prefix}_std": std_value,
        f"{prefix}_cv": safe_divide(std_value, mean_value, fill_value=np.nan),
        f"{prefix}_min": float(np.min(flattened)),
        f"{prefix}_max": float(np.max(flattened)),
        f"{prefix}_range": float(np.max(flattened) - np.min(flattened)),
    }



def compute_thermal_features(group: pd.DataFrame, schema: ColumnSchema) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if not schema.cell_temp_cols:
        return features
    temp_matrix = group[schema.cell_temp_cols].apply(pd.to_numeric, errors="coerce")
    max_temp = temp_matrix.max(axis=1)
    min_temp = temp_matrix.min(axis=1)
    mean_temp = temp_matrix.mean(axis=1)
    delta_t = max_temp - min_temp
    windows = _window_slices(len(group))

    features.update(_series_stats(mean_temp, "module_temp_mean_series"))
    features.update(_series_stats(delta_t, "module_temp_gradient_series"))
    features["temp_peak"] = float(max_temp.max()) if not max_temp.empty else np.nan
    features["temp_min"] = float(min_temp.min()) if not min_temp.empty else np.nan
    features["temp_rise"] = float(mean_temp.iloc[-1] - mean_temp.iloc[0]) if len(mean_temp) >= 2 else np.nan

    for name, window in windows.items():
        sub = temp_matrix.iloc[window]
        features[f"sigma_t_{name}"] = float(sub.std(axis=1, ddof=0).mean()) if not sub.empty else np.nan
        delta_sub = sub.max(axis=1) - sub.min(axis=1)
        features[f"delta_t_{name}"] = float(delta_sub.mean()) if not delta_sub.empty else np.nan

    if schema.time_col and schema.time_col in group.columns:
        t = pd.to_numeric(group[schema.time_col], errors="coerce").to_numpy()
    else:
        t = np.arange(len(group), dtype=float)

    for col in schema.cell_temp_cols:
        series = pd.to_numeric(group[col], errors="coerce")
        features[f"{col}_slope"] = compute_slope(t, series.fillna(method="ffill").fillna(0.0).to_numpy())
        features.update(rolling_stats(series.fillna(method="ffill").fillna(0.0), window=min(30, max(3, len(series) // 10 or 3))))

    return features



def compute_module_level_aggregates(group: pd.DataFrame, schema: ColumnSchema) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if schema.module_current_col and schema.module_current_col in group.columns:
        features.update(_series_stats(pd.to_numeric(group[schema.module_current_col], errors="coerce"), "module_current"))
    if schema.module_voltage_col and schema.module_voltage_col in group.columns:
        features.update(_series_stats(pd.to_numeric(group[schema.module_voltage_col], errors="coerce"), "module_voltage"))
    return features



def compute_static_metadata(group: pd.DataFrame, schema: ColumnSchema) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for attr_name, col_name in [
        ("chemistry", schema.chemistry_col),
        ("ageing", schema.ageing_col),
        ("operating_temperature", schema.operating_temp_col),
        ("interconnection_resistance", schema.interconnection_res_col),
        ("module_id", schema.module_id_col),
    ]:
        if col_name and col_name in group.columns:
            value_series = group[col_name].dropna()
            out[attr_name] = value_series.iloc[0] if not value_series.empty else np.nan
    return out



def build_feature_table_from_timeseries(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    if timeseries_df.empty:
        return pd.DataFrame()
    schema = infer_schema(timeseries_df)
    if schema.time_col is None or len(schema.cell_current_cols) < 2:
        logger.warning("Timeseries schema is incomplete for feature generation.")
        return pd.DataFrame()

    group_cols = [c for c in [schema.test_id_col, schema.module_id_col, "source_file", "source_table"] if c and c in timeseries_df.columns]
    if not group_cols:
        timeseries_df = timeseries_df.copy()
        timeseries_df["synthetic_test_id"] = "test_001"
        group_cols = ["synthetic_test_id"]

    rows: List[Dict[str, object]] = []
    for group_key, group in timeseries_df.groupby(group_cols, dropna=False):
        group = group.sort_values(schema.time_col).reset_index(drop=True)
        row: Dict[str, object] = {}
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for col, val in zip(group_cols, group_key):
            row[col] = val

        row.update(compute_static_metadata(group, schema))
        row.update(compute_module_level_aggregates(group, schema))
        row.update(compute_imbalance_metrics(group, schema))
        row.update(compute_soc_features(group, schema))
        row.update(compute_thermal_features(group, schema))
        row["ttsb"] = compute_ttsb(group, schema)

        # Cell-current features and dispersion.
        for col in schema.cell_current_cols:
            row.update(_series_stats(pd.to_numeric(group[col], errors="coerce"), col))
        row.update(compute_dispersion_features(group, schema.cell_current_cols, "current_cells"))
        if schema.cell_temp_cols:
            row.update(compute_dispersion_features(group, schema.cell_temp_cols, "temp_cells"))

        row["n_samples"] = len(group)
        rows.append(row)

    feature_df = pd.DataFrame(rows)
    feature_df.columns = [str(c) for c in feature_df.columns]
    return feature_df



def integrate_characterization_features(
    feature_df: pd.DataFrame,
    characterization_df: pd.DataFrame,
) -> pd.DataFrame:
    if feature_df.empty or characterization_df.empty:
        return feature_df
    char = characterization_df.copy()
    char.columns = [str(c) for c in char.columns]
    numeric_cols = char.select_dtypes(include=[np.number]).columns.tolist()
    join_cols = [c for c in ["test_id", "module_id", "chemistry", "ageing", "source_file"] if c in char.columns and c in feature_df.columns]
    if join_cols:
        agg_map = {c: ["mean", "std", "min", "max"] for c in numeric_cols}
        char_agg = char.groupby(join_cols, dropna=False).agg(agg_map)
        char_agg.columns = ["_".join([lvl for lvl in col if lvl]) for col in char_agg.columns.to_flat_index()]
        char_agg = char_agg.reset_index()
        return feature_df.merge(char_agg, on=join_cols, how="left")

    if numeric_cols:
        global_summary = {}
        for col in numeric_cols:
            global_summary[f"char_{col}_mean"] = char[col].mean()
            global_summary[f"char_{col}_std"] = char[col].std(ddof=0)
        for col, value in global_summary.items():
            feature_df[col] = value
    return feature_df



def build_risk_scores(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return feature_df
    df = feature_df.copy()
    harmful_candidates = [
        "sigma_i_end",
        "sigma_i_mid",
        "delta_t_max",
        "sigma_t_mean",
        "delta_soc_max",
        "ttsb",
        "current_cells_std",
        "temp_cells_std",
    ]
    capacity_candidates = [c for c in df.columns if "capacity" in c and df[c].dtype != object]
    resistance_candidates = [c for c in df.columns if ("resistance" in c or "r0" in c) and df[c].dtype != object]

    def zscore(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        std = s.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    risk = pd.Series(0.0, index=df.index)
    used_features: List[str] = []
    for col in harmful_candidates:
        if col in df.columns:
            risk += zscore(df[col]).clip(lower=-3, upper=3)
            used_features.append(col)

    for col in resistance_candidates[:5]:
        risk += 0.5 * zscore(df[col]).clip(lower=-3, upper=3)
        used_features.append(col)

    for col in capacity_candidates[:5]:
        risk -= 0.35 * zscore(df[col]).clip(lower=-3, upper=3)
        used_features.append(col)

    if "ageing" in df.columns:
        ageing = df["ageing"].astype(str).str.lower().str.contains("aged|true|1|yes")
        risk += ageing.astype(float) * 0.75
        used_features.append("ageing")

    if "chemistry" in df.columns:
        mix = df["chemistry"].astype(str).str.lower().str.contains("mix|nca.*nmc|nmc.*nca")
        risk += mix.astype(float) * 0.5
        used_features.append("chemistry")

    risk_score = (risk - risk.min()) / max((risk.max() - risk.min()), 1e-9) * 100.0
    df["degradation_risk_score"] = risk_score.clip(0, 100)
    df["relative_lifetime_index"] = (100 - df["degradation_risk_score"]).clip(0, 100)
    df["estimated_cycle_life_band"] = pd.cut(
        df["relative_lifetime_index"],
        bins=[-0.1, 25, 50, 75, 100],
        labels=["Very Low", "Low", "Moderate", "High"],
    )
    df["risk_model_features_used"] = ", ".join(as_ordered_unique(used_features))
    return df
