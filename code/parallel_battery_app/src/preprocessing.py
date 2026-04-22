from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import ColumnSchema, common_join_keys, infer_schema, normalize_columns


@dataclass
class PreparedData:
    timeseries_df: pd.DataFrame
    characterization_df: pd.DataFrame
    feature_df: pd.DataFrame
    schema_timeseries: Optional[ColumnSchema]
    schema_characterization: Optional[ColumnSchema]
    notes: List[str]


def _parse_test_metadata_from_name(name: str) -> Dict[str, object]:
    text = Path(str(name)).stem
    patterns = [
        r"(?i)\bM(?P<module>\d+)[_\-](?P<chem>mixed|nmc|nca)[_\-](?P<age>unaged|aged)[_\-]R?(?P<r>\d+)[_\-]T?(?P<t>\d+)\b",
        r"(?i)\bM(?P<module>\d+)[_\-](?P<chem>mixed|nmc|nca)[_\-](?P<age>new|aged)[_\-](?P<r>\d+)[_\-](?P<t>\d+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if not m:
            continue
        gd = m.groupdict()
        chem = str(gd.get("chem", "")).upper()
        if chem == "MIXED":
            chem = "Mixed"
        age = str(gd.get("age", "")).lower().replace("new", "unaged")
        return {
            "module_id": f"M{gd['module']}",
            "chemistry": chem,
            "ageing": age,
            "interconnection_resistance": pd.to_numeric(gd.get("r"), errors="coerce"),
            "operating_temperature": pd.to_numeric(gd.get("t"), errors="coerce"),
            "test_id": text,
        }
    return {"test_id": text}


def _ensure_source_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if "source_file" not in out.columns:
        out["source_file"] = "uploaded_table"
    if "source_table" not in out.columns:
        out["source_table"] = "table"
    if "test_id" not in out.columns:
        out["test_id"] = out["source_file"].astype(str).map(lambda x: Path(x).stem)
    return out


def _standardize_module_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    rename_map: Dict[str, str] = {}
    for old, new in {
        "test_time_s": "elapsed_time",
        "timedata": "elapsed_time",
        "current_a": "module_current",
        "currentdata": "module_current",
        "voltage_v": "module_voltage",
        "voltagedata": "module_voltage",
        "internal_resistance_ohm": "internal_resistance_ohm",
        "charge_capacity_ah": "charge_capacity_ah",
        "discharge_capacity_ah": "discharge_capacity_ah",
    }.items():
        if old in out.columns:
            rename_map[old] = new
    for i in range(1, 5):
        for cand in [f"aux_voltage_v_{i}", f"current_a_cell{i}", f"current_a_cell_{i}", f"cell_current_{i}", f"i_cell_{i}"]:
            if cand in out.columns:
                rename_map[cand] = f"i_cell_{i}"
                break
        for cand in [f"aux_temperature_ae_{i}", f"aux_temperature_{i}", f"temp_cell_{i}", f"cell_temperature_{i}", f"t_cell_{i}"]:
            if cand in out.columns:
                rename_map[cand] = f"t_cell_{i}"
                break
    if "aux_temperature_ae_5" in out.columns:
        rename_map["aux_temperature_ae_5"] = "ambient_temperature"
    elif "aux_temperature_5" in out.columns:
        rename_map["aux_temperature_5"] = "ambient_temperature"
    if "aux_voltage_v_5" in out.columns:
        rename_map["aux_voltage_v_5"] = "hall_power_supply_voltage"
    out = out.rename(columns=rename_map)
    source_name = str(out["source_file"].dropna().iloc[0]) if "source_file" in out.columns and out["source_file"].notna().any() else "uploaded_table"
    meta = _parse_test_metadata_from_name(source_name)
    for k, v in meta.items():
        if k not in out.columns or out[k].isna().all():
            out[k] = v
    for col in ["elapsed_time", "module_current", "module_voltage", "ambient_temperature", "operating_temperature", "interconnection_resistance"] + [f"i_cell_{i}" for i in range(1, 5)] + [f"t_cell_{i}" for i in range(1, 5)]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _standardize_characterization_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    rename_map = {
        "timedata": "elapsed_time",
        "currentdata": "cell_current",
        "tempdata": "cell_temperature",
        "voltagedata": "cell_voltage",
        "cycleindex": "cycle_index",
        "stepindex": "step_index",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    if "source_file" in out.columns:
        stem = out["source_file"].astype(str).map(lambda x: Path(x).stem)
        if "cell_name" not in out.columns:
            out["cell_name"] = stem.str.extract(r"_(P\d+|F\d+|Y1|GS3)$", expand=False)
        cell_name = out["cell_name"].astype("string")
        if "chemistry" not in out.columns:
            chemistry = pd.Series(pd.NA, index=out.index, dtype="string")
            chemistry.loc[cell_name.str.startswith("P", na=False)] = "NMC"
            chemistry.loc[cell_name.str.startswith("F", na=False)] = "NCA"
            out["chemistry"] = chemistry
        if "ageing" not in out.columns:
            ageing = pd.Series("unaged", index=out.index, dtype="string")
            ageing.loc[cell_name.isin(["Y1", "GS3"])] = "aged"
            out["ageing"] = ageing
    for col in ["elapsed_time", "cell_current", "cell_temperature", "cell_voltage", "ocv"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def validate_timeseries_df(df: pd.DataFrame, schema: ColumnSchema) -> List[str]:
    issues: List[str] = []
    if schema.time_col is None:
        issues.append("Could not infer a time column in the timeseries table.")
    if len(schema.cell_current_cols) < 2:
        issues.append("Could not infer at least two cell current columns in the timeseries table.")
    return issues


def aggregate_characterization_df(char_df: pd.DataFrame) -> pd.DataFrame:
    if char_df.empty:
        return char_df
    numeric_cols = char_df.select_dtypes(include=[np.number]).columns.tolist()
    group_candidates = [c for c in ["cell_name", "chemistry", "ageing", "source_file"] if c in char_df.columns]
    if not group_candidates:
        return pd.DataFrame([char_df[numeric_cols].mean(numeric_only=True).to_dict()]) if numeric_cols else pd.DataFrame()
    agg_map = {c: ["mean", "std", "min", "max"] for c in numeric_cols}
    grouped = char_df.groupby(group_candidates, dropna=False).agg(agg_map)
    grouped.columns = ["_".join([lvl for lvl in col if lvl]) for col in grouped.columns.to_flat_index()]
    return grouped.reset_index()


def attach_characterization_features(feature_df: pd.DataFrame, characterization_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty or characterization_df.empty:
        return feature_df
    join_keys = common_join_keys(feature_df, characterization_df)
    if join_keys:
        return feature_df.merge(characterization_df, on=join_keys, how="left")
    numeric_cols = characterization_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        summary = characterization_df[numeric_cols].mean(numeric_only=True).add_prefix("global_char_")
        for k, v in summary.items():
            feature_df[k] = v
    return feature_df


def prepare_data(timeseries_df: pd.DataFrame, characterization_df: Optional[pd.DataFrame] = None, feature_df: Optional[pd.DataFrame] = None) -> PreparedData:
    notes: List[str] = []
    timeseries_df = normalize_columns(timeseries_df) if timeseries_df is not None else pd.DataFrame()
    characterization_df = normalize_columns(characterization_df) if characterization_df is not None else pd.DataFrame()
    feature_df = normalize_columns(feature_df) if feature_df is not None else pd.DataFrame()
    timeseries_df = _ensure_source_cols(timeseries_df)
    if not characterization_df.empty:
        characterization_df = _ensure_source_cols(characterization_df)
    timeseries_df = _standardize_module_timeseries(timeseries_df)
    if not characterization_df.empty:
        characterization_df = _standardize_characterization_df(characterization_df)
    schema_timeseries = infer_schema(timeseries_df) if not timeseries_df.empty else None
    schema_characterization = infer_schema(characterization_df) if not characterization_df.empty else None
    if schema_timeseries:
        notes.extend(validate_timeseries_df(timeseries_df, schema_timeseries))
    if feature_df.empty:
        notes.append("No precomputed feature table was supplied. The app will build a feature table from raw time series.")
    else:
        feature_df = attach_characterization_features(feature_df, aggregate_characterization_df(characterization_df))
    return PreparedData(timeseries_df, characterization_df, feature_df, schema_timeseries, schema_characterization, notes)
