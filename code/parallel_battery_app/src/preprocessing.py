from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import re

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


REQUIRED_TIMESERIES_CORE = ["time-like column", "at least 2 cell current columns"]


# ---------------------------------------------------------------------
# Helpers for the Mendeley parallel-module dataset
# ---------------------------------------------------------------------

def _parse_test_metadata_from_name(name: str) -> Dict[str, object]:
    """
    Parse module-level test metadata from file names such as:
      - M1_Mixed_Unaged_R0_T10.xlsx
      - M2_NMC_Aged_3_40.mat
      - TEST_25_MIX_R0_T10_NEW.xlsx
    """
    text = Path(str(name)).stem
    patterns = [
        r"(?i)\bM(?P<module>\d+)[_\-](?P<chem>mixed|nmc|nca)[_\-](?P<age>unaged|aged)[_\-]R?(?P<r>\d+)[_\-]T?(?P<t>\d+)\b",
        r"(?i)\bM(?P<module>\d+)[_\-](?P<chem>mixed|nmc|nca)[_\-](?P<age>new|aged)[_\-](?P<r>\d+)[_\-](?P<t>\d+)\b",
        r"(?i)\bTEST[_\-](?P<id>\d+)[_\-](?P<chem>mix|mixed|nmc|nca)[_\-]R?(?P<r>\d+)[_\-]T?(?P<t>\d+)[_\-](?P<age>new|aged)\b",
    ]

    for pat in patterns:
        match = re.search(pat, text)
        if not match:
            continue

        gd = match.groupdict()
        chem = str(gd.get("chem", "")).upper()
        if chem == "MIX":
            chem = "Mixed"

        age = str(gd.get("age", "")).lower()
        if age == "new":
            age = "unaged"

        return {
            "module_id": f"M{gd['module']}" if gd.get("module") else np.nan,
            "chemistry": chem,
            "ageing": age,
            "interconnection_resistance": pd.to_numeric(gd.get("r"), errors="coerce"),
            "operating_temperature": pd.to_numeric(gd.get("t"), errors="coerce"),
        }

    return {}



def _ensure_source_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    if "source_file" not in df.columns:
        df["source_file"] = "uploaded_table"
    if "source_table" not in df.columns:
        df["source_table"] = "table"
    if "test_id" not in df.columns:
        df["test_id"] = df["source_file"].astype(str).map(lambda x: Path(x).stem)
    return df



def _standardize_module_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize raw XLSX module-level files and processed MAT module-level files
    into one schema that infer_schema() can understand.

    Important mapping for this dataset:
      - raw Data sheet:
          Current(A)              -> module_current
          Voltage(V)              -> module_voltage
          Test_Time(s)            -> elapsed_time
          Aux_Voltage(V)_1..4     -> i_cell_1..4      (Hall sensor outputs / branch current proxies)
          Aux_Voltage(V)_5        -> hall_power_supply_voltage
          Aux_Temperature_*_1..4  -> t_cell_1..4      (cell temperatures)
          Aux_Temperature_*_5     -> ambient_temperature
      - processed MAT files typically already contain Current(A)_Cell(1-4)
        and temperature-per-cell columns; those are also mapped below.
    """
    if df.empty:
        return df

    df = df.copy()
    rename_map: Dict[str, str] = {}

    # Prefer numeric elapsed time instead of datetime for dt estimation.
    if "test_time_s" in df.columns:
        rename_map["test_time_s"] = "elapsed_time"
    elif "timedata" in df.columns:
        rename_map["timedata"] = "elapsed_time"
    elif "time_data" in df.columns:
        rename_map["time_data"] = "elapsed_time"
    elif "test_times" in df.columns:
        rename_map["test_times"] = "elapsed_time"

    # Module-level electrical quantities.
    if "current_a" in df.columns:
        rename_map["current_a"] = "module_current"
    elif "currentdata" in df.columns:
        rename_map["currentdata"] = "module_current"

    if "voltage_v" in df.columns:
        rename_map["voltage_v"] = "module_voltage"
    elif "voltagedata" in df.columns:
        rename_map["voltagedata"] = "module_voltage"
    elif "ocv" in df.columns and "module_voltage" not in df.columns:
        rename_map["ocv"] = "module_voltage"

    # Raw XLSX: Hall sensor voltages for 4 branches.
    for i in range(1, 5):
        candidates = [
            f"aux_voltage_v_{i}",
            f"currenta_cell{i}",
            f"current_a_cell{i}",
            f"current_a_cell_{i}",
            f"current_cell_{i}",
            f"cell_current_{i}",
        ]
        for raw_name in candidates:
            if raw_name in df.columns:
                rename_map[raw_name] = f"i_cell_{i}"
                break

    # Raw XLSX: 4 cell thermocouples + 1 ambient thermocouple.
    for i in range(1, 5):
        candidates = [
            f"aux_temperature_{i}",
            f"temperaturec_cell{i}",
            f"temperaturec_cell_{i}",
            f"temp_cell_{i}",
            f"cell_temperature_{i}",
            f"t_cell_{i}",
        ]
        for raw_name in candidates:
            if raw_name in df.columns:
                rename_map[raw_name] = f"t_cell_{i}"
                break

    for raw_ambient in [
        "aux_temperature_5",
        "ambient_temperaturec",
        "ambient_tempdata",
        "ambient_temperature",
        "ambient_temp",
    ]:
        if raw_ambient in df.columns:
            rename_map[raw_ambient] = "ambient_temperature"
            break

    # Hall power supply voltage, not a branch current.
    if "aux_voltage_v_5" in df.columns:
        rename_map["aux_voltage_v_5"] = "hall_power_supply_voltage"

    df = df.rename(columns=rename_map)

    # Add metadata parsed from file name.
    source_name = ""
    if "source_file" in df.columns and df["source_file"].notna().any():
        source_name = str(df["source_file"].dropna().iloc[0])
    parsed_meta = _parse_test_metadata_from_name(source_name)
    for col, value in parsed_meta.items():
        if col not in df.columns or df[col].isna().all():
            df[col] = value

    if "module_id" not in df.columns:
        df["module_id"] = "M1"
    if "test_id" not in df.columns:
        df["test_id"] = Path(source_name).stem if source_name else "test_001"

    # Numeric coercion on expected columns.
    num_cols = [
        "elapsed_time",
        "module_current",
        "module_voltage",
        "interconnection_resistance",
        "operating_temperature",
        "ambient_temperature",
        "charge_capacity_ah",
        "discharge_capacity_ah",
        "internal_resistance_ohm",
    ] + [f"i_cell_{i}" for i in range(1, 5)] + [f"t_cell_{i}" for i in range(1, 5)]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df



def _standardize_characterization_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    rename_map = {
        "timedata": "elapsed_time",
        "currentdata": "cell_current",
        "tempdata": "cell_temperature",
        "voltagedata": "cell_voltage",
        "ocv": "ocv",
        "cycleindex": "cycle_index",
        "stepindex": "step_index",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "source_file" in df.columns:
        stem = df["source_file"].astype(str).map(lambda x: Path(x).stem)
        if "cell_name" not in df.columns:
            df["cell_name"] = stem.str.extract(r"_(P\d+|F\d+|Y1|GS3)$", expand=False)
        if "chemistry" not in df.columns:
            df["chemistry"] = np.where(
                df["cell_name"].astype(str).str.startswith("P"),
                "NMC",
                np.where(df["cell_name"].astype(str).str.startswith("F"), "NCA", np.nan),
            )
        if "ageing" not in df.columns:
            df["ageing"] = np.where(df["cell_name"].astype(str).isin(["Y1", "GS3"]), "aged", "unaged")

    for col in ["elapsed_time", "cell_current", "cell_temperature", "cell_voltage", "ocv"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------
# Validation / aggregation
# ---------------------------------------------------------------------

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

    df = normalize_columns(char_df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    group_candidates = [c for c in ["cell_name", "chemistry", "ageing", "source_file"] if c in df.columns]

    if not group_candidates:
        if not numeric_cols:
            return pd.DataFrame()
        return pd.DataFrame([df[numeric_cols].mean(numeric_only=True).to_dict()])

    agg_map = {c: ["mean", "std", "min", "max"] for c in numeric_cols}
    grouped = df.groupby(group_candidates, dropna=False).agg(agg_map)
    grouped.columns = ["_".join([lvl for lvl in col if lvl]) for col in grouped.columns.to_flat_index()]
    return grouped.reset_index()



def attach_characterization_features(feature_df: pd.DataFrame, characterization_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty or characterization_df.empty:
        return feature_df

    feat = feature_df.copy()
    char = characterization_df.copy()
    join_keys = common_join_keys(feat, char)

    if join_keys:
        return feat.merge(char, on=join_keys, how="left")

    numeric_cols = char.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        summary = char[numeric_cols].mean(numeric_only=True).add_prefix("global_char_")
        for col, value in summary.items():
            feat[col] = value

    return feat


# ---------------------------------------------------------------------
# Main API used by app.py
# ---------------------------------------------------------------------

def prepare_data(
    timeseries_df: pd.DataFrame,
    characterization_df: Optional[pd.DataFrame] = None,
    feature_df: Optional[pd.DataFrame] = None,
) -> PreparedData:
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

    aggregated_char = aggregate_characterization_df(characterization_df) if not characterization_df.empty else pd.DataFrame()

    if feature_df.empty:
        notes.append("No precomputed feature table was supplied. The app will build a feature table from raw time series.")
    else:
        feature_df = attach_characterization_features(feature_df, aggregated_char)

    return PreparedData(
        timeseries_df=timeseries_df,
        characterization_df=characterization_df,
        feature_df=feature_df,
        schema_timeseries=schema_timeseries,
        schema_characterization=schema_characterization,
        notes=notes,
    )
