from __future__ import annotations

import io
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER_NAME = "parallel_battery_app"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


@dataclass
class ColumnSchema:
    time_col: Optional[str]
    test_id_col: Optional[str]
    module_id_col: Optional[str]
    cell_current_cols: List[str]
    cell_temp_cols: List[str]
    module_current_col: Optional[str]
    module_voltage_col: Optional[str]
    soc_col: Optional[str]
    chemistry_col: Optional[str]
    ageing_col: Optional[str]
    operating_temp_col: Optional[str]
    interconnection_res_col: Optional[str]
    capacity_cols: List[str]
    resistance_cols: List[str]
    ocv_cols: List[str]


KEY_ALIASES: Dict[str, List[str]] = {
    "time": ["time", "timestamp", "t_s", "seconds", "elapsed_time", "step_time"],
    "test_id": ["test_id", "test", "experiment_id", "condition_id", "case_id", "run_id"],
    "module_id": ["module_id", "module", "string_id", "pack_id"],
    "module_current": ["module_current", "pack_current", "total_current", "current_module", "i_module"],
    "module_voltage": ["module_voltage", "pack_voltage", "total_voltage", "voltage_module", "v_module"],
    "soc": ["soc", "state_of_charge"],
    "chemistry": ["chemistry", "cell_chemistry", "chem", "chemistry_mix", "chemistry_combination"],
    "ageing": ["ageing", "aging", "age_status", "aged", "ageing_status"],
    "operating_temp": ["operating_temperature", "temperature_setpoint", "ambient_temperature", "test_temperature", "operating_temp", "temp_set"],
    "interconnection_res": ["interconnection_resistance", "branch_resistance", "interconnect_resistance", "r_interconnection", "interconnection_r_mohm"],
}


CELL_CURRENT_REGEXES = [
    r"^i[_\- ]?cell[_\- ]?\d+$",
    r"^cell[_\- ]?\d+[_\- ]?current$",
    r"^current[_\- ]?cell[_\- ]?\d+$",
    r"^branch[_\- ]?\d+[_\- ]?current$",
    r"^i[_\- ]?\d+$",
]

CELL_TEMP_REGEXES = [
    r"^t[_\- ]?cell[_\- ]?\d+$",
    r"^cell[_\- ]?\d+[_\- ]?temp(?:erature)?$",
    r"^temperature[_\- ]?cell[_\- ]?\d+$",
    r"^branch[_\- ]?\d+[_\- ]?temp(?:erature)?$",
]

CAPACITY_REGEXES = [r"capacity", r"discharge_capacity", r"cn", r"cap_ah"]
RESISTANCE_REGEXES = [r"resistance", r"ohmic", r"r0", r"r_0", r"internal_res"]
OCV_REGEXES = [r"ocv", r"open_circuit_voltage", r"pseudo_ocv"]


SAFE_EPS = 1e-12


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")



def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [slugify(c) for c in df.columns]
    return df



def is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)



def find_first_matching_column(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    normalized = {slugify(c): c for c in columns}
    for alias in aliases:
        alias_slug = slugify(alias)
        if alias_slug in normalized:
            return normalized[alias_slug]
    for col in columns:
        if any(alias_slug in slugify(col) for alias_slug in [slugify(a) for a in aliases]):
            return col
    return None



def find_regex_columns(columns: Sequence[str], regexes: Sequence[str]) -> List[str]:
    matches: List[str] = []
    for col in columns:
        col_slug = slugify(col)
        if any(re.search(pattern, col_slug) for pattern in regexes):
            matches.append(col)
    return sorted(matches)



def infer_schema(df: pd.DataFrame) -> ColumnSchema:
    cols = list(df.columns)
    time_col = find_first_matching_column(cols, KEY_ALIASES["time"])
    test_id_col = find_first_matching_column(cols, KEY_ALIASES["test_id"])
    module_id_col = find_first_matching_column(cols, KEY_ALIASES["module_id"])
    module_current_col = find_first_matching_column(cols, KEY_ALIASES["module_current"])
    module_voltage_col = find_first_matching_column(cols, KEY_ALIASES["module_voltage"])
    soc_col = find_first_matching_column(cols, KEY_ALIASES["soc"])
    chemistry_col = find_first_matching_column(cols, KEY_ALIASES["chemistry"])
    ageing_col = find_first_matching_column(cols, KEY_ALIASES["ageing"])
    operating_temp_col = find_first_matching_column(cols, KEY_ALIASES["operating_temp"])
    interconnection_res_col = find_first_matching_column(cols, KEY_ALIASES["interconnection_res"])

    cell_current_cols = find_regex_columns(cols, CELL_CURRENT_REGEXES)
    cell_temp_cols = find_regex_columns(cols, CELL_TEMP_REGEXES)

    capacity_cols = [c for c in cols if any(re.search(p, slugify(c)) for p in CAPACITY_REGEXES)]
    resistance_cols = [c for c in cols if any(re.search(p, slugify(c)) for p in RESISTANCE_REGEXES)]
    ocv_cols = [c for c in cols if any(re.search(p, slugify(c)) for p in OCV_REGEXES)]

    return ColumnSchema(
        time_col=time_col,
        test_id_col=test_id_col,
        module_id_col=module_id_col,
        cell_current_cols=cell_current_cols,
        cell_temp_cols=cell_temp_cols,
        module_current_col=module_current_col,
        module_voltage_col=module_voltage_col,
        soc_col=soc_col,
        chemistry_col=chemistry_col,
        ageing_col=ageing_col,
        operating_temp_col=operating_temp_col,
        interconnection_res_col=interconnection_res_col,
        capacity_cols=capacity_cols,
        resistance_cols=resistance_cols,
        ocv_cols=ocv_cols,
    )



def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



def safe_divide(a: Any, b: Any, fill_value: float = 0.0) -> Any:
    if isinstance(a, (pd.Series, pd.DataFrame, np.ndarray)) or isinstance(
        b, (pd.Series, pd.DataFrame, np.ndarray)
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.divide(a, np.where(np.abs(b) < SAFE_EPS, np.nan, b))
        if isinstance(out, np.ndarray):
            out = np.where(np.isfinite(out), out, fill_value)
        return out
    try:
        if abs(float(b)) < SAFE_EPS:
            return fill_value
        value = float(a) / float(b)
        return value if math.isfinite(value) else fill_value
    except Exception:
        return fill_value



def compute_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    try:
        coeffs = np.polyfit(x.astype(float), y.astype(float), 1)
        return float(coeffs[0])
    except Exception:
        return 0.0



def rolling_stats(series: pd.Series, window: int) -> Dict[str, float]:
    if series.empty:
        return {
            f"roll_mean_{window}": np.nan,
            f"roll_std_{window}": np.nan,
            f"roll_max_{window}": np.nan,
            f"roll_min_{window}": np.nan,
        }
    rolled = series.rolling(window=window, min_periods=max(1, window // 2))
    return {
        f"roll_mean_{window}": float(rolled.mean().iloc[-1]),
        f"roll_std_{window}": float(rolled.std().iloc[-1]),
        f"roll_max_{window}": float(rolled.max().iloc[-1]),
        f"roll_min_{window}": float(rolled.min().iloc[-1]),
    }



def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "missing", "missing_pct"])
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing": df.isna().sum().values,
            "missing_pct": (df.isna().mean() * 100.0).values,
        }
    ).sort_values(["missing", "missing_pct"], ascending=False)
    return summary



def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")



def dict_to_pretty_json(data: Dict[str, Any]) -> str:
    def default_serializer(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, indent=2, default=default_serializer, ensure_ascii=False)



def as_ordered_unique(seq: Iterable[Any]) -> List[Any]:
    seen: set = set()
    out: List[Any] = []
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out



def common_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> List[str]:
    candidate_keys = [
        "test_id",
        "experiment_id",
        "condition_id",
        "module_id",
        "chemistry",
        "ageing",
        "aging",
        "ageing_status",
        "operating_temperature",
        "ambient_temperature",
        "interconnection_resistance",
    ]
    return [key for key in candidate_keys if key in left.columns and key in right.columns]



def html_report(title: str, sections: Dict[str, str]) -> str:
    body = "\n".join(
        [f"<h2>{section}</h2>\n<div>{content}</div>" for section, content in sections.items()]
    )
    return f"""
    <html>
      <head>
        <meta charset='utf-8'>
        <title>{title}</title>
        <style>
          body {{ font-family: Arial, sans-serif; background: #111827; color: #f3f4f6; padding: 24px; }}
          h1, h2 {{ color: #60a5fa; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #374151; padding: 8px; }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        {body}
      </body>
    </html>
    """



def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
