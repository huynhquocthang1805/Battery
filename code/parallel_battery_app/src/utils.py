from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("battery_app")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def slugify(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [slugify(c) for c in out.columns]
    return out


@dataclass
class ColumnSchema:
    time_col: Optional[str] = None
    module_current_col: Optional[str] = None
    module_voltage_col: Optional[str] = None
    cell_current_cols: List[str] | None = None
    cell_temp_cols: List[str] | None = None
    chemistry_col: Optional[str] = None
    ageing_col: Optional[str] = None
    operating_temp_col: Optional[str] = None
    interconnection_res_col: Optional[str] = None
    module_id_col: Optional[str] = None
    test_id_col: Optional[str] = None

    def __post_init__(self):
        self.cell_current_cols = self.cell_current_cols or []
        self.cell_temp_cols = self.cell_temp_cols or []


TIME_CANDIDATES = [
    "elapsed_time",
    "test_time_s",
    "timedata",
    "time_s",
    "time",
    "test_time",
    "step_time_s",
]


def infer_schema(df: pd.DataFrame) -> ColumnSchema:
    cols = list(df.columns)
    schema = ColumnSchema()
    for c in TIME_CANDIDATES:
        if c in cols:
            schema.time_col = c
            break
    if schema.time_col is None and "date_time" in cols:
        schema.time_col = "date_time"

    for c in ["module_current", "current_a", "currentdata", "current"]:
        if c in cols:
            schema.module_current_col = c
            break
    for c in ["module_voltage", "voltage_v", "voltagedata", "voltage"]:
        if c in cols:
            schema.module_voltage_col = c
            break

    schema.cell_current_cols = sorted([c for c in cols if re.fullmatch(r"i_cell_\d+", c) or re.fullmatch(r"current_a_cell_?\d+", c)])
    schema.cell_temp_cols = sorted([c for c in cols if re.fullmatch(r"t_cell_\d+", c) or re.fullmatch(r"temp_cell_\d+", c) or re.fullmatch(r"cell_temperature_\d+", c)])

    for c in ["chemistry", "chemistry_combination"]:
        if c in cols:
            schema.chemistry_col = c
            break
    for c in ["ageing", "aging", "ageing_status", "aging_status"]:
        if c in cols:
            schema.ageing_col = c
            break
    for c in ["operating_temperature", "ambient_temperature", "test_temperature"]:
        if c in cols:
            schema.operating_temp_col = c
            break
    for c in ["interconnection_resistance", "branch_resistance", "r0_condition"]:
        if c in cols:
            schema.interconnection_res_col = c
            break
    for c in ["module_id"]:
        if c in cols:
            schema.module_id_col = c
            break
    for c in ["test_id", "experiment_id", "synthetic_test_id"]:
        if c in cols:
            schema.test_id_col = c
            break
    return schema


def common_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> List[str]:
    priority = ["test_id", "module_id", "chemistry", "ageing", "source_file", "source_table"]
    return [c for c in priority if c in left.columns and c in right.columns]


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "missing", "missing_pct"])
    out = pd.DataFrame({
        "column": df.columns,
        "missing": df.isna().sum().values,
        "missing_pct": (df.isna().mean().values * 100.0),
    })
    return out.sort_values(["missing", "column"], ascending=[False, True]).reset_index(drop=True)


def html_report(title: str, sections: dict[str, str]) -> str:
    body = "".join([f"<h2>{k}</h2>{v}" for k, v in sections.items()])
    return f"""
    <html>
    <head>
      <meta charset='utf-8'/>
      <title>{title}</title>
      <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; }}
      table {{ border-collapse: collapse; width: 100%; }}
      td, th {{ border: 1px solid #ccc; padding: 6px; }}
      h1, h2 {{ color: #1f2937; }}
      </style>
    </head>
    <body><h1>{title}</h1>{body}</body></html>
    """
