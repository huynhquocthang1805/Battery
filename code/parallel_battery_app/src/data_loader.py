from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import logger, normalize_columns, slugify

try:
    from scipy.io import loadmat  # type: ignore
except Exception:
    loadmat = None

try:
    import h5py  # type: ignore
except Exception:
    h5py = None

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".mat"}

@dataclass
class LoadedTable:
    source_file: str
    table_name: str
    role_hint: str
    df: pd.DataFrame

@dataclass
class DatasetBundle:
    tables: List[LoadedTable]
    catalog: pd.DataFrame
    errors: List[str]

    def table_dict(self) -> Dict[str, pd.DataFrame]:
        return {f"{t.source_file}::{t.table_name}": t.df for t in self.tables}

def discover_files(input_path: str | Path) -> List[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return [path]
    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files)

def _role_hint_from_name(file_name: str, table_name: str) -> str:
    combined = slugify(f"{file_name}_{table_name}")
    if "char" in combined or "capacity" in combined or "hppc" in combined or "ocv" in combined:
        return "characterization"
    if "time" in combined or "module" in combined or "test" in combined or "discharge" in combined:
        return "timeseries"
    return "unknown"

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # SỬA LỖI: Tránh FutureWarning của pd.to_numeric
    for col in df.columns:
        if df[col].dtype == object:
            try:
                # Chỉ ép kiểu nếu cột đó chứa dữ liệu số thực sự
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Nếu không phải số, giữ nguyên cột object
                continue
    return df

def load_csv_file(path: Path) -> List[LoadedTable]:
    df = pd.read_csv(path)
    df = _clean_df(df)
    return [LoadedTable(path.name, "csv", _role_hint_from_name(path.name, "csv"), df)]

def load_excel_file(path: Path) -> List[LoadedTable]:
    xls = pd.ExcelFile(path)
    tables: List[LoadedTable] = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name)
        df = _clean_df(df)
        if df.empty:
            continue
        tables.append(
            LoadedTable(
                source_file=path.name,
                table_name=sheet_name,
                role_hint=_role_hint_from_name(path.name, sheet_name),
                df=df,
            )
        )
    return tables

def _to_dataframe_from_numpy(name: str, arr: np.ndarray) -> Optional[pd.DataFrame]:
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.dtype.names:
            return pd.DataFrame({field: arr[field].tolist() for field in arr.dtype.names})
        return pd.DataFrame({name: arr.tolist()})
    if arr.ndim == 2:
        if arr.dtype.names:
            return pd.DataFrame({field: arr[field].tolist() for field in arr.dtype.names})
        if arr.shape[0] == 1 or arr.shape[1] == 1:
            return pd.DataFrame({name: arr.reshape(-1).tolist()})
        columns = [f"{slugify(name)}_{i+1}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=columns)
    return None

def load_mat_file(path: Path) -> List[LoadedTable]:
    tables: List[LoadedTable] = []
    # Thử Scipy trước (cho các file v7 trở xuống)
    if loadmat is not None:
        try:
            content = loadmat(path, squeeze_me=True, struct_as_record=False)
            for key, value in content.items():
                if key.startswith("__"): continue
                if isinstance(value, np.ndarray):
                    df = _to_dataframe_from_numpy(key, value)
                    if df is not None and not df.empty:
                        tables.append(
                            LoadedTable(path.name, key, _role_hint_from_name(path.name, key), _clean_df(df))
                        )
        except Exception as exc:
            # Im lặng nếu không phải định dạng loadmat hỗ trợ để tránh rác log
            pass

    # Thử h5py nếu Scipy thất bại (cho file v7.3)
    if not tables and h5py is not None:
        try:
            with h5py.File(path, "r") as h5f:
                for key in h5f.keys():
                    data = np.array(h5f[key])
                    df = _to_dataframe_from_numpy(key, data)
                    if df is not None and not df.empty:
                        tables.append(
                            LoadedTable(path.name, key, _role_hint_from_name(path.name, key), _clean_df(df))
                        )
        except Exception:
            pass # Không in cảnh báo fallback ra log nữa

    return tables

def load_file(path: Path) -> List[LoadedTable]:
    suffix = path.suffix.lower()
    if suffix == ".csv": return load_csv_file(path)
    if suffix in {".xlsx", ".xls"}: return load_excel_file(path)
    if suffix == ".mat": return load_mat_file(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")

def build_catalog(tables: List[LoadedTable]) -> pd.DataFrame:
    rows = []
    for table in tables:
        rows.append({
            "source_file": table.source_file,
            "table_name": table.table_name,
            "role_hint": table.role_hint,
            "rows": len(table.df),
            "columns": len(table.df.columns),
            "column_names": ", ".join(map(str, table.df.columns[:15])),
        })
    return pd.DataFrame(rows)

def load_dataset_bundle(input_path: str | Path) -> DatasetBundle:
    files = discover_files(input_path)
    tables: List[LoadedTable] = []
    errors: List[str] = []

    for file_path in files:
        try:
            loaded = load_file(file_path)
            if not loaded:
                errors.append(f"No readable tables found in {file_path.name}")
                continue
            tables.extend(loaded)
            logger.info("Loaded %s tables from %s", len(loaded), file_path.name)
        except Exception as exc:
            msg = f"Failed to load {file_path.name}: {exc}"
            logger.exception(msg)
            errors.append(msg)

    catalog = build_catalog(tables)
    return DatasetBundle(tables=tables, catalog=catalog, errors=errors)

def classify_tables(bundle: DatasetBundle) -> Dict[str, List[LoadedTable]]:
    grouped = {"timeseries": [], "characterization": [], "unknown": []}
    for table in bundle.tables:
        grouped.setdefault(table.role_hint, []).append(table)
    return grouped

def concat_tables(tables: List[LoadedTable], add_source_cols: bool = True) -> pd.DataFrame:
    if not tables: return pd.DataFrame()
    dfs = []
    for table in tables:
        df = table.df.copy()
        if add_source_cols:
            df["source_file"] = table.source_file
            df["source_table"] = table.table_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=False)