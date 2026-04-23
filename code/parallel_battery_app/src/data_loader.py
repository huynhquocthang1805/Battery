from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import logger, normalize_columns, slugify

try:
    from scipy.io import loadmat  # type: ignore
    from scipy.io.matlab import mat_struct  # type: ignore
except Exception:
    loadmat = None
    mat_struct = None

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

    if any(k in combined for k in ["char", "capacity", "hppc", "ocv", "multisine", "single_cell"]):
        return "characterization"

    if "hall" in combined or "calibration" in combined:
        return "unknown"

    if table_name.lower() == "data" and (
        file_name.lower().startswith("m1_") or file_name.lower().startswith("m2_")
    ):
        return "timeseries"

    if any(k in combined for k in ["module", "timeseries", "test", "discharge"]):
        return "timeseries"

    return "unknown"


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for col in df.columns:
        if df[col].dtype == object:
            try:
                converted = pd.to_numeric(df[col], errors="raise")
                non_null_ratio = converted.notna().mean() if len(converted) else 0.0
                if non_null_ratio > 0.8:
                    df[col] = converted
            except Exception:
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


def _is_scalar_like(value: Any) -> bool:
    return isinstance(value, (str, bytes, int, float, complex, bool, np.number, np.bool_))


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return str(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        try:
            return _safe_scalar(value.reshape(-1)[0])
        except Exception:
            return str(value)
    if _is_scalar_like(value):
        return value.item() if hasattr(value, "item") and not isinstance(value, (str, bytes)) else value
    return value


def _object_to_rows(obj: Any, prefix: str = "", max_depth: int = 4) -> List[Dict[str, Any]]:
    if max_depth < 0:
        return [{prefix or "value": str(obj)}]

    if obj is None:
        return [{prefix or "value": None}]

    obj = _safe_scalar(obj)

    if _is_scalar_like(obj) or obj is pd.NA:
        return [{prefix or "value": obj}]

    if mat_struct is not None and isinstance(obj, mat_struct):
        field_names = getattr(obj, "_fieldnames", []) or []
        row: Dict[str, Any] = {}
        for field in field_names:
            value = getattr(obj, field)
            value = _safe_scalar(value)
            if _is_scalar_like(value):
                row[field] = value
            elif isinstance(value, np.ndarray) and value.ndim == 1 and value.size <= 50 and np.issubdtype(value.dtype, np.number):
                for i, item in enumerate(value.reshape(-1), start=1):
                    row[f"{field}_{i}"] = _safe_scalar(item)
            else:
                row[field] = str(value)
        return [row] if row else []

    if isinstance(obj, dict):
        row: Dict[str, Any] = {}
        for k, v in obj.items():
            v = _safe_scalar(v)
            if _is_scalar_like(v):
                row[str(k)] = v
            else:
                row[str(k)] = str(v)
        return [row] if row else []

    if isinstance(obj, np.ndarray):
        if obj.size == 0:
            return []
        if obj.dtype.names:
            rows = []
            flat = obj.reshape(-1)
            for rec in flat:
                row: Dict[str, Any] = {}
                for field in obj.dtype.names:
                    value = _safe_scalar(rec[field])
                    if _is_scalar_like(value):
                        row[field] = value
                    else:
                        row[field] = str(value)
                if row:
                    rows.append(row)
            return rows

        if obj.dtype == object:
            rows: List[Dict[str, Any]] = []
            flat = obj.reshape(-1)
            for idx, item in enumerate(flat):
                item_rows = _object_to_rows(item, prefix=f"{prefix}item" if prefix else "item", max_depth=max_depth - 1)
                for r in item_rows:
                    if len(flat) > 1 and "row_id" not in r:
                        r["row_id"] = idx
                    rows.append(r)
            return rows

        if np.issubdtype(obj.dtype, np.number):
            if obj.ndim == 1:
                return [{f"{prefix or 'value'}_{i+1}": _safe_scalar(v) for i, v in enumerate(obj.reshape(-1))}]
            if obj.ndim == 2:
                if obj.shape[0] > 1 and obj.shape[1] > 1:
                    rows = []
                    for row in obj:
                        rows.append({f"{prefix or 'value'}_{i+1}": _safe_scalar(v) for i, v in enumerate(row)})
                    return rows
                return [{f"{prefix or 'value'}_{i+1}": _safe_scalar(v) for i, v in enumerate(obj.reshape(-1))}]

        return [{prefix or "value": str(obj)}]

    if isinstance(obj, (list, tuple)):
        rows: List[Dict[str, Any]] = []
        for idx, item in enumerate(obj):
            item_rows = _object_to_rows(item, prefix=f"{prefix}item" if prefix else "item", max_depth=max_depth - 1)
            for r in item_rows:
                if len(obj) > 1 and "row_id" not in r:
                    r["row_id"] = idx
                rows.append(r)
        return rows

    return [{prefix or "value": str(obj)}]


def _rows_to_dataframe(rows: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not rows:
        return None
    try:
        df = pd.DataFrame(rows)
    except Exception:
        return None
    if df.empty:
        return None
    if df.shape[0] == 1 and df.shape[1] == 1 and isinstance(df.iloc[0, 0], str) and len(str(df.iloc[0, 0])) > 5000:
        return None
    return df


def _to_dataframe_from_any(name: str, value: Any) -> Optional[pd.DataFrame]:
    value = _safe_scalar(value)

    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        if value.ndim == 1:
            return pd.DataFrame({name: value.reshape(-1)})
        if value.ndim == 2:
            cols = [f"{slugify(name)}_{i+1}" for i in range(value.shape[1])]
            return pd.DataFrame(value, columns=cols)

    rows = _object_to_rows(value, prefix=slugify(name))
    return _rows_to_dataframe(rows)


def _extract_h5_item(h5obj: Any, root_file: "h5py.File", depth: int = 0) -> Any:
    if depth > 6:
        return None

    if h5py is None:
        return None

    if isinstance(h5obj, h5py.Dataset):
        data = h5obj[()]
        if isinstance(data, np.ndarray) and data.dtype.kind == "O":
            return [
                _extract_h5_item(root_file[ref], root_file, depth + 1)
                if isinstance(ref, h5py.Reference) and ref
                else None
                for ref in data.reshape(-1)
            ]
        if isinstance(data, np.ndarray) and data.dtype.kind == "u" and data.ndim >= 2:
            try:
                chars = "".join(chr(int(x)) for x in data.reshape(-1) if int(x) != 0)
                if chars:
                    return chars
            except Exception:
                pass
        return data

    if isinstance(h5obj, h5py.Group):
        result: Dict[str, Any] = {}
        for key in h5obj.keys():
            try:
                result[key] = _extract_h5_item(h5obj[key], root_file, depth + 1)
            except Exception:
                continue
        return result

    return None


def load_mat_file(path: Path) -> List[LoadedTable]:
    tables: List[LoadedTable] = []

    if loadmat is not None:
        try:
            content = loadmat(path, squeeze_me=True, struct_as_record=False)
            for key, value in content.items():
                if key.startswith("__"):
                    continue
                df = _to_dataframe_from_any(key, value)
                if df is not None and not df.empty:
                    tables.append(
                        LoadedTable(path.name, key, _role_hint_from_name(path.name, key), _clean_df(df))
                    )
        except Exception as exc:
            logger.warning("scipy loadmat failed for %s: %s", path.name, exc)

    if not tables and h5py is not None:
        try:
            with h5py.File(path, "r") as h5f:
                for key in h5f.keys():
                    try:
                        extracted = _extract_h5_item(h5f[key], h5f)
                        df = _to_dataframe_from_any(key, extracted)
                        if df is not None and not df.empty:
                            tables.append(
                                LoadedTable(path.name, key, _role_hint_from_name(path.name, key), _clean_df(df))
                            )
                    except Exception as exc:
                        logger.warning("Failed to parse MAT group %s in %s: %s", key, path.name, exc)
        except Exception as exc:
            logger.warning("h5py fallback failed for %s: %s", path.name, exc)

    return tables


def load_file(path: Path) -> List[LoadedTable]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv_file(path)
    if suffix in {".xlsx", ".xls"}:
        return load_excel_file(path)
    if suffix == ".mat":
        return load_mat_file(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def build_catalog(tables: List[LoadedTable]) -> pd.DataFrame:
    rows = []
    for table in tables:
        rows.append(
            {
                "source_file": table.source_file,
                "table_name": table.table_name,
                "role_hint": table.role_hint,
                "rows": len(table.df),
                "columns": len(table.df.columns),
                "column_names": ", ".join(map(str, table.df.columns[:15])),
            }
        )
    return pd.DataFrame(rows)


def load_dataset_bundle(input_path: str | Path) -> DatasetBundle:
    files = discover_files(input_path)
    tables: List[LoadedTable] = []
    errors: List[str] = []

    for file_path in files:
        try:
            loaded = load_file(file_path)
            if not loaded:
                logger.warning("No readable tables found in %s", file_path.name)
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
    if not tables:
        return pd.DataFrame()
    dfs = []
    for table in tables:
        df = table.df.copy()
        if add_source_cols:
            df["source_file"] = table.source_file
            df["source_table"] = table.table_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=False)
