from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .feature_engineering import build_risk_scores
from .modeling import load_model


@dataclass
class ScenarioPrediction:
    input_df: pd.DataFrame
    outputs: pd.DataFrame



def make_prediction(pipeline: Pipeline, input_df: pd.DataFrame) -> pd.DataFrame:
    preds = pipeline.predict(input_df)
    return pd.DataFrame({"prediction": np.ravel(preds)}, index=input_df.index)



def load_model_and_predict(model_path: str, input_df: pd.DataFrame) -> pd.DataFrame:
    pipeline = load_model(model_path)
    return make_prediction(pipeline, input_df)



def scenario_dataframe_from_controls(controls: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([controls])



def add_risk_outputs(feature_df: pd.DataFrame) -> pd.DataFrame:
    return build_risk_scores(feature_df)



def rule_based_recommendations(row: pd.Series) -> List[str]:
    recs: List[str] = []
    if row.get("interconnection_resistance", 0) is not None:
        try:
            if float(row.get("interconnection_resistance", 0)) > 1.0:
                recs.append("Giảm interconnection resistance hoặc cải thiện busbar / weld quality.")
        except Exception:
            pass
    chemistry = str(row.get("chemistry", "")).lower()
    if "mix" in chemistry or ("nmc" in chemistry and "nca" in chemistry):
        recs.append("Tránh chemistry mix nếu mục tiêu là tối thiểu hóa current/thermal gradient.")
    ageing = str(row.get("ageing", "")).lower()
    if any(token in ageing for token in ["aged", "true", "1", "yes"]):
        recs.append("Tránh mismatch aged/unaged trong cùng module khi có thể.")
    if pd.notna(row.get("degradation_risk_score")) and float(row["degradation_risk_score"]) >= 70:
        recs.append("Tăng kiểm soát nhiệt và giảm gradient nhiệt giữa các cell.")
    if not recs:
        recs.append("Scenario tương đối an toàn; tiếp tục xác minh bằng dữ liệu thực nghiệm hoặc digital twin.")
    return recs
