from __future__ import annotations

import pandas as pd


def scenario_dataframe_from_controls(controls: dict) -> pd.DataFrame:
    return pd.DataFrame([controls])


def rule_based_recommendations(row: pd.Series) -> list[str]:
    recs: list[str] = []
    if float(row.get("interconnection_resistance", 0) or 0) > 1.5:
        recs.append("Reduce interconnection resistance to improve current sharing.")
    if str(row.get("chemistry", "")).lower() in {"mix", "mixed"}:
        recs.append("Avoid chemistry mixing when possible to reduce imbalance and thermal mismatch.")
    if float(row.get("operating_temperature", row.get("ambient_temperature", 0)) or 0) >= 35:
        recs.append("Increase thermal control or cooling at high operating temperature.")
    if str(row.get("ageing", "")).lower() == "aged":
        recs.append("Avoid aged cell mismatch in parallel modules.")
    if float(row.get("degradation_risk_score", 0) or 0) >= 70:
        recs.append("High-risk scenario: prioritize safer operating window and tighter cell matching.")
    if not recs:
        recs.append("Scenario looks acceptable under the current proxy model.")
    return recs
