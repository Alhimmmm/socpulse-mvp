from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = [
    "program_id",
    "name",
    "region",
    "budget_plan",
    "budget_actual",
    "beneficiaries_plan",
    "beneficiaries_actual",
    "kpi_plan",
    "kpi_actual",
    "start_date",
    "end_date",
]


def _safe_ratio(actual: float, planned: float) -> float:
    if planned is None or pd.isna(planned) or float(planned) <= 0:
        return 0.0
    if actual is None or pd.isna(actual):
        return 0.0
    return max(float(actual) / float(planned), 0.0)


def _schedule_progress(start: pd.Timestamp, end: pd.Timestamp, today: date) -> float:
    if pd.isna(start) or pd.isna(end) or end <= start:
        return 1.0
    now = pd.Timestamp(today)
    if now <= start:
        return 0.0
    if now >= end:
        return 1.0
    return float((now - start) / (end - start))


def validate_programs(df: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def enrich_programs(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """Add normalized progress, risk score and status to source program data."""
    missing = validate_programs(df)
    if missing:
        raise ValueError("Не хватает столбцов: " + ", ".join(missing))

    result = df.copy()
    today = today or date.today()

    for column in ["start_date", "end_date"]:
        result[column] = pd.to_datetime(result[column], errors="coerce")

    numeric_columns: Iterable[str] = [
        "budget_plan",
        "budget_actual",
        "beneficiaries_plan",
        "beneficiaries_actual",
        "kpi_plan",
        "kpi_actual",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

    result["budget_progress"] = result.apply(
        lambda row: _safe_ratio(row["budget_actual"], row["budget_plan"]), axis=1
    )
    result["beneficiaries_progress"] = result.apply(
        lambda row: _safe_ratio(row["beneficiaries_actual"], row["beneficiaries_plan"]), axis=1
    )
    result["kpi_progress"] = result.apply(
        lambda row: _safe_ratio(row["kpi_actual"], row["kpi_plan"]), axis=1
    )
    result["schedule_progress"] = result.apply(
        lambda row: _schedule_progress(row["start_date"], row["end_date"], today), axis=1
    )
    result["delivery_progress"] = (
        result["beneficiaries_progress"] * 0.45 + result["kpi_progress"] * 0.55
    )
    result["lag"] = result["schedule_progress"] - result["delivery_progress"]
    result["budget_efficiency_gap"] = result["budget_progress"] - result["delivery_progress"]

    def risk_score(row: pd.Series) -> int:
        score = 0
        lag = float(row["lag"])
        budget_gap = float(row["budget_efficiency_gap"])
        schedule = float(row["schedule_progress"])
        kpi = float(row["kpi_progress"])

        if lag > 0.25:
            score += 3
        elif lag > 0.12:
            score += 2
        elif lag > 0.05:
            score += 1

        if budget_gap > 0.30 and row["budget_progress"] > 0.50:
            score += 2
        elif budget_gap > 0.18 and row["budget_progress"] > 0.40:
            score += 1

        if schedule > 0.75 and kpi < 0.60:
            score += 2
        elif schedule > 0.50 and kpi < 0.45:
            score += 1

        return score

    result["risk_score"] = result.apply(risk_score, axis=1)
    result["status"] = result["risk_score"].map(
        lambda score: "Высокий риск" if score >= 4 else "Требует внимания" if score >= 2 else "В норме"
    )
    return result


def portfolio_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {
            "programs": 0,
            "high_risk": 0,
            "attention": 0,
            "budget_plan": 0.0,
            "budget_actual": 0.0,
            "avg_delivery": 0.0,
        }
    return {
        "programs": int(len(df)),
        "high_risk": int((df["status"] == "Высокий риск").sum()),
        "attention": int((df["status"] == "Требует внимания").sum()),
        "budget_plan": float(df["budget_plan"].sum()),
        "budget_actual": float(df["budget_actual"].sum()),
        "avg_delivery": float(df["delivery_progress"].mean()),
    }


def program_context(row: pd.Series) -> dict[str, object]:
    """Compact, JSON-serializable context sent to GigaChat."""
    return {
        "program_id": str(row["program_id"]),
        "name": str(row["name"]),
        "region": str(row["region"]),
        "status": str(row["status"]),
        "risk_score": int(row["risk_score"]),
        "budget_plan": round(float(row["budget_plan"]), 2),
        "budget_actual": round(float(row["budget_actual"]), 2),
        "budget_progress_pct": round(float(row["budget_progress"]) * 100, 1),
        "beneficiaries_plan": round(float(row["beneficiaries_plan"]), 2),
        "beneficiaries_actual": round(float(row["beneficiaries_actual"]), 2),
        "beneficiaries_progress_pct": round(float(row["beneficiaries_progress"]) * 100, 1),
        "kpi_plan": round(float(row["kpi_plan"]), 2),
        "kpi_actual": round(float(row["kpi_actual"]), 2),
        "kpi_progress_pct": round(float(row["kpi_progress"]) * 100, 1),
        "schedule_progress_pct": round(float(row["schedule_progress"]) * 100, 1),
        "delivery_progress_pct": round(float(row["delivery_progress"]) * 100, 1),
        "lag_pct_points": round(float(row["lag"]) * 100, 1),
        "start_date": row["start_date"].date().isoformat() if not pd.isna(row["start_date"]) else None,
        "end_date": row["end_date"].date().isoformat() if not pd.isna(row["end_date"]) else None,
    }
