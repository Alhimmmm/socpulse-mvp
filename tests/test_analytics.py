from datetime import date

import pandas as pd

from src.analytics import enrich_programs, portfolio_metrics, program_context


def test_enrich_programs_marks_problematic_program():
    df = pd.DataFrame(
        [
            {
                "program_id": "X1",
                "name": "Тестовая программа",
                "region": "Регион",
                "budget_plan": 100,
                "budget_actual": 80,
                "beneficiaries_plan": 100,
                "beneficiaries_actual": 30,
                "kpi_plan": 100,
                "kpi_actual": 25,
                "start_date": "2026-01-01",
                "end_date": "2026-12-31",
            }
        ]
    )
    result = enrich_programs(df, today=date(2026, 9, 16))
    row = result.iloc[0]
    assert row["status"] == "Высокий риск"
    assert row["risk_score"] >= 4
    assert row["budget_efficiency_gap"] > 0


def test_portfolio_metrics_and_context():
    df = pd.DataFrame(
        [
            {
                "program_id": "X2",
                "name": "Стабильная программа",
                "region": "Регион",
                "budget_plan": 100,
                "budget_actual": 70,
                "beneficiaries_plan": 100,
                "beneficiaries_actual": 72,
                "kpi_plan": 100,
                "kpi_actual": 70,
                "start_date": "2026-01-01",
                "end_date": "2026-12-31",
            }
        ]
    )
    result = enrich_programs(df, today=date(2026, 9, 16))
    metrics = portfolio_metrics(result)
    assert metrics["programs"] == 1
    assert metrics["budget_plan"] == 100
    context = program_context(result.iloc[0])
    assert context["name"] == "Стабильная программа"
    assert "delivery_progress_pct" in context
